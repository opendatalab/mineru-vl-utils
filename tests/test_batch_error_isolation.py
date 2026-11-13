"""验证普通批量容错、严格错误及四个后端的公开开关传递。"""

import asyncio
import importlib
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from mineru_vl_utils import MinerUClient
from mineru_vl_utils.vlm_client.base_client import (
    EngineUnavailableError,
    HttpResponseError,
    UnsupportedError,
    VlmClient,
    new_vlm_client,
)
from mineru_vl_utils.vlm_client.http_client import HttpVlmClient
from mineru_vl_utils.vlm_client.llama_cpp_engine_client import LlamaCppEngineVlmClient
from mineru_vl_utils.vlm_client.lmdeploy_engine_client import LmdeployEngineVlmClient
from mineru_vl_utils.vlm_client.vllm_async_engine_client import VllmAsyncEngineVlmClient


ASYNC_CLIENTS = [HttpVlmClient, LlamaCppEngineVlmClient, LmdeployEngineVlmClient, VllmAsyncEngineVlmClient]


def _async_client(cls, enabled, predict):
    """用真实批量调度器运行受控的单块预测，不加载可选模型。"""
    client = object.__new__(cls)
    VlmClient.__init__(client, isolate_block_errors=enabled)
    client.max_concurrency = 3
    client.use_tqdm = False
    client.aio_predict = predict
    return client


@pytest.mark.parametrize("cls", ASYNC_CLIENTS)
@pytest.mark.parametrize("enabled", [False, True])
def test_async_partial_failure_preserves_order(cls, enabled):
    """同一个局部解码错误默认抛出，开启后仅留空对应输入。"""
    error = UnicodeDecodeError("utf-8", b"\xe9", 0, 1, "unexpected end of data")
    calls = []

    async def predict(image, prompt, **kwargs):
        """用不同完成顺序验证结果索引，而非任务完成顺序。"""
        calls.append(prompt)
        if prompt == "bad":
            raise error
        await asyncio.sleep(0.01 if prompt == "first" else 0)
        return prompt

    client = _async_client(cls, enabled, predict)
    if enabled:
        assert asyncio.run(client.aio_batch_predict([None] * 3, ["first", "bad", "last"])) == ["first", "", "last"]
        assert sorted(calls) == ["bad", "first", "last"]
    else:
        with pytest.raises(UnicodeDecodeError) as raised:
            asyncio.run(client.aio_batch_predict([None] * 3, ["first", "bad", "last"]))
        assert raised.value is error


@pytest.mark.parametrize("cls", ASYNC_CLIENTS)
@pytest.mark.parametrize("valid_empty", [False, True])
def test_all_failed_raises_first_input_error_but_empty_success_counts(cls, valid_empty):
    """全部失败重抛输入零的错误；合法空文本不能被误判成失败。"""
    first_error = ValueError("first input")

    async def predict(image, prompt, **kwargs):
        """让第二项先失败，检验最终错误的输入顺序稳定性。"""
        if prompt == "first":
            await asyncio.sleep(0.01)
            raise first_error
        if valid_empty:
            return ""
        raise ValueError("second input")

    client = _async_client(cls, True, predict)
    if valid_empty:
        assert asyncio.run(client.aio_batch_predict([None, None], ["first", "second"])) == ["", ""]
    else:
        with pytest.raises(ValueError) as raised:
            asyncio.run(client.aio_batch_predict([None, None], ["first", "second"]))
        assert raised.value is first_error


@pytest.mark.parametrize("cls", ASYNC_CLIENTS)
@pytest.mark.parametrize(
    "error",
    [
        MemoryError("oom"),
        TypeError("program error"),
        AttributeError("program error"),
        AssertionError("invalid input"),
        UnsupportedError("unsupported"),
        KeyError("program error"),
        IndexError("program error"),
        RuntimeError("engine stopped"),
        EngineUnavailableError("engine unavailable"),
    ],
)
def test_fatal_error_is_never_isolated(cls, error):
    """即使有成功结果，内存、编程及不支持的操作错误也必须抛出。"""

    async def predict(image, prompt, **kwargs):
        """同批安排一个成功请求和一个不可降级错误。"""
        if prompt == "bad":
            raise error
        return "success"

    client = _async_client(cls, True, predict)
    with pytest.raises(type(error)) as raised:
        asyncio.run(client.aio_batch_predict([None, None], ["good", "bad"]))
    assert raised.value is error


@pytest.mark.parametrize("cls", ASYNC_CLIENTS)
def test_isolation_cancellation_drains_requests(cls):
    """取消启用容错的批次仍要传播取消并等待所有子任务清理。"""

    async def run():
        """精确控制两个在途请求的启动和退出。"""
        entered = asyncio.Event()
        started, cleaned = [], []

        async def predict(image, prompt, **kwargs):
            """记录取消触发的清理，不把取消转换为空文本。"""
            started.append(prompt)
            if len(started) == 2:
                entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.append(prompt)

        client = _async_client(cls, True, predict)
        task = asyncio.create_task(client.aio_batch_predict([None, None], ["a", "b"]))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(cleaned) == ["a", "b"]

    asyncio.run(run())


@pytest.mark.parametrize("cls", ASYNC_CLIENTS)
def test_empty_and_invalid_priority_batches(cls):
    """空批次返回空列表，长度错误仍在调度之前抛出。"""

    async def predict(**kwargs):
        """不应在空批次或长度验证失败时执行。"""
        raise AssertionError("must not be called")

    client = _async_client(cls, True, predict)
    assert asyncio.run(client.aio_batch_predict([])) == []
    with pytest.raises(AssertionError, match="priority"):
        asyncio.run(client.aio_batch_predict([None], priority=[]))


@pytest.mark.parametrize("status", [400, 413, 422, 401, 403, 404, 407, 429, 500, 503])
@pytest.mark.parametrize("use_async", [False, True])
def test_http_status_classification_and_single_request_strictness(monkeypatch, status, use_async):
    """通过真实响应解析检查局部请求错误与认证、配额、服务错误的分类。"""
    image = Image.new("RGB", (4, 4))

    def respond(request):
        """根据提示词返回一个正常响应或指定 HTTP 错误。"""
        import json

        if "bad" in json.loads(request.content)["messages"][-1]["content"][-1]["text"]:
            return httpx.Response(status, text="synthetic error")
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "good"}}]})

    async def new_aio_client(self):
        """使用无网络传输测试真实异步请求和同步桥接。"""
        return httpx.AsyncClient(transport=httpx.MockTransport(respond))

    monkeypatch.setattr(HttpVlmClient, "_new_aio_client", new_aio_client)
    client = HttpVlmClient(
        server_url="http://test", model_name="test", skip_model_name_checking=True, isolate_block_errors=True, use_tqdm=False
    )

    async def run():
        """同一事件循环内验证批量、单次和客户端关闭语义。"""
        try:
            if status in {400, 413, 422}:
                assert await client.aio_batch_predict([image, image], ["good", "bad"]) == ["good", ""]
            else:
                with pytest.raises(HttpResponseError) as raised:
                    await client.aio_batch_predict([image, image], ["good", "bad"])
                assert raised.value.status_code == status
            with pytest.raises(HttpResponseError):
                await client.aio_predict(image, "bad")
        finally:
            await client.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await client.aio_batch_predict([image, image])

    if use_async:
        asyncio.run(run())
    else:
        try:
            if status in {400, 413, 422}:
                assert client.batch_predict([image, image], ["good", "bad"]) == ["good", ""]
            else:
                with pytest.raises(HttpResponseError):
                    client.batch_predict([image, image], ["good", "bad"])
        finally:
            asyncio.run(client.aclose())


@pytest.mark.parametrize("enabled", [False, True])
def test_llama_cpp_sync_and_single_prediction(enabled):
    """同步线程池执行真实客户端预测；单次错误不受批量开关影响。"""
    native = pytest.importorskip("mineru_llama_cpp")
    engine = object.__new__(native.Engine)
    error = ValueError("bad block")

    def generate(messages, params):
        """按真实消息中的提示词控制原生替身结果。"""
        prompt = messages[-1]["content"][-1]["text"]
        if prompt == "bad":
            raise error
        return native.GenerateResult(content=prompt, finish_reason="stop", tokens_evaluated=1, tokens_predicted=1, timings=None)

    engine.generate = generate
    client = LlamaCppEngineVlmClient(engine, isolate_block_errors=enabled, use_tqdm=False)
    if enabled:
        assert client.batch_predict([None, None], ["good", "bad"]) == ["good", ""]
    else:
        with pytest.raises(ValueError):
            client.batch_predict([None, None], ["good", "bad"])
    with pytest.raises(ValueError) as raised:
        client.batch_predict([None, None], ["bad", "bad"])
    assert raised.value is error
    with pytest.raises(ValueError):
        client.predict(None, "bad")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("use_async", [False, True])
def test_lmdeploy_real_thread_path_preserves_single_strictness(enabled, use_async):
    """使用真实 LMDeploy 适配路径与可观测 Pipeline 替身检查同步桥接。"""
    client = object.__new__(LmdeployEngineVlmClient)
    VlmClient.__init__(client, isolate_block_errors=enabled)
    client.max_concurrency, client.batch_size, client.model_max_length = 2, 0, 128
    client.use_tqdm = False
    client.LmdeployGenerationConfig = SimpleNamespace
    calls = []

    def infer(prompts, **kwargs):
        """记录每次 Pipeline 请求，并仅让指定块抛出可降级错误。"""
        calls.append(list(prompts))
        if "bad" in prompts:
            raise ValueError("bad block")
        return [SimpleNamespace(text=p, finish_reason="stop") for p in prompts]

    client.lmdeploy_engine = SimpleNamespace(infer=infer)
    if enabled:
        output = (
            client.aio_batch_predict([None, None], ["good", "bad"])
            if use_async
            else client.batch_predict([None, None], ["good", "bad"])
        )
        assert (asyncio.run(output) if use_async else output) == ["good", ""]
        assert sorted(calls) == [["bad"], ["good"]]
    else:
        with pytest.raises(ValueError):
            if use_async:
                asyncio.run(client.aio_batch_predict([None, None], ["good", "bad"]))
            else:
                client.batch_predict([None, None], ["good", "bad"])
    with pytest.raises(ValueError):
        client.predict(None, "bad")


def test_vllm_real_generation_and_scored_batch_remain_strict(native_errors):
    """普通批量经真实 generate 解析降级，评分批量仍传播原异常。"""
    calls = []
    client = object.__new__(VllmAsyncEngineVlmClient)
    VlmClient.__init__(client, isolate_block_errors=True)
    client.max_concurrency, client.model_max_length = 2, 128
    client.VllmSamplingParams = SimpleNamespace
    client.VllmRequestOutputKind = SimpleNamespace(FINAL_ONLY="final")
    client.tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: messages[-1]["content"][-1]["text"])
    client.debug = False

    async def generate(prompt, **kwargs):
        """记录请求及流清理；指定坏块通过真实异步生成器抛错。"""
        text = prompt["prompt"]
        calls.append(text)
        if text == "bad":
            raise ValueError("bad block")
        yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(finish_reason="stop", text=text)])

    client.vllm_async_llm = SimpleNamespace(generate=generate)
    assert asyncio.run(client.aio_batch_predict([None, None], ["good", "bad"])) == ["good", ""]
    assert sorted(calls) == ["bad", "good"]
    with pytest.raises(ValueError):
        asyncio.run(client.aio_predict(None, "bad"))

    async def score(image, prompt, **kwargs):
        """评分路径用原异常验证其严格契约。"""
        raise ValueError("score failed")

    client.aio_predict_scored = score
    with pytest.raises(ValueError, match="score failed"):
        asyncio.run(client.aio_batch_predict_scored([None]))


@pytest.mark.parametrize(
    "backend,module_name,class_name",
    [
        ("http-client", "http_client", "HttpVlmClient"),
        ("llama-cpp-engine", "llama_cpp_engine_client", "LlamaCppEngineVlmClient"),
        ("lmdeploy-engine", "lmdeploy_engine_client", "LmdeployEngineVlmClient"),
        ("vllm-async-engine", "vllm_async_engine_client", "VllmAsyncEngineVlmClient"),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_factory_and_mineru_client_forward_isolation(monkeypatch, backend, module_name, class_name, enabled):
    """公开高层入口和工厂均将开关交给正确的后端，不依赖模型安装。"""
    seen = []

    def constructor(**kwargs):
        """捕获真实工厂传递的关键字，避免启动模型。"""
        seen.append(kwargs["isolate_block_errors"])
        return SimpleNamespace()

    module = importlib.import_module("mineru_vl_utils.vlm_client." + module_name)
    monkeypatch.setattr(module, class_name, constructor)
    new_vlm_client(backend, isolate_block_errors=enabled)
    MinerUClient(
        backend=backend,
        isolate_block_errors=enabled,
        sampling_params={},  # 本测试只核对错误隔离透传，不构造后端采样默认值。
        model=object(),
        processor=object(),
        llama_cpp_engine=object(),
        lmdeploy_engine=object(),
        vllm_async_llm=object(),
    )
    assert seen == [enabled, enabled]


@pytest.mark.parametrize("backend", ["transformers", "mlx-engine", "vllm-engine"])
def test_unsupported_backend_rejected_before_model_loading(backend):
    """显式开启不支持的后端必须在模型加载前给出清晰异常。"""
    with pytest.raises(UnsupportedError, match="isolate_block_errors"):
        new_vlm_client(backend, isolate_block_errors=True)
    with pytest.raises(UnsupportedError, match="isolate_block_errors"):
        MinerUClient(backend=backend, isolate_block_errors=True)


@pytest.mark.parametrize("valid_empty", [False, True])
def test_sync_llama_all_failed_uses_input_order(valid_empty):
    """线程池先完成第二项时仍按输入顺序选错，并承认合法空文本成功。"""
    import time

    first_error = ValueError("first input")
    client = object.__new__(LlamaCppEngineVlmClient)
    VlmClient.__init__(client, isolate_block_errors=True)
    client.max_concurrency, client.use_tqdm = 2, False

    def predict(image, prompt, sampling_params=None, priority=None):
        """使用有限延迟构造不同于输入顺序的完成顺序。"""
        if prompt == "first":
            time.sleep(0.02)
            raise first_error
        if valid_empty:
            return ""
        raise ValueError("second input")

    client.predict = predict
    if valid_empty:
        assert client.batch_predict([None, None], ["first", "second"]) == ["", ""]
    else:
        with pytest.raises(ValueError) as raised:
            client.batch_predict([None, None], ["first", "second"])
        assert raised.value is first_error


def test_lmdeploy_isolated_batch_cancellation_keeps_thread_lease():
    """可选隔离不能在 Pipeline 同步线程结束前释放共享引擎租约。"""
    import threading

    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    client = object.__new__(LmdeployEngineVlmClient)
    VlmClient.__init__(client, isolate_block_errors=True)
    client.max_concurrency, client.batch_size, client.model_max_length = 1, 0, 128
    client.use_tqdm = False
    client.LmdeployGenerationConfig = SimpleNamespace

    def infer(prompts, **kwargs):
        """用事件控制真实适配器中的在途线程。"""
        entered.set()
        try:
            assert release.wait(3)
            return [SimpleNamespace(text="done", finish_reason="stop")]
        finally:
            exited.set()

    client.lmdeploy_engine = SimpleNamespace(infer=infer)

    async def run():
        """重复取消之后仍等待线程退出，并释放占用的并发槽。"""
        semaphore = asyncio.Semaphore(1)
        task = asyncio.create_task(client.aio_batch_predict([None], semaphore=semaphore))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            assert semaphore.locked()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()
        assert not semaphore.locked()

    asyncio.run(run())


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("use_async", [False, True])
def test_mineru_layout_extraction_keeps_successful_blocks(monkeypatch, enabled, use_async):
    """真实高层裁剪和后处理链路必须保留正常块，默认仍传播坏块错误。"""
    import base64
    import io
    import json
    from mineru_vl_utils.structs import ContentBlock

    def respond(request):
        """解码实际裁剪图像，仅拒绝黑色区域的块请求。"""
        parts = json.loads(request.content)["messages"][-1]["content"]
        url = next(part["image_url"]["url"] for part in parts if part["type"] == "image_url")
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as crop:
            if crop.getpixel((0, 0)) == (0, 0, 0):
                return httpx.Response(422, text="bad image block")
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "recognized"}}]})

    async def new_aio_client(self):
        """为真实高层请求注入无需服务器的传输。"""
        return httpx.AsyncClient(transport=httpx.MockTransport(respond))

    monkeypatch.setattr(HttpVlmClient, "_new_aio_client", new_aio_client)
    client = MinerUClient(
        backend="http-client",
        server_url="http://test",
        model_name="test",
        skip_model_name_checking=True,
        isolate_block_errors=enabled,
        use_tqdm=False,
    )
    image = Image.new("RGB", (10, 10), "white")
    image.paste((0, 0, 0), (5, 0, 10, 10))
    blocks = [[ContentBlock("text", [0, 0, 0.5, 1]), ContentBlock("text", [0.5, 0, 1, 1])]]

    async def run():
        """在拥有连接池的事件循环中完成提取和显式关闭。"""
        try:
            return await client.aio_batch_extract_with_layout([image], blocks)
        finally:
            await client.aclose()

    try:
        if not enabled:
            with pytest.raises(HttpResponseError):
                if use_async:
                    asyncio.run(run())
                else:
                    client.batch_extract_with_layout([image], blocks)
        else:
            result = asyncio.run(run()) if use_async else client.batch_extract_with_layout([image], blocks)
            assert [block.content for block in result[0]] == ["recognized", ""]
    finally:
        asyncio.run(client.aclose())
