"""验证 Transformers 5 与 LMDeploy Pipeline 的调用、批量及异步契约。"""

from __future__ import annotations

import asyncio
import sys
import subprocess
import threading
from types import ModuleType, SimpleNamespace
from collections.abc import Iterator
from typing import Any

import pytest

from mineru_vl_utils.vlm_client.base_client import SamplingParams, ServerError
from mineru_vl_utils.vlm_client.lmdeploy_engine_client import LmdeployEngineVlmClient


def test_importing_vllm_client_does_not_patch_or_import_engine() -> None:
    """客户端模块导入不得尝试加载 vLLM 或安装已失效的全局 logprobs 补丁。"""
    code = """
import importlib.abc
import sys
attempts = []

class CheckImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # 记录被旧补丁吞掉的导入错误，确保导入期间完全没有引擎副作用。
        if fullname.split('.')[0] == 'vllm':
            attempts.append(fullname)
            raise ImportError(fullname)

sys.meta_path.insert(0, CheckImports())
from mineru_vl_utils.vlm_client.vllm_engine_client import VllmEngineVlmClient
assert not attempts, attempts
"""
    process = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
    assert process.returncode == 0, process.stderr


@pytest.fixture
def pipeline_type(monkeypatch: pytest.MonkeyPatch) -> type:
    """提供公开 Pipeline 契约替身，不需要 CUDA 或 LMDeploy 二进制安装。"""

    class GenerationConfig(SimpleNamespace):
        """保存真实客户端传给 LMDeploy 的生成参数。"""

    class Pipeline:
        """模拟可从多线程调用的 Pipeline，记录并发数和参数。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """构造已解析的 backend_config 和线程安全计数器。"""
            self.backend_config = SimpleNamespace(session_len=128)
            self.calls: list[dict[str, Any]] = []
            self.active = 0
            self.peak = 0
            self.lock = threading.Lock()
            self.error = False

        def infer(self, prompts: list[Any], *, gen_config: list[Any], **kwargs: Any) -> list[Any]:
            """模拟阻塞推理并返回保持输入顺序的 Response。"""
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.calls.append({"prompts": prompts, "gen_config": gen_config, "thread": threading.get_ident(), **kwargs})
            try:
                threading.Event().wait(0.03)
                if self.error:
                    return [SimpleNamespace(text="", finish_reason="error") for _ in prompts]
                return [
                    SimpleNamespace(text=prompt[0] if isinstance(prompt, tuple) else prompt, finish_reason="stop")
                    for prompt in prompts
                ]
            finally:
                with self.lock:
                    self.active -= 1

        def stream_infer(self, prompts: list[Any], *, gen_config: list[Any], stream_response: bool, **kwargs: Any) -> Iterator[Any]:
            """模拟完整响应流，倒序交付以检查结果索引回填。"""
            responses = self.infer(prompts, gen_config=gen_config, stream_response=stream_response, **kwargs)
            for index in reversed(range(len(responses))):
                responses[index].index = index
                yield responses[index]

    module = ModuleType("lmdeploy")
    module.GenerationConfig = GenerationConfig
    module.pipeline = Pipeline
    pipeline_module = ModuleType("lmdeploy.pipeline")
    pipeline_module.Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, "lmdeploy", module)
    monkeypatch.setitem(sys.modules, "lmdeploy.pipeline", pipeline_module)
    return Pipeline


@pytest.mark.parametrize("enabled", [False, True])
def test_pipeline_preserves_batch_order_priority_and_sampling(pipeline_type: type, enabled: bool) -> None:
    """不同优先级请求可以拆批，但必须保持结果顺序及各自生成参数。"""
    pipeline = pipeline_type()
    client = LmdeployEngineVlmClient(pipeline, batch_size=3, use_tqdm=enabled)
    outputs = client.batch_predict(
        [None] * 4,
        ["a", "b", "c", "d"],
        [SamplingParams(max_new_tokens=i + 1) for i in range(4)],
        priority=[3, 3, 1, 3],
    )
    assert outputs == ["a", "b", "c", "d"]
    assert [call["priority"] for call in pipeline.calls] == [3, 1, 3]
    assert [config.max_new_tokens for call in pipeline.calls for config in call["gen_config"]] == [1, 2, 3, 4]
    assert all(not config.skip_special_tokens for call in pipeline.calls for config in call["gen_config"])


def test_pipeline_async_calls_use_threads_and_limit_concurrency(pipeline_type: type) -> None:
    """异步批量接口不得阻塞事件循环，并保持并发限制和请求顺序。"""
    pipeline = pipeline_type()
    client = LmdeployEngineVlmClient(pipeline, max_concurrency=2, use_tqdm=False)
    main_thread = threading.get_ident()

    async def run() -> list[str]:
        """使用真实 asyncio 调度检查非阻塞适配。"""
        return await client.aio_batch_predict([None] * 5, [str(i) for i in range(5)], priority=4)

    assert asyncio.run(run()) == [str(i) for i in range(5)]
    assert pipeline.peak == 2
    assert all(call["thread"] != main_thread and call["priority"] == 4 for call in pipeline.calls)


def test_pipeline_errors_reach_sync_and_async_callers(pipeline_type: type) -> None:
    """后端错误响应不能被转换成成功的空字符串。"""
    pipeline = pipeline_type()
    pipeline.error = True
    client = LmdeployEngineVlmClient(pipeline, use_tqdm=False)
    with pytest.raises(ServerError, match="LMDeploy inference failed"):
        client.predict(None, "x")
    with pytest.raises(ServerError, match="LMDeploy inference failed"):
        asyncio.run(client.aio_predict(None, "x"))


def test_mineru_client_constructs_public_pipeline(pipeline_type: type) -> None:
    """仅传模型路径的公共入口也必须构造新版 Pipeline。"""
    from mineru_vl_utils import MinerUClient

    client = MinerUClient(backend="lmdeploy-engine", model_path="local-model")
    assert isinstance(client.client.lmdeploy_engine, pipeline_type)


def test_pipeline_cancellation_waits_for_inflight_worker(pipeline_type: type) -> None:
    """传播取消前必须完成不可中断的同步调用，防止引擎卸载后仍有线程使用它。"""
    pipeline = pipeline_type()
    client = LmdeployEngineVlmClient(pipeline, max_concurrency=1, use_tqdm=False)

    async def cancel() -> None:
        """在工作线程执行期间取消，并检查没有遗留在途调用。"""
        task = asyncio.create_task(client.aio_predict(None, "x"))
        while pipeline.active == 0:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pipeline.active == 0

    asyncio.run(cancel())


def test_transformers_uses_text_config_and_all_special_token_ids() -> None:
    """使用实际 Qwen2-VL 配置验证嵌套参数和多 EOS 过滤，再经过完整客户端批量调用。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers", minversion="5.10.1")
    from transformers import Qwen2VLConfig
    from transformers.feature_extraction_utils import BatchFeature
    from mineru_vl_utils.vlm_client.transformers_client import TransformersVlmClient

    class Model:
        """保留真实配置的生成替身，记录送入 generate 的参数。"""

        def __init__(self) -> None:
            """让所有特殊 token 来源不同，避免测试因字段重复而漏检。"""
            self.config = Qwen2VLConfig(
                text_config={"max_position_embeddings": 32, "bos_token_id": 10, "eos_token_id": [11, 12], "pad_token_id": 13}
            )
            self.generation_config = SimpleNamespace(eos_token_id=[14, 15], bos_token_id=16, pad_token_id=18)
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.kwargs: dict[str, Any] = {}

        def generate(self, **kwargs: Any) -> Any:
            """返回包含普通 token 和所有特殊 token 的生成序列。"""
            self.kwargs = kwargs
            suffix = torch.tensor([[17, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22]])
            return torch.cat([kwargs["input_ids"], suffix.repeat(kwargs["input_ids"].shape[0], 1)], dim=1)

    class Processor:
        """提供真实 BatchFeature 的处理器替身，保留整数 input_ids 的搬运语义。"""

        tokenizer = SimpleNamespace(bos_token_id=20, eos_token_id=21, pad_token_id=22)

        def apply_chat_template(self, messages: list[Any], **kwargs: Any) -> str:
            """返回占位 prompt，实际参数流仍经过客户端。"""
            return "prompt"

        def __call__(self, *, text: list[str], **kwargs: Any) -> Any:
            """构造两个输入 token，检验客户端对生成前缀的截取。"""
            return BatchFeature(
                {"input_ids": torch.tensor([[1, 2]] * len(text)), "attention_mask": torch.ones(len(text), 2, dtype=torch.long)}
            )

        def batch_decode(self, ids: list[list[int]], **kwargs: Any) -> list[str]:
            """仅把保留的 token 转为字符串，方便检查过滤后的精确序列。"""
            return [",".join(str(value) for value in row) for row in ids]

    model = Model()
    client = TransformersVlmClient(model, Processor(), batch_size=2, use_tqdm=False)
    assert not hasattr(model.config, "max_position_embeddings")
    assert client.model_max_length == 32
    assert client.batch_predict([None, None], ["a", "b"]) == ["17", "17"]
    assert model.kwargs["use_cache"] is True
    assert model.kwargs["max_length"] == 32
    assert model.kwargs["input_ids"].dtype == torch.long


def test_processor_backend_is_configured_only_on_image_component(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 backend 不得传给 AutoProcessor 的视频组件，同时保留 tokenizer 和 chat template。"""
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    from mineru_vl_utils.transformers_loading import load_transformers_processor

    processor = SimpleNamespace(image_processor=None, tokenizer=object(), chat_template="template", video_processor=object())
    image_processor = object()
    calls = []

    def load_processor(path: str, **kwargs: Any) -> Any:
        """模拟会拒绝全局 backend 的复合处理器加载。"""
        assert "backend" not in kwargs
        return processor

    def load_image_processor(path: str, **kwargs: Any) -> Any:
        """记录只传给图像组件的后端配置。"""
        calls.append((path, kwargs))
        return image_processor

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", load_image_processor)
    assert load_transformers_processor("model") is processor
    assert processor.image_processor is image_processor
    assert processor.chat_template == "template"
    assert calls == [("model", {"backend": "torchvision"})]


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("raw_config", [{}, {"tie_word_embeddings": None}])
def test_model_loader_preserves_nested_embedding_tying(
    monkeypatch: pytest.MonkeyPatch, tied: bool, raw_config: dict[str, Any]
) -> None:
    """两个方向的绑定配置都由文本模型决定，避免缺失 lm_head 或意外合并独立权重。"""
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    from mineru_vl_utils.transformers_loading import load_transformers_model

    config = transformers.Qwen2VLConfig(text_config={"tie_word_embeddings": tied})
    config.tie_word_embeddings = not tied
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda path: config)
    monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict", lambda path: (raw_config, {}))
    seen = {}

    def load(path: str, **kwargs: Any) -> Any:
        """记录传入真实模型加载器的配置及设备，不创建大模型。"""
        seen.update(kwargs)
        return config

    monkeypatch.setattr(transformers.Qwen2VLForConditionalGeneration, "from_pretrained", load)
    assert load_transformers_model("model", device_map={"": "cpu"}) is config
    assert seen["config"].tie_word_embeddings is tied
    assert seen["dtype"] == "auto"
    assert seen["device_map"] == {"": "cpu"}


def test_model_loader_preserves_explicit_untied_root_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式配置的独立 lm_head 优先于文本子配置，不能静默覆盖调用方权重语义。"""
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    from mineru_vl_utils.transformers_loading import load_transformers_model

    config = transformers.Qwen2VLConfig(text_config={"tie_word_embeddings": True}, tie_word_embeddings=False)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda path: config)
    monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict", lambda path: ({"tie_word_embeddings": False}, {}))
    monkeypatch.setattr(transformers.Qwen2VLForConditionalGeneration, "from_pretrained", lambda path, **kwargs: config)
    assert load_transformers_model("model").tie_word_embeddings is False


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("two_step", [False, True])
def test_lmdeploy_extraction_progress_ownership(
    pipeline_type: type,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    use_async: bool,
    two_step: bool,
) -> None:
    """运行真实高层布局/内容抽取，确认标准同步有进度、异步与两阶段不嵌套。"""
    from unittest.mock import MagicMock
    from PIL import Image
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.structs import ContentBlock
    from mineru_vl_utils.vlm_client import utils

    pipeline = pipeline_type()
    client = MinerUClient(backend="lmdeploy-engine", lmdeploy_engine=pipeline, use_tqdm=enabled)
    original_infer = pipeline.infer
    bars = []

    def infer(prompts: list[Any], **kwargs: Any) -> list[Any]:
        """生成合法布局和内容，仍通过已有替身记录参数与线程生命周期。"""
        responses = original_infer(prompts, **kwargs)
        for response in responses:
            response.text = (
                "<|box_start|>0 0 1000 1000<|box_end|><|ref_start|>text<|ref_end|>"
                if response.text == client.prompts["[layout]"]
                else "recognized"
            )
        return responses

    def progress(**kwargs: Any) -> MagicMock:
        """记录聚合进度，后端进度通过 infer 的参数独立验证。"""
        bar = MagicMock()
        bar.__enter__.return_value = bar
        bars.append((kwargs, bar))
        return bar

    monkeypatch.setattr(pipeline, "infer", infer)
    monkeypatch.setattr(utils, "tqdm", progress)
    monkeypatch.setattr("mineru_vl_utils.vlm_client.lmdeploy_engine_client.tqdm", progress)
    images = [Image.new("RGB", (32, 32)) for _ in range(2)]
    try:
        if two_step:
            results = (
                asyncio.run(client.aio_batch_two_step_extract(images)) if use_async else client.batch_two_step_extract(images)
            )
        else:
            blocks = [[ContentBlock("text", [0, 0, 1, 1])] for _ in images]
            results = (
                asyncio.run(client.aio_batch_extract_with_layout(images, blocks))
                if use_async
                else client.batch_extract_with_layout(images, blocks)
            )
        assert [page[0].content for page in results] == ["recognized", "recognized"]
        assert pipeline.calls
        assert all(call.get("use_tqdm", False) is False for call in pipeline.calls)
        assert any("stream_response" in call for call in pipeline.calls) is (enabled and not use_async and not two_step)
        visible = [(options, bar) for options, bar in bars if not options.get("disable", False)]
        if enabled:
            assert len(visible) == 1
            options, bar = visible[0]
            assert options["total"] == 2
            assert options["desc"] == ("Two Step Extraction" if two_step else "VLM Predict")
            assert sum(call.args[0] for call in bar.update.call_args_list) == 2
        else:
            assert not visible
    finally:
        for image in images:
            image.close()


def test_lmdeploy_empty_batch_does_not_call_pipeline(pipeline_type: type) -> None:
    """空请求不进入 Pipeline，自然不会创建后端进度条。"""
    pipeline = pipeline_type()
    client = LmdeployEngineVlmClient(pipeline)
    assert client.batch_predict([]) == []
    assert asyncio.run(client.aio_batch_predict([], use_tqdm=True)) == []
    assert not pipeline.calls


def test_lmdeploy_stream_progress_updates_before_batch_finishes(pipeline_type: type, monkeypatch: pytest.MonkeyPatch) -> None:
    """慢首请求之前先完成尾请求，进度必须立即更新且最终输出仍按原序排列。"""
    from unittest.mock import MagicMock

    pipeline = pipeline_type()
    bar = MagicMock()
    bar.__enter__.return_value = bar
    factory = MagicMock(return_value=bar)
    drained = []

    def stream(prompts: list[Any], *, gen_config: list[Any], stream_response: bool, **kwargs: Any) -> Iterator[Any]:
        """逐条推进生成器，在下一条响应交付前断言前一条已计入进度。"""
        assert prompts == ["a", "b", "c"]
        assert stream_response is False
        assert kwargs == {"priority": 7}
        try:
            yield SimpleNamespace(index=2, text="c", finish_reason="stop")
            bar.update.assert_called_once_with(1)
            yield SimpleNamespace(index=0, text="a", finish_reason="stop")
            yield SimpleNamespace(index=1, text="b", finish_reason="length")
        finally:
            drained.append(True)

    monkeypatch.setattr(pipeline, "stream_infer", stream)
    monkeypatch.setattr("mineru_vl_utils.vlm_client.lmdeploy_engine_client.tqdm", factory)
    client = LmdeployEngineVlmClient(pipeline)
    assert client.batch_predict([None] * 3, ["a", "b", "c"], priority=7) == ["a", "b", "c"]
    assert drained == [True]
    assert not pipeline.calls
    factory.assert_called_once_with(total=3, desc="VLM Predict")
    assert bar.update.call_count == 3
    bar.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize(
    ("bad_response", "message"),
    [
        (SimpleNamespace(index=1, text="", finish_reason="error"), "inference failed"),
        (SimpleNamespace(index=2, text="late", finish_reason="stop"), "data after request completion"),
        (SimpleNamespace(index=-1, text="x", finish_reason="stop"), "invalid response index"),
        (SimpleNamespace(index=3, text="x", finish_reason="stop"), "invalid response index"),
        (SimpleNamespace(index="1", text="x", finish_reason="stop"), "invalid response index"),
        (SimpleNamespace(index=True, text="x", finish_reason="stop"), "invalid response index"),
        (SimpleNamespace(index=1, text=None, finish_reason="stop"), "incomplete response"),
        (None, "unexpected number of responses"),
    ],
)
def test_lmdeploy_stream_errors_drain_batch_before_raising(
    pipeline_type: type,
    monkeypatch: pytest.MonkeyPatch,
    bad_response: Any,
    message: str,
) -> None:
    """响应错误不能中断本批消费；缺失和终止后重复响应也不得虚增完成计数。"""
    from unittest.mock import MagicMock

    pipeline = pipeline_type()
    bar = MagicMock()
    bar.__enter__.return_value = bar
    drained = []

    def stream(*args: Any, **kwargs: Any) -> Iterator[Any]:
        """错误响应之后仍有合法在途结果，必须消费到末尾再退出。"""
        yield SimpleNamespace(index=2, text="c", finish_reason="stop")
        if bad_response is not None:
            yield bad_response
        yield SimpleNamespace(index=0, text="a", finish_reason="stop")
        drained.append(True)

    monkeypatch.setattr(pipeline, "stream_infer", stream)
    monkeypatch.setattr("mineru_vl_utils.vlm_client.lmdeploy_engine_client.tqdm", MagicMock(return_value=bar))
    with pytest.raises(ServerError, match=message):
        LmdeployEngineVlmClient(pipeline).batch_predict([None] * 3, ["a", "b", "c"])
    assert drained == [True]
    assert bar.update.call_count == 2
    assert bar.__exit__.call_args.args[0] is ServerError


def test_lmdeploy_stream_exception_closes_progress(pipeline_type: type, monkeypatch: pytest.MonkeyPatch) -> None:
    """迭代器自身抛出异常时保留异常并关闭进度上下文。"""
    from unittest.mock import MagicMock

    pipeline = pipeline_type()
    bar = MagicMock()
    bar.__enter__.return_value = bar
    cleaned = []

    def stream(*args: Any, **kwargs: Any) -> Iterator[Any]:
        """模拟后端迭代异常及其内部清理。"""
        try:
            yield SimpleNamespace(index=0, text="a", finish_reason="stop")
            raise RuntimeError("stream failed")
        finally:
            cleaned.append(True)

    monkeypatch.setattr(pipeline, "stream_infer", stream)
    monkeypatch.setattr("mineru_vl_utils.vlm_client.lmdeploy_engine_client.tqdm", MagicMock(return_value=bar))
    with pytest.raises(RuntimeError, match="stream failed"):
        LmdeployEngineVlmClient(pipeline).batch_predict([None, None])
    assert cleaned == [True]
    bar.update.assert_called_once_with(1)
    assert bar.__exit__.call_args.args[0] is RuntimeError


class _ScriptedResponse:
    """镜像 LMDeploy 0.17 Response 的多帧合并语义，供多帧流测试使用。"""

    def __init__(self, text: str, finish_reason: str | None, index: int = 0) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.index = index

    def extend(self, other: "_ScriptedResponse") -> "_ScriptedResponse":
        """按官方语义合并：文本拼接，终止原因与索引以后帧为准。"""
        self.text += other.text
        self.finish_reason = other.finish_reason
        self.index = other.index
        return self


def test_lmdeploy_stream_aggregates_interleaved_multi_frame_responses(
    pipeline_type: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中间帧 finish_reason=None 是 LMDeploy 0.17 正常状态，必须聚合而非判为异常。"""
    from unittest.mock import MagicMock
    from mineru_vl_utils.vlm_client import lmdeploy_engine_client

    pipeline = pipeline_type()
    client = LmdeployEngineVlmClient(pipeline)
    bar = MagicMock()
    bar.__enter__.return_value = bar
    factory = MagicMock(return_value=bar)

    def stream(prompts: list[Any], *, gen_config: list[Any], stream_response: bool, **kwargs: Any) -> Iterator[Any]:
        """多请求交错的增量帧与终止帧，覆盖流式与单帧两种交付形态。"""
        yield _ScriptedResponse("a1", None, index=0)
        yield _ScriptedResponse("b1", None, index=1)
        yield _ScriptedResponse("a2", None, index=0)
        yield _ScriptedResponse("c1", None, index=2)
        yield _ScriptedResponse("A!", "stop", index=0)
        yield _ScriptedResponse("b2", None, index=1)
        yield _ScriptedResponse("B!", "stop", index=1)
        yield _ScriptedResponse("C!", "length", index=2)

    monkeypatch.setattr(pipeline, "stream_infer", stream)
    monkeypatch.setattr(lmdeploy_engine_client, "tqdm", factory)
    assert client.batch_predict([None] * 3, ["p0", "p1", "p2"]) == ["a1a2A!", "b1b2B!", "c1C!"]
    factory.assert_called_once_with(total=3, desc="VLM Predict")
    assert [call.args for call in bar.update.call_args_list] == [(1,), (1,), (1,)]


def test_lmdeploy_stream_delta_without_terminal_reports_missing_response(
    pipeline_type: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """流结束仍未收到终止帧的请求按数量不足报错，而不是误报中间帧异常。"""
    pipeline = pipeline_type()

    def stream(prompts: list[Any], *, gen_config: list[Any], stream_response: bool, **kwargs: Any) -> Iterator[Any]:
        """只交付增量帧，模拟终止帧丢失。"""
        yield _ScriptedResponse("only-delta", None, index=0)

    monkeypatch.setattr(pipeline, "stream_infer", stream)
    with pytest.raises(ServerError, match="unexpected number of responses"):
        LmdeployEngineVlmClient(pipeline).batch_predict([None], ["a"])
