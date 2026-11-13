"""以官方异常源码和真实适配入口验证生命周期，不加载 GPU 模型。"""

from __future__ import annotations

import asyncio
import builtins
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mineru_vl_utils import MinerUClient
from mineru_vl_utils.vlm_client.base_client import EngineUnavailableError, UnsupportedError, VlmClient
from mineru_vl_utils.vlm_client.batch_errors import capture_block_error
from mineru_vl_utils.vlm_client.lmdeploy_engine_client import LmdeployEngineVlmClient
from mineru_vl_utils.vlm_client.mlx_client import MlxVlmClient
from mineru_vl_utils.vlm_client.vllm_async_engine_client import VllmAsyncEngineVlmClient
from mineru_vl_utils.vlm_client.vllm_engine_client import VllmEngineVlmClient
from mineru_vl_utils.vlm_client.vllm_health import VllmHealthProbe


def make_client(native_errors: ModuleType, *, asynchronous: bool, error: Exception | None = None) -> Any:
    """仅替换模型计算，公开预测、评分、批调度和错误处理均运行真实实现。"""
    cls = VllmAsyncEngineVlmClient if asynchronous else VllmEngineVlmClient
    client = object.__new__(cls)
    VlmClient.__init__(client, isolate_block_errors=asynchronous)
    client.use_tqdm = False
    client.batch_size = 2
    client.max_concurrency = 1
    client.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt")
    client.build_vllm_sampling_params = lambda _: SimpleNamespace()
    client._build_score_prompt_pair = lambda *args: ("prompt", 1)
    client.get_output_content = lambda output: output.text
    client._get_output_scored = client.get_output_scored = lambda output: output
    client._extract_prompt_logprobs = lambda output, count: output
    client.VllmRequestOutputKind = SimpleNamespace(FINAL_ONLY="final")
    client.model_max_length = 100
    state = SimpleNamespace(error=error, calls=0, cleaned=0, health_calls=0, gate=None)

    def generate(**kwargs: Any) -> list[SimpleNamespace]:
        """同步生成直接抛出真实上游错误。"""
        state.calls += 1
        if state.error is not None:
            raise state.error
        return [SimpleNamespace(text="success") for _ in kwargs["prompts"]]

    async def async_generate(**kwargs: Any):
        """先交出中间响应，再抛出故障，确保错误来自真实生成器迭代边界。"""
        state.calls += 1
        try:
            yield SimpleNamespace(text="partial")
            if state.error is not None:
                raise state.error
            yield SimpleNamespace(text="success")
        finally:
            state.cleaned += 1

    async def render(prompt: Any) -> Any:
        """绕过视觉编码但保留异步预处理边界。"""
        return prompt

    async def check_health() -> None:
        """可暂停的健康接口用于验证超时复用及关闭清理。"""
        state.health_calls += 1
        if state.gate is not None:
            await state.gate.wait()
        if state.error is not None:
            raise state.error

    client._render_vllm_cmpl_input = render
    client.vllm_llm = SimpleNamespace(generate=generate)
    client.vllm_async_llm = SimpleNamespace(generate=async_generate, check_health=check_health)
    client._health_probe = VllmHealthProbe(check_health)
    client.test_state = state
    return client


@pytest.mark.parametrize(
    "method", ["predict", "batch_predict", "predict_scored", "batch_predict_scored", "score", "batch_score"]
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_all_vllm_inference_entries_normalize_real_death(native_errors: ModuleType, method: str, asynchronous: bool) -> None:
    """同步、异步及评分接口均保留原异常链，不让致命错误穿透成可降级类型。"""
    error = native_errors.EngineDeadError()
    assert not isinstance(error, RuntimeError)
    client = make_client(native_errors, asynchronous=asynchronous, error=error)
    args = [[None, None]] if method.startswith("batch_") else [None]
    if method in {"score", "batch_score"}:
        args.append(["answer", "answer"] if method.startswith("batch_") else "answer")
    with pytest.raises(EngineUnavailableError) as raised:
        result = getattr(client, "aio_" + method if asynchronous else method)(*args)
        if asynchronous:
            asyncio.run(result)
    assert raised.value.__cause__ is error
    assert raised.value.error_code == "engine_unavailable"
    assert str(error) in raised.value.error_message
    if asynchronous:
        assert client.test_state.cleaned == client.test_state.calls


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("order", ["fatal-first", "fatal-last"])
def test_mixed_batch_never_isolates_native_death(native_errors: ModuleType, enabled: bool, order: str) -> None:
    """成功项不能掩盖引擎死亡，开启或关闭容错以及请求顺序均不影响传播。"""
    client = make_client(native_errors, asynchronous=True)
    client.isolate_block_errors = enabled
    native_generate = client.vllm_async_llm.generate
    error = native_errors.EngineDeadError()

    async def generate(**kwargs: Any):
        """让真实适配器消费一项成功和一项致命生成器。"""
        client.test_state.error = error if client.test_state.calls == (0 if order == "fatal-first" else 1) else None
        async for output in native_generate(**kwargs):
            yield output

    client.vllm_async_llm.generate = generate
    with pytest.raises(EngineUnavailableError):
        asyncio.run(client.aio_batch_predict([None, None]))


def test_ordinary_native_failure_keeps_identity(native_errors: ModuleType) -> None:
    """上游明确可恢复的请求错误不能被提升为实例死亡。"""
    error = native_errors.EngineGenerateError()
    client = make_client(native_errors, asynchronous=True, error=error)
    with pytest.raises(type(error)) as raised:
        asyncio.run(client.aio_predict(None))
    assert raised.value is error


@pytest.mark.parametrize("grouped", [False, True])
def test_native_death_inside_wrapper_retains_original_chain(native_errors: ModuleType, grouped: bool) -> None:
    """适配边界遍历真实原生异常的包装与分组，并完整保留外层诊断。"""
    native = native_errors.EngineDeadError()
    if grouped and sys.version_info < (3, 11):
        pytest.skip("Exception groups require Python 3.11")
    wrapper = builtins.ExceptionGroup("mixed", [ValueError("crop"), native]) if grouped else ValueError("outer")
    if not grouped:
        wrapper.__cause__ = native
    client = make_client(native_errors, asynchronous=True, error=wrapper)
    with pytest.raises(EngineUnavailableError) as raised:
        asyncio.run(client.aio_predict(None))
    assert raised.value.__cause__ is wrapper
    assert str(native) in raised.value.error_message


@pytest.mark.parametrize("fatal_first", [False, True])
def test_cleanup_fatal_error_has_priority_over_ordinary_error(fatal_first: bool) -> None:
    """同批普通错误触发取消时，清理才暴露的引擎死亡也必须传到宿主。"""
    from mineru_vl_utils.vlm_client.utils import gather_tasks

    async def run() -> None:
        """精确安排一个普通错误和一个清理阶段的致命错误。"""
        entered = asyncio.Event()
        fatal = EngineUnavailableError("death during cleanup")

        async def ordinary() -> None:
            """等待另一项开始后才报告单次失败。"""
            await entered.wait()
            raise ValueError("bad request")

        async def dying() -> None:
            """模拟收到取消后由引擎清理暴露明确的整体故障。"""
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                raise fatal

        tasks = [dying(), ordinary()] if fatal_first else [ordinary(), dying()]
        with pytest.raises(EngineUnavailableError) as raised:
            await gather_tasks(tasks)
        assert raised.value is fatal

    asyncio.run(run())


def test_health_close_waits_for_cleanup_under_repeated_cancellation() -> None:
    """关闭探测不能只取消 Future；重复取消仍等待底层 finally 结束。"""

    async def run() -> None:
        """由事件明确控制探测清理阶段，避免依赖 GPU 或随机时序。"""
        cleaning, release = asyncio.Event(), asyncio.Event()
        finished = False

        async def check() -> None:
            """取消后先完成必要清理，再结束探测任务。"""
            nonlocal finished
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                finished = True

        probe = VllmHealthProbe(check)
        assert (await probe.check(0.01)).state == "unavailable"
        closing = asyncio.create_task(probe.aclose())
        await cleaning.wait()
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        assert not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert finished
        await probe.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("grouped", [False, True])
def test_wrapped_public_fatal_error_is_not_isolated(grouped: bool) -> None:
    """公共致命异常被包装或放入异常组后仍不能降级为空文本。"""
    fatal = EngineUnavailableError("dead")
    if grouped and sys.version_info < (3, 11):
        pytest.skip("Exception groups require Python 3.11")
    error = builtins.ExceptionGroup("mixed", [ValueError("crop"), fatal]) if grouped else ValueError("wrapped")
    if not grouped:
        error.__cause__ = fatal
    with pytest.raises(type(error)):
        capture_block_error(error, enabled=True, backend="test", index=0)


def test_health_capabilities_are_explicit() -> None:
    """没有健康协议的同步 vLLM 和 MLX 不得伪造健康状态。"""
    for cls, expected in [
        (VllmEngineVlmClient, "none"),
        (MlxVlmClient, "none"),
        (LmdeployEngineVlmClient, "sync"),
        (VllmAsyncEngineVlmClient, "async"),
    ]:
        assert cls.health_probe_mode == expected
    with pytest.raises(UnsupportedError):
        asyncio.run(VlmClient().aio_probe_health())


def test_async_health_timeout_reuses_probe_and_close_leaves_engine_owned(native_errors: ModuleType) -> None:
    """超时后只保留一个探测；恢复后可再探测，关闭只清理客户端任务。"""

    async def run() -> None:
        """在单一所属循环内运行真实客户端的健康接口。"""
        client = make_client(native_errors, asynchronous=True)
        client.test_state.gate = asyncio.Event()
        for _ in range(3):
            assert (await client.aio_probe_health(0.01)).state == "unavailable"
        assert client.test_state.health_calls == 1
        client.test_state.gate.set()
        assert (await client.aio_probe_health()).state == "healthy"
        client.test_state.error = native_errors.EngineDeadError()
        with pytest.raises(EngineUnavailableError):
            await client.aio_probe_health()
        wrapper = object.__new__(MinerUClient)
        wrapper.backend, wrapper.client = "vllm-async-engine", client
        await wrapper.aclose()
        assert (await client.aio_probe_health()).state == "dead"

    asyncio.run(run())
