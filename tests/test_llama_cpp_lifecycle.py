"""使用底层真实异常和异步桥接验证 llama.cpp 生命周期适配。"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mineru_vl_utils.vlm_client.base_client import EngineUnavailableError, RequestError, ServerError
from mineru_vl_utils.vlm_client import llama_cpp_engine_client as module


native = pytest.importorskip("mineru_llama_cpp")


@pytest.fixture
def engine():
    """绕过模型加载，保留真实 Engine.agenerate 的线程排空逻辑。"""
    instance = object.__new__(native.Engine)
    instance._closed = False
    instance._close_lock = threading.Lock()
    instance._core = SimpleNamespace(check_health=lambda: None)
    return instance


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("error_type", ["EngineClosedError", "EngineUnavailableError", "EngineError", "InvalidRequestError"])
def test_error_mapping_preserves_native_diagnostics(engine, asynchronous, error_type):
    """严格按公开异常类型转换，结构化详情和原始异常链均可追溯。"""
    original = getattr(native, error_type)("native detail", code="original_code", details={"native_type": "test"})
    engine.generate = MagicMock(side_effect=original)
    client = module.LlamaCppEngineVlmClient(engine, use_tqdm=False)
    expected = (
        EngineUnavailableError
        if isinstance(original, native.EngineUnavailableError)
        else RequestError
        if isinstance(original, native.InvalidRequestError)
        else ServerError
    )
    with pytest.raises(expected) as raised:
        if asynchronous:
            asyncio.run(client.aio_predict(None))
        else:
            client.predict(None)
    assert raised.value.__cause__ is original
    assert raised.value.__cause__.details == {"native_type": "test"}
    if expected is ServerError:
        assert not isinstance(raised.value, EngineUnavailableError)
    if expected is not RequestError:
        assert raised.value.error_code == "original_code"


def test_health_reads_native_state_and_never_closes_engine(engine):
    """健康检测复用底层状态；关闭或死亡均以明确异常上报，不接管引擎。"""
    engine.check_health = MagicMock()
    engine.close = MagicMock()
    client = module.LlamaCppEngineVlmClient(engine)
    assert client.health_probe_mode == "sync"
    assert client.probe_health().state == "healthy"
    original = native.EngineClosedError("closed", code="engine_closed")
    engine.check_health.side_effect = original
    with pytest.raises(EngineUnavailableError) as raised:
        client.probe_health()
    assert raised.value.__cause__ is original
    engine.close.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("fatal_first", [False, True])
def test_mixed_batch_fatal_wins_and_drains(engine, monkeypatch, asynchronous, isolated, fatal_first):
    """已有成功、两种错误顺序和容错开关均不能吞掉晚到致命错误。"""
    barrier = threading.Barrier(3)
    later = threading.Event()
    finished = []
    original = native.EngineUnavailableError("loop exited", code="engine_dead")
    ordinary = native.InvalidRequestError("bad crop")
    capture = module.capture_block_error

    def capture_and_release(error, **kwargs):
        """让后一错误严格等待先前错误到达批次边界，避免用 sleep 猜顺序。"""
        if isinstance(error, EngineUnavailableError) == fatal_first:
            later.set()
        return capture(error, **kwargs)

    def generate(messages, params):
        """三个请求先同时进入推理，再按明确顺序返回成功或失败。"""
        prompt = messages[-1]["content"][-1]["text"]
        barrier.wait(5)
        if prompt == "ok":
            finished.append(prompt)
            return native.GenerateResult("ok", "stop", 1, 1, None)
        if (prompt == "fatal") != fatal_first:
            assert later.wait(5)
        finished.append(prompt)
        raise original if prompt == "fatal" else ordinary

    monkeypatch.setattr(module, "capture_block_error", capture_and_release)
    engine.generate = generate
    client = module.LlamaCppEngineVlmClient(engine, max_concurrency=3, use_tqdm=False, isolate_block_errors=isolated)
    with pytest.raises(EngineUnavailableError) as raised:
        if asynchronous:
            asyncio.run(client.aio_batch_predict([None] * 3, ["ok", "ordinary", "fatal"]))
        else:
            client.batch_predict([None] * 3, ["ok", "ordinary", "fatal"])
    assert raised.value.__cause__ is original
    assert sorted(finished) == ["fatal", "ok", "ordinary"]


def test_ordinary_error_retains_batch_fallback(engine):
    """普通服务端错误仍可按原有策略隔离，后续同实例推理可继续。"""
    engine.generate = MagicMock(
        side_effect=[native.EngineError("crop failure"), native.GenerateResult("ok", "stop", 1, 1, None)]
    )
    client = module.LlamaCppEngineVlmClient(engine, max_concurrency=1, use_tqdm=False, isolate_block_errors=True)
    assert client.batch_predict([None, None]) == ["", "ok"]


@pytest.mark.parametrize("wrapped", ["cause", "group"])
def test_wrapped_fatal_is_never_isolated(engine, wrapped):
    """原生致命异常被上层包装后仍可识别，异常链保留全部上下文。"""
    import builtins

    fatal = native.EngineUnavailableError("stopped", code="engine_dead")
    if wrapped == "cause":
        error = ValueError("wrapper")
        error.__cause__ = fatal
    else:
        if not hasattr(builtins, "ExceptionGroup"):
            pytest.skip("ExceptionGroup requires Python 3.11")
        error = builtins.ExceptionGroup("batch", [ValueError("ordinary"), fatal])
    engine.generate = MagicMock(side_effect=error)
    client = module.LlamaCppEngineVlmClient(engine, use_tqdm=False, isolate_block_errors=True)
    with pytest.raises(EngineUnavailableError) as raised:
        client.batch_predict([None])
    assert raised.value.__cause__ is error
