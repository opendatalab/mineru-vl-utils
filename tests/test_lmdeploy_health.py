"""以真实循环线程验证 LMDeploy 适配器的故障优先级、探测租约和关闭。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest

from mineru_vl_utils.vlm_client.base_client import EngineUnavailableError, ServerError
from mineru_vl_utils.vlm_client.lmdeploy_engine_client import LmdeployEngineVlmClient, _inference_error
from mineru_vl_utils.vlm_client.lmdeploy_health import LmdeployHealthProbe


class Backend:
    """实现两版 PyTorch/TurboMind 共用的异步健康协议。"""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """使用可控状态模拟健康、停机与延迟探测。"""
        self.loop = loop
        self.alive = True
        self.block = False
        self.calls = 0
        self.entered = threading.Event()
        self.finished = threading.Event()
        self.gate: asyncio.Event | None = None

    async def get_health_status(self) -> dict[str, Any]:
        """确认探测运行在 Pipeline 循环，取消后仍完成清理。"""
        assert asyncio.get_running_loop() is self.loop
        self.calls += 1
        self.gate = asyncio.Event()
        self.entered.set()
        try:
            if self.block:
                await self.gate.wait()
            return {"alive": self.alive, "message": "alive" if self.alive else "stopped", "schedule_metrics": None}
        finally:
            await asyncio.sleep(0.01)
            self.finished.set()


@pytest.fixture
def pipeline() -> Iterator[SimpleNamespace]:
    """创建真实 Pipeline 形态的循环线程，结束时检查所有探测均已排空。"""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run() -> None:
        """执行后台事件循环并清理测试创建的资源。"""
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    thread = threading.Thread(target=run)
    thread.start()
    assert ready.wait(2)
    pipe = SimpleNamespace(
        internal_thread=SimpleNamespace(loop=loop, thread=thread, closed=False),
        async_engine=SimpleNamespace(is_sleeping=False, engine=Backend(loop)),
    )
    try:
        yield pipe
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)
        assert not thread.is_alive()


def test_health_distinguishes_sleep_and_engine_death(pipeline: SimpleNamespace) -> None:
    """健康可用、睡眠暂不可用、明确停止死亡三种状态不可混淆。"""
    probe = LmdeployHealthProbe(pipeline)
    assert probe.check().state == "healthy"
    pipeline.async_engine.is_sleeping = True
    assert probe.check().state == "unavailable"
    pipeline.async_engine.is_sleeping = False
    pipeline.async_engine.engine.alive = False
    assert probe.check().state == "dead"
    probe.close()


def test_timeout_reuses_one_probe_and_later_recovers(pipeline: SimpleNamespace) -> None:
    """多次超时只保留一个在途探测，随后成功结果可以恢复可用状态。"""
    backend = pipeline.async_engine.engine
    backend.block = True
    probe = LmdeployHealthProbe(pipeline)
    for _ in range(3):
        assert probe.check(0.02).state == "unavailable"
    assert backend.calls == 1
    pipeline.internal_thread.loop.call_soon_threadsafe(backend.gate.set)
    assert backend.finished.wait(2)
    assert probe.check().state == "healthy"
    probe.close()


def test_close_cancels_probe_and_waits_for_actual_cleanup(pipeline: SimpleNamespace) -> None:
    """关闭等待探测的 finally，而不是仅看到并发 Future 已标记取消。"""
    backend = pipeline.async_engine.engine
    backend.block = True
    probe = LmdeployHealthProbe(pipeline)
    assert probe.check(0.02).state == "unavailable"
    probe.close()
    assert backend.finished.is_set()
    assert probe.check().state == "dead"


def test_close_before_probe_starts_does_not_leak_or_wait_forever(pipeline: SimpleNamespace) -> None:
    """探测尚未进入循环时关闭，也必须完成等待通知。"""
    block = threading.Event()
    entered = threading.Event()

    def hold_loop() -> None:
        """暂缓所属循环，制造提交与首次执行之间的关闭竞态。"""
        entered.set()
        block.wait(2)

    pipeline.internal_thread.loop.call_soon_threadsafe(hold_loop)
    assert entered.wait(2)
    probe = LmdeployHealthProbe(pipeline)
    assert probe.check(0.01).state == "unavailable"
    with ThreadPoolExecutor(1) as executor:
        closing = executor.submit(probe.close)
        block.set()
        closing.result(timeout=2)
    assert probe.check().state == "dead"


@pytest.mark.parametrize("codes", [("internal_error", "engine_unavailable"), ("engine_unavailable", "internal_error")])
@pytest.mark.parametrize("progress", [False, True])
def test_fatal_response_wins_in_both_orders_and_drains_stream(codes: tuple[str, str], progress: bool) -> None:
    """混合错误不依赖响应顺序，错误后的响应也必须被消费完成。"""
    responses = [
        SimpleNamespace(index=i, text="", finish_reason="error", error_code=code, error_message=f"detail-{code}")
        for i, code in enumerate(codes)
    ]
    consumed: list[int] = []

    def stream(*args: Any, **kwargs: Any) -> Iterator[SimpleNamespace]:
        """记录原始进度迭代器是否被完整消费。"""
        for response in responses:
            consumed.append(response.index)
            yield response

    client = object.__new__(LmdeployEngineVlmClient)
    client.lmdeploy_engine = SimpleNamespace(infer=MagicMock(return_value=responses), stream_infer=stream)
    with pytest.raises(EngineUnavailableError) as raised:
        client._predict_one_batch([None, None], ["a", "b"], [], use_tqdm=progress)
    assert raised.value.error_code == "engine_unavailable"
    assert raised.value.error_message == "detail-engine_unavailable"
    if progress:
        assert consumed == [0, 1]


def test_ordinary_backend_error_keeps_details_and_remains_recoverable() -> None:
    """普通请求错误保留诊断信息，不升级为整体引擎死亡。"""
    error = _inference_error(
        [SimpleNamespace(finish_reason="error", error_code="internal_error", error_message="original details")], "failed"
    )
    assert type(error) is ServerError
    assert "internal_error" in str(error) and "original details" in str(error)
    assert error.error_code == "internal_error"
    assert error.error_message == "original details"
