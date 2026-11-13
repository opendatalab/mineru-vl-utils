"""以真实循环线程验证 LMDeploy 适配器的故障优先级、探测租约和关闭。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
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


def test_close_waits_for_actual_cleanup(pipeline: SimpleNamespace) -> None:
    """关闭不能用取消冒充排空；异步探测也必须实际完成后才能返回。"""
    backend = pipeline.async_engine.engine
    backend.block = True
    probe = LmdeployHealthProbe(pipeline)
    assert probe.check(0.02).state == "unavailable"
    with pytest.raises(TimeoutError):
        probe.close(0.02)
    assert not backend.finished.is_set()
    pipeline.internal_thread.loop.call_soon_threadsafe(backend.gate.set)
    probe.close()
    assert backend.finished.is_set()
    assert probe.check().state == "dead"


@pytest.mark.parametrize("fails", [False, True])
def test_close_drains_executor_backed_health(pipeline: SimpleNamespace, fails: bool) -> None:
    """复现两版 TurboMind 的执行器调用；协程结束不能代替底层线程结束。"""
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def native_health() -> dict[str, bool]:
        """模拟仍在访问模型通信器的工作线程，正常和异常返回都要先排空。"""
        calls.append(1)
        entered.set()
        try:
            assert release.wait(5)
            if fails:
                raise RuntimeError("health worker failed")
            return {"alive": True}
        finally:
            finished.set()

    with ThreadPoolExecutor(1) as health_executor:
        async def get_health_status() -> dict[str, bool]:
            """沿用 LMDeploy 0.17/0.18 TurboMind 的真实公开接口实现方式。"""
            return await asyncio.get_running_loop().run_in_executor(health_executor, native_health)

        pipeline.async_engine.engine.get_health_status = get_health_status
        probe = LmdeployHealthProbe(pipeline)
        try:
            assert probe.check(0.02).state == "unavailable"
            assert entered.wait(2)
            for _ in range(2):
                with pytest.raises(TimeoutError, match="did not finish"):
                    probe.close(0.02)
            assert not finished.is_set()
            assert probe.check().state == "dead"
            assert len(calls) == 1
        finally:
            release.set()
            probe.close(2)
        assert finished.is_set()


def test_close_waits_for_executor_worker_without_deadline(pipeline: SimpleNamespace) -> None:
    """宿主已有进程级关闭期限时，探测租约必须持续到工作线程真实退出。"""
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def native_health() -> dict[str, bool]:
        """保留一个可控的在途健康工作。"""
        entered.set()
        assert release.wait(5)
        finished.set()
        return {"alive": True}

    with ThreadPoolExecutor(1) as health_executor, ThreadPoolExecutor(2) as callers:
        async def get_health_status() -> dict[str, bool]:
            """使关闭协程和底层线程的完成时刻可以明确区分。"""
            return await asyncio.get_running_loop().run_in_executor(health_executor, native_health)

        pipeline.async_engine.engine.get_health_status = get_health_status
        probe = LmdeployHealthProbe(pipeline)
        assert probe.check(0.02).state == "unavailable"
        assert entered.wait(2)
        closing = [callers.submit(probe.close, None) for _ in range(2)]
        try:
            for call in closing:
                with pytest.raises(FutureTimeoutError):
                    call.result(0.03)
            assert not finished.is_set()
        finally:
            release.set()
            for call in closing:
                call.result(2)
        assert finished.is_set()


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


def test_close_after_task_creation_skips_unstarted_backend(pipeline: SimpleNamespace) -> None:
    """任务已创建但尚未首次执行时关闭，不能在随后启动迟到的后端工作。"""
    probe = LmdeployHealthProbe(pipeline)
    start = probe._start

    def start_then_close(future: Any) -> None:
        """在同一个循环回调内完成创建和关闭，固定首次执行之前的竞态。"""
        start(future)
        with pytest.raises(TimeoutError):
            probe.close(0)

    probe._start = start_then_close
    assert probe.check().state == "dead"
    probe.close()
    assert pipeline.async_engine.engine.calls == 0


def test_external_cancellation_is_not_confirmed_drainage(pipeline: SimpleNamespace) -> None:
    """外部取消丢失线程完成信号时，关闭不能虚报成功或再提交一个健康任务。"""
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def native_health() -> dict[str, bool]:
        """模拟被取消协程遗留的真实工作线程。"""
        entered.set()
        assert release.wait(5)
        finished.set()
        return {"alive": True}

    with ThreadPoolExecutor(1) as executor:
        async def get_health_status() -> dict[str, bool]:
            """取消此协程会取消等待 Future，但不会终止已经运行的线程。"""
            return await asyncio.get_running_loop().run_in_executor(executor, native_health)

        pipeline.async_engine.engine.get_health_status = get_health_status
        probe = LmdeployHealthProbe(pipeline)
        try:
            assert probe.check(0.02).state == "unavailable"
            assert entered.wait(2)
            pipeline.internal_thread.loop.call_soon_threadsafe(probe._task.cancel)
            assert probe.check().state == "unavailable"
            with pytest.raises(TimeoutError):
                probe.close(0.02)
            assert not finished.is_set()
        finally:
            release.set()


def test_client_aclose_cancellation_keeps_probe_lease(pipeline: SimpleNamespace) -> None:
    """取消客户端异步关闭时仍要排空正在执行的健康工作。"""
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def native_health() -> dict[str, bool]:
        """用可控工作线程验证取消不能先于资源使用结束返回。"""
        entered.set()
        assert release.wait(5)
        finished.set()
        return {"alive": True}

    with ThreadPoolExecutor(1) as executor:
        async def get_health_status() -> dict[str, bool]:
            """让真实客户端关闭路径观察执行器工作。"""
            return await asyncio.get_running_loop().run_in_executor(executor, native_health)

        pipeline.async_engine.engine.get_health_status = get_health_status
        client = object.__new__(LmdeployEngineVlmClient)
        client._health_probe = LmdeployHealthProbe(pipeline)
        assert client.probe_health(0.02).state == "unavailable"
        assert entered.wait(2)
        async def close_client() -> None:
            """沿用项目既有 asyncio.run 测试方式，不引入额外测试插件依赖。"""
            closing = asyncio.create_task(client.aclose())
            try:
                await asyncio.sleep(0)
                closing.cancel()
                await asyncio.sleep(0.03)
                assert not closing.done() and not finished.is_set()
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await closing

        asyncio.run(close_client())
        assert finished.is_set()
        client.close_health_probe()


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
