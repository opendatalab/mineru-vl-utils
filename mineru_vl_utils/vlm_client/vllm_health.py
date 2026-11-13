"""在调用方提供的引擎循环内维护唯一的 vLLM 健康探测。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .base_client import EngineHealth, EngineUnavailableError
from .vllm_errors import vllm_error_scope


class VllmHealthProbe:
    """只持有公开 check_health 回调，不拥有引擎及其事件循环。"""

    def __init__(self, check_health: Callable[[], Awaitable[None]]) -> None:
        """延迟到首次调用绑定循环，不创建后台线程或加载模型。"""
        self._check_health = check_health
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def _require_loop(self) -> None:
        """跨循环使用属于调用契约错误，不能误报为引擎死亡。"""
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("vLLM health probe must run on its owning event loop")

    async def _check(self) -> None:
        """保留公开健康接口抛出的真实致命异常与原始 traceback。"""
        with vllm_error_scope():
            await self._check_health()

    @staticmethod
    def _observe(task: asyncio.Task[None]) -> None:
        """超时后的探测异常也及时取走，结果仍保留供下次探测传播。"""
        if not task.cancelled():
            task.exception()

    async def check(self, timeout: float = 5.0) -> EngineHealth:
        """超时不取消原任务；明确死亡抛公共异常，未知探测错误暂时不可用。"""
        if timeout <= 0:
            raise ValueError("Health probe timeout must be positive")
        self._require_loop()
        if self._closed:
            return EngineHealth("dead", "vLLM health probe is closed")
        if self._task is None:
            self._task = asyncio.create_task(self._check())
            self._task.add_done_callback(self._observe)
        task = self._task
        try:
            done, _ = await asyncio.wait([task], timeout=timeout)
            if not done:
                return EngineHealth("unavailable", f"vLLM health probe timed out after {timeout:g}s")
            if task.cancelled():
                return EngineHealth("unavailable", "vLLM health probe canceled during shutdown")
            task.result()
            return EngineHealth("healthy")
        except Exception as error:
            if EngineUnavailableError.find_in(error) is not None:
                raise
            return EngineHealth("unavailable", f"vLLM health probe failed: {type(error).__name__}: {error}")
        finally:
            if self._task is task and task.done():
                self._task = None

    async def aclose(self) -> None:
        """取消并排空客户端自有任务，重复取消不提前释放引擎租约。"""
        self._require_loop()
        already_closed = self._closed
        self._closed = True
        task = self._task
        if task is None:
            return
        if not task.done() and not already_closed:
            task.cancel()
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.wait([task])
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        self._observe(task)
        self._task = None
        if cancellation is not None:
            raise cancellation


__all__ = ["VllmHealthProbe"]
