"""在 LMDeploy Pipeline 所属循环探测健康，不触发推理或读取其私有任务。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from functools import partial
from typing import Any

from loguru import logger

from .base_client import EngineHealth


class LmdeployHealthProbe:
    """为 LMDeploy 0.17/0.18 的共同后端健康接口维护至多一个在途调用。"""

    def __init__(self, pipeline: Any) -> None:
        """只保留宿主提供的 Pipeline，不创建循环、线程或模型。"""
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self._future: Future[EngineHealth] | None = None
        self._task: asyncio.Task[EngineHealth] | None = None
        self._finished = threading.Event()
        self._finished.set()
        self._closed = False

    async def _probe(self) -> EngineHealth:
        """在引擎循环内读取两版共同接口，睡眠和异常探测不等同于死亡。"""
        engine = self._pipeline.async_engine
        if engine.is_sleeping:
            return EngineHealth("unavailable", "LMDeploy engine is sleeping")
        status = await engine.engine.get_health_status()
        if not isinstance(status, dict) or type(status.get("alive")) is not bool:
            return EngineHealth("unavailable", "LMDeploy returned an invalid health response")
        return EngineHealth("healthy" if status["alive"] else "dead", str(status.get("message") or ""))

    def _finish(self, future: Future[EngineHealth], task: asyncio.Task[EngineHealth]) -> None:
        """任务真正结束后再通知同步等待者，覆盖首次运行前就取消的情况。"""
        with self._lock:
            try:
                future.set_result(task.result())
            except asyncio.CancelledError:
                future.set_result(EngineHealth("unavailable", "LMDeploy health probe canceled during shutdown"))
            except BaseException as exc:
                future.set_exception(exc)
            self._task = None
            self._finished.set()

    def _start(self, future: Future[EngineHealth]) -> None:
        """在所属循环创建唯一任务，关闭已开始时直接结束尚未执行的探测。"""
        with self._lock:
            if self._closed:
                future.set_result(EngineHealth("dead", "LMDeploy health probe is closed"))
                self._finished.set()
                return
            self._task = asyncio.create_task(self._probe())
            self._task.add_done_callback(partial(self._finish, future))

    def _cancel(self) -> None:
        """仅在 Pipeline 循环中请求取消，不把 Future 取消误当作资源清理完成。"""
        with self._lock:
            if self._task is not None:
                self._task.cancel()

    def check(self, timeout: float = 5.0) -> EngineHealth:
        """超时保留同一个 Future 供下次检查，不排队创建更多后台探测。"""
        if timeout <= 0:
            raise ValueError("Health probe timeout must be positive")
        with self._lock:
            thread = self._pipeline.internal_thread
            if self._closed or thread.closed or not thread.thread.is_alive() or not thread.loop.is_running():
                return EngineHealth("dead", "LMDeploy Pipeline event loop is closed")
            if self._future is None:
                self._finished.clear()
                self._future = Future()
                try:
                    thread.loop.call_soon_threadsafe(self._start, self._future)
                except RuntimeError:
                    self._future.set_result(EngineHealth("dead", "LMDeploy Pipeline event loop is closed"))
                    self._finished.set()
            future = self._future
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return EngineHealth("unavailable", f"LMDeploy health probe timed out after {timeout:g}s")
        except Exception as exc:
            logger.exception("LMDeploy health probe failed")
            return EngineHealth("unavailable", f"LMDeploy health probe failed: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                if self._future is future and future.done() and self._finished.is_set():
                    self._future = None

    def close(self, timeout: float = 5.0) -> None:
        """停止接收探测并等待已提交调用退出，不关闭宿主提供的引擎。"""
        with self._lock:
            self._closed = True
            future = self._future
            if future is not None:
                loop = self._pipeline.internal_thread.loop
                if not loop.is_closed():
                    loop.call_soon_threadsafe(self._cancel)
        if not self._finished.wait(timeout):
            raise TimeoutError("LMDeploy health probe did not finish during client shutdown")


__all__ = ["LmdeployHealthProbe"]
