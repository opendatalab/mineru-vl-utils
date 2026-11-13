"""验证批次内部取消不会丢失同步线程排空时才到达的致命异常。"""

import asyncio
import builtins
import sys
import threading
from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any

import pytest

from mineru_vl_utils.vlm_client.base_client import EngineUnavailableError, ServerError
from mineru_vl_utils.vlm_client.lmdeploy_engine_client import LmdeployEngineVlmClient
from mineru_vl_utils.vlm_client.utils import gather_tasks, run_in_thread_until_complete


class ObservedTask(asyncio.Task):
    """在 Python 3.10 也能观测取消请求，无需读取任务私有状态。"""

    def __init__(self, coroutine: Coroutine, *, loop: asyncio.AbstractEventLoop, **kwargs: Any) -> None:
        """保留原生任务调度，仅增加测试所需的取消标志。"""
        super().__init__(coroutine, loop=loop, **kwargs)
        self.cancel_requested = False

    def cancel(self, msg: object = None) -> bool:
        """记录公开取消方法的调用，取消消息仍原样传递。"""
        result = super().cancel(msg)
        self.cancel_requested = self.cancel_requested or result
        return result


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kind", ["fatal", "cause", "group", "ordinary"])
def test_thread_failure_survives_batch_cleanup(kind: str, nested: bool) -> None:
    """内部取消跨越嵌套批次和线程边界后，仍保留原异常、异常链与异常组。"""
    if kind == "group" and sys.version_info < (3, 11):
        pytest.skip("Exception groups require Python 3.11")
    fatal = EngineUnavailableError("worker died")
    worker_error = fatal
    if kind == "cause":
        worker_error = ValueError("wrapped death")
        worker_error.__cause__ = fatal
    elif kind == "group":
        worker_error = builtins.ExceptionGroup("mixed", [ValueError("crop"), fatal])
    elif kind == "ordinary":
        worker_error = ServerError("late request error")
    primary = ValueError("first request error")

    async def run() -> None:
        """用事件控制线程完成时点，确保错误发生在同批取消已经到达之后。"""
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))

        def worker() -> None:
            """保留真实后台工作，取消协程不能提前释放它。"""
            entered.set()
            try:
                assert release.wait(5)
                raise worker_error
            finally:
                finished.set()

        work = ObservedTask(run_in_thread_until_complete(worker), loop=asyncio.get_running_loop())

        async def sibling() -> None:
            """显式保留任务引用，以观测取消已到达线程桥接层。"""
            await work

        async def ordinary() -> None:
            """等待线程开始后报告另一项普通请求错误。"""
            assert await asyncio.to_thread(entered.wait, 2)
            raise primary

        call = gather_tasks([sibling()]) if nested else sibling()
        batch = asyncio.create_task(gather_tasks([ordinary(), call]))
        try:
            async def wait_for_cancel() -> None:
                """等到内部取消真正发出，避免测试依赖线程执行快慢。"""
                while not work.cancel_requested:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_cancel(), 2)
            assert not batch.done()
            assert not finished.is_set()
            release.set()
            expected = primary if kind == "ordinary" else worker_error
            with pytest.raises(type(expected)) as raised:
                await batch
            assert raised.value is expected
            assert finished.is_set()
            await asyncio.sleep(0)
            assert not unhandled
        finally:
            release.set()
            await asyncio.gather(batch, work, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("batch", [False, True])
def test_caller_cancellation_keeps_intent_after_fatal_thread_failure(batch: bool) -> None:
    """调用方主动取消仍返回取消，重复取消也必须等待线程排空且不遗留异常日志。"""
    async def run() -> None:
        """分别覆盖直接调用线程桥和取消整批请求。"""
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))

        def worker() -> None:
            """取消之后才报告明确死亡。"""
            entered.set()
            try:
                assert release.wait(5)
                raise EngineUnavailableError("death after caller cancellation")
            finally:
                finished.set()

        call = run_in_thread_until_complete(worker)
        task = asyncio.create_task(gather_tasks([call]) if batch else call)
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel("caller canceled")
            await asyncio.sleep(0)
            task.cancel("caller canceled again")
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
            await asyncio.sleep(0)
            assert not unhandled
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("isolate", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_lmdeploy_batch_preserves_late_fatal_response(isolate: bool, reverse: bool) -> None:
    """真实 LMDeploy 批量入口、响应分类与线程桥必须共同把后到的死亡报告给宿主。"""
    async def run() -> None:
        """只替换模型推理，保留适配器的全部调度和异常传播路径。"""
        def task_factory(loop: asyncio.AbstractEventLoop, coroutine: Coroutine, **kwargs: Any) -> ObservedTask:
            """为实际批量入口创建可观测的原生任务子类。"""
            return ObservedTask(coroutine, loop=loop, **kwargs)

        asyncio.get_running_loop().set_task_factory(task_factory)
        entered, release, ordinary_finished = threading.Event(), threading.Event(), threading.Event()
        tasks = {}

        def infer(prompts: list[str], **kwargs: Any) -> list[SimpleNamespace]:
            """模拟 Pipeline.infer 返回结构化请求错误或引擎死亡响应。"""
            if prompts[0] == "fatal":
                entered.set()
                assert release.wait(5)
                code = "engine_unavailable"
            else:
                assert entered.wait(2)
                code = "internal_error"
                ordinary_finished.set()
            return [SimpleNamespace(text="", finish_reason="error", error_code=code, error_message=code)]

        client = object.__new__(LmdeployEngineVlmClient)
        client.lmdeploy_engine = SimpleNamespace(infer=infer)
        client.batch_size = 0
        client.max_concurrency = 2
        client.isolate_block_errors = isolate
        client.build_lmdeploy_generation_config = lambda params: None
        predict = client.aio_predict

        async def tracked_predict(**kwargs: Any) -> str:
            """记录当前任务以等待取消，推理仍调用真实适配器。"""
            tasks[kwargs["prompt"]] = asyncio.current_task()
            return await predict(**kwargs)

        client.aio_predict = tracked_predict
        prompts = ["fatal", "ordinary"] if reverse else ["ordinary", "fatal"]
        batch = asyncio.create_task(client.aio_batch_predict([None, None], prompts))
        try:
            assert await asyncio.to_thread(ordinary_finished.wait, 2)
            if not isolate:
                async def wait_for_cancel() -> None:
                    """确保普通错误先触发批次清理，再允许致命响应到达。"""
                    while not tasks["fatal"].cancel_requested:
                        await asyncio.sleep(0)

                await asyncio.wait_for(wait_for_cancel(), 2)
            assert not batch.done()
            release.set()
            with pytest.raises(EngineUnavailableError) as raised:
                await batch
            assert raised.value.error_code == "engine_unavailable"
            assert raised.value.error_message == "engine_unavailable"
        finally:
            release.set()
            await asyncio.gather(batch, return_exceptions=True)

    asyncio.run(run())
