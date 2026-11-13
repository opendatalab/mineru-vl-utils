"""验证新版后端的视觉特征保留和线程取消生命周期。"""

import asyncio
import threading
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from mineru_vl_utils.vlm_client.mlx_client import MlxVlmClient
from mineru_vl_utils.vlm_client.vllm_engine_client import VllmEngineVlmClient


class RenderingEngine:
    """模拟同步引擎内部渲染；重复输入 EngineInput 会丢失视觉内容。"""

    def __init__(self) -> None:
        """记录每次引擎内部处理得到的实际图像特征。"""
        self.renderer = SimpleNamespace(render_cmpl=self.render)
        self.render_count = 0
        self.features = []
        self.progress_options = []

    def render(self, prompts: list[dict]) -> list[dict]:
        """将 raw 图像物化为特征，刻意不把已物化输入当作 raw 图像。"""
        self.render_count += 1
        return [{"type": "multimodal", "mm_kwargs": p.get("multi_modal_data", {}).get("image", [])} for p in prompts]

    def generate(self, prompts: list[dict], **kwargs: object) -> list[object]:
        """同步 generate 自行渲染，输出后续断言要核对的图像像素。"""
        progress = kwargs["use_tqdm"]
        self.progress_options.append(progress)
        if progress:
            with progress(total=len(prompts), desc="Processed prompts", dynamic_ncols=True, postfix="engine speed") as bar:
                bar.update(len(prompts))
        rendered = self.render(prompts)
        self.features.extend([im.getpixel((0, 0)) for p in rendered for im in p["mm_kwargs"]])
        return [SimpleNamespace(text="recognized") for _ in prompts]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mode", ["predict", "predict_scored", "score"])
def test_vllm_sync_keeps_visual_features(mode: str, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """普通预测和两种评分都必须让引擎恰好渲染一次，并保留两张不同图像。"""
    factory = MagicMock()
    monkeypatch.setattr("mineru_vl_utils.vlm_client.utils.tqdm", factory)
    client = object.__new__(VllmEngineVlmClient)
    client.vllm_llm = RenderingEngine()
    client.use_tqdm = enabled
    client.batch_size = 0
    client.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: "prompt")
    client.build_messages = lambda *a: []
    client.build_vllm_sampling_params = lambda *a: SimpleNamespace()
    client.get_output_content = lambda output: output.text
    client.get_output_scored = lambda output: output
    client._build_score_prompt_pair = lambda *a: ("prompt", 1)
    client._extract_prompt_logprobs = lambda output, count: output
    images = [Image.new("RGB", (2, 2), "red"), Image.new("RGB", (2, 2), "blue")]
    if mode == "predict":
        outputs = client.batch_predict(images)
    elif mode == "predict_scored":
        outputs = client.batch_predict_scored(images)
    else:
        outputs = client.batch_score(images, ["a", "b"])
    assert len(outputs) == 2
    if enabled:
        assert callable(client.vllm_llm.progress_options[0])
        factory.assert_called_once_with(total=2, desc="VLM Predict", dynamic_ncols=True, postfix="engine speed")
    else:
        assert client.vllm_llm.progress_options == [False]
        factory.assert_not_called()
    assert client.vllm_llm.render_count == 1
    assert client.vllm_llm.features == [(255, 0, 0), (0, 0, 255)]


@pytest.mark.parametrize("batch", [False, True])
def test_mlx_cancellation_holds_lease_until_worker_exits(batch: bool) -> None:
    """取消单图或批量请求时，不能在底层线程仍使用模型时提前结束协程。"""
    entered, release = threading.Event(), threading.Event()
    client = object.__new__(MlxVlmClient)

    def predict(*args: object, **kwargs: object) -> str:
        """用事件精确控制在途推理，不依赖模型下载或 GPU。"""
        entered.set()
        assert release.wait(5)
        return "done"

    client.predict = predict
    client._batch_predict = predict

    async def run() -> None:
        """重复取消仍须等待线程，异常完成后释放并发名额。"""
        semaphore = asyncio.Semaphore(1)
        call = client.aio_batch_predict([None], semaphore=semaphore) if batch else client.aio_predict(None)
        task = asyncio.create_task(call)
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            if batch:
                assert semaphore.locked()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not semaphore.locked()

    asyncio.run(run())


def test_vllm_async_cancellation_reaches_generate_cleanup() -> None:
    """取消异步适配器必须传递至引擎生成器，使引擎有机会执行 abort。"""
    from mineru_vl_utils.vlm_client.vllm_async_engine_client import VllmAsyncEngineVlmClient

    async def run() -> None:
        """用真实适配器与受控引擎生成器检查取消传播。"""
        entered = asyncio.Event()
        cleaned = False

        class Engine:
            """提供可观测清理行为的异步生成器。"""

            async def generate(self, **kwargs: object) -> AsyncIterator[object]:
                """在取消时记录引擎自己的清理动作。"""
                nonlocal cleaned
                entered.set()
                try:
                    await asyncio.Event().wait()
                    yield None
                finally:
                    cleaned = True

        client = object.__new__(VllmAsyncEngineVlmClient)
        client.vllm_async_llm = Engine()
        client.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt")
        client.build_messages = lambda *args: []
        client.build_vllm_sampling_params = lambda *args: SimpleNamespace()
        task = asyncio.create_task(client.aio_predict(None))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned

    asyncio.run(run())


def test_thread_failure_after_cancel_is_drained_without_unhandled_log() -> None:
    """Python 3.14 取消后线程失败不得重复记录已处理的异常。"""
    from mineru_vl_utils.vlm_client.utils import run_in_thread_until_complete

    async def run() -> None:
        """记录循环异常回调，并在重复取消后释放真实工作线程。"""
        entered, release = threading.Event(), threading.Event()
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))

        def fail() -> None:
            """让故障在等待者取消之后才从线程返回。"""
            entered.set()
            assert release.wait(3)
            raise RuntimeError("failure after cancellation")

        task = asyncio.create_task(run_in_thread_until_complete(fail))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)
            assert not unhandled
        finally:
            release.set()

    asyncio.run(run())
