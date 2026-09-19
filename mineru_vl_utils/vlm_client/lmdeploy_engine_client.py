import asyncio
from io import BytesIO
from itertools import groupby
from typing import Any, Sequence

from PIL import Image
from tqdm import tqdm

from .base_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    ImageType,
    SamplingParams,
    ServerError,
    SingleImageType,
    UnsupportedError,
    VlmClient,
)
from .utils import VLM_PREDICT_DESC, gather_tasks, get_rgb_image, load_resource, run_in_thread_until_complete


class LmdeployEngineVlmClient(VlmClient):
    def __init__(
        self,
        lmdeploy_engine,  # LMDeploy 0.17 的公开 Pipeline 实例
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        batch_size: int = 0,  # batch size for sync predict
        max_concurrency: int = 100,  # max concurrency for async predict
        use_tqdm: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )

        try:
            from lmdeploy import GenerationConfig
            from lmdeploy.pipeline import Pipeline
        except ImportError:
            raise ImportError("Please install lmdeploy to use LmdeployEngineVlmClient.")

        if not lmdeploy_engine:
            raise ValueError("lmdeploy_engine is None.")
        if not isinstance(lmdeploy_engine, Pipeline):
            raise ValueError("lmdeploy_engine must be an instance of lmdeploy.pipeline.Pipeline.")

        self.lmdeploy_engine = lmdeploy_engine
        self.model_max_length = lmdeploy_engine.backend_config.session_len
        self.LmdeployGenerationConfig = GenerationConfig
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.use_tqdm = use_tqdm
        self.debug = debug

    def build_lmdeploy_generation_config(self, sampling_params: SamplingParams | None):
        sp = self.build_sampling_params(sampling_params)

        do_sample = ((sp.temperature or 0.0) > 0.0) and ((sp.top_k or 1) > 1)

        lmdeploy_sp_dict = {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "repetition_penalty": sp.repetition_penalty,
            # WARNING - engine.py:606: num tokens is larger than max session len xxx. Update max_new_tokens=xxx.
            "max_new_tokens": sp.max_new_tokens if sp.max_new_tokens is not None else self.model_max_length,
        }

        return self.LmdeployGenerationConfig(
            **{k: v for k, v in lmdeploy_sp_dict.items() if v is not None},
            do_sample=do_sample,
            skip_special_tokens=False,
        )

    def predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        return self.batch_predict(
            [image],  # type: ignore
            [prompt],
            [sampling_params],
            [priority],
        )[0]

    def batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        """同步批量推理使用实例进度配置，不修改共享客户端状态。"""
        return self._batch_predict(
            images,
            prompts,
            sampling_params,
            priority,
            use_tqdm=self.use_tqdm,
        )

    def _batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        *,
        use_tqdm: bool,
    ) -> list[str]:
        """使用调用级进度选项执行原有批处理，避免并发调用串用开关。"""
        if not isinstance(prompts, str):
            assert len(prompts) == len(images), "Length of prompts and images must match."
        if isinstance(sampling_params, Sequence):
            assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        if isinstance(priority, Sequence):
            assert len(priority) == len(images), "Length of priority and images must match."

        image_objs: list[Image.Image | None] = []
        for image in images:
            if image is None:
                image_objs.append(None)
                continue
            if not isinstance(image, SingleImageType):
                raise UnsupportedError("LmdeployEngineVlmClient haven't support non-single image yet.")
            if isinstance(image, str):
                image = load_resource(image)
            if not isinstance(image, Image.Image):
                image = Image.open(BytesIO(image))
            image = get_rgb_image(image)
            image_objs.append(image)

        if isinstance(prompts, str):
            chat_prompts: list[str] = [prompts] * len(images)
        else:  # isinstance(prompts, Sequence[str])
            chat_prompts: list[str] = list(prompts)

        if not isinstance(sampling_params, Sequence):
            gen_configs = [self.build_lmdeploy_generation_config(sampling_params)] * len(images)
        else:  # isinstance(sampling_params, Sequence)
            gen_configs = [self.build_lmdeploy_generation_config(sp) for sp in sampling_params]

        outputs = []
        batch_size = self.batch_size if self.batch_size > 0 else len(images)
        batch_size = max(1, batch_size)

        priorities = priority if isinstance(priority, Sequence) else [priority] * len(images)
        # Pipeline 的 priority 作用于整次调用；相邻同优先级请求仍按原顺序批处理。
        for current_priority, group in groupby(
            zip(image_objs, chat_prompts, gen_configs, priorities), key=lambda item: item[3]
        ):
            items = list(group)
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                outputs.extend(
                    self._predict_one_batch(
                        [item[0] for item in batch],
                        [item[1] for item in batch],
                        [item[2] for item in batch],
                        priority=current_priority,
                        use_tqdm=use_tqdm,
                    )
                )

        return outputs

    def _predict_one_batch(
        self,
        image_objs: list[Image.Image | None],
        chat_prompts: list[str],
        gen_configs: list[Any],
        priority: int | None = None,
        *,
        use_tqdm: bool,
    ) -> list[str]:
        """通过公开 Pipeline 接口推理，并将后端错误传播给同步与异步调用方。"""
        lmdeploy_prompts = [(prompt, image) if image is not None else prompt for prompt, image in zip(chat_prompts, image_objs)]
        generate_kwargs = {} if priority is None else {"priority": priority}
        if use_tqdm:
            return self._predict_with_progress(lmdeploy_prompts, gen_configs, priority)
        outputs = self.lmdeploy_engine.infer(
            lmdeploy_prompts,  # type: ignore
            gen_config=gen_configs,
            use_tqdm=False,
            **generate_kwargs,
        )
        if len(outputs) != len(lmdeploy_prompts):
            raise ServerError("LMDeploy returned an unexpected number of responses.")
        if any(getattr(output, "finish_reason", None) == "error" for output in outputs):
            raise ServerError("LMDeploy inference failed.")
        return [output.text for output in outputs]

    def _predict_with_progress(
        self,
        prompts: list[str | tuple[str, Image.Image]],
        gen_configs: list[Any],
        priority: int | None,
    ) -> list[str]:
        """按请求索引聚合多路复用响应流，按完成请求更新进度，并在本批结束后传播响应错误。

        LMDeploy 0.17 对同一请求可能交出多帧 Response：中间帧 finish_reason 为 None，
        仅终止帧携带 stop/length/error 等终止原因；官方 Pipeline.infer 即按 Response.extend()
        语义聚合多帧结果，此处对多请求交错流做相同的按索引聚合。
        """
        aggregated: list[Any | None] = [None] * len(prompts)
        completed: set[int] = set()
        error: ServerError | None = None
        generate_kwargs = {} if priority is None else {"priority": priority}
        with tqdm(total=len(prompts), desc=VLM_PREDICT_DESC) as pbar:
            responses = self.lmdeploy_engine.stream_infer(
                prompts,
                gen_config=gen_configs,
                stream_response=False,
                **generate_kwargs,
            )
            for response in responses:
                index = getattr(response, "index", None)
                if type(index) is not int or not 0 <= index < len(prompts):
                    error = error or ServerError("LMDeploy returned an invalid response index.")
                    continue
                if index in completed:
                    error = error or ServerError("LMDeploy returned data after request completion.")
                    continue
                if not isinstance(getattr(response, "text", None), str):
                    error = error or ServerError("LMDeploy returned an incomplete response.")
                    continue
                current = aggregated[index]
                if current is None:
                    aggregated[index] = response
                    current = response
                else:
                    current.extend(response)
                finish_reason = getattr(current, "finish_reason", None)
                if finish_reason is None:
                    continue  # 中间帧是正常状态，继续等待该请求的终止帧。
                completed.add(index)
                if finish_reason == "error":
                    error = error or ServerError("LMDeploy inference failed.")
                    continue
                pbar.update(1)
            # 不提前中断响应迭代，确保错误响应之后的在途请求仍完成本批清理。
            if error is not None:
                raise error
            if len(completed) != len(prompts):
                raise ServerError("LMDeploy returned an unexpected number of responses.")
        return [response.text for response in aggregated if response is not None]

    async def aio_predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        """在线程中复用 Pipeline；取消时等待在途调用结束，再释放并发名额和共享引擎租约。"""
        outputs = await run_in_thread_until_complete(
            self._batch_predict,
            [image],
            [prompt],
            [sampling_params],
            [priority],
            use_tqdm=False,
        )
        return outputs[0]

    async def aio_batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm=False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        assert len(priority) == len(images), "Length of priority and images must match."

        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        async def predict_with_semaphore(
            image: ImageType,
            prompt: str,
            sampling_params: SamplingParams | None,
            priority: int | None,
        ):
            async with semaphore:
                return await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    priority=priority,
                )

        return await gather_tasks(
            tasks=[
                predict_with_semaphore(*args)
                for args in zip(
                    images,
                    prompts,
                    sampling_params,
                    priority,
                )
            ],
            use_tqdm=use_tqdm,
            tqdm_desc=tqdm_desc,
        )
