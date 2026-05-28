import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Sequence

from loguru import logger

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput

from .base_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    ImageType,
    RequestError,
    SamplingParams,
    ScoredOutput,
    ServerError,
    UnsupportedError,
    VlmClient,
    compute_confidence_metrics,
)
from .utils import aio_image_to_obj_list, gather_tasks
from .vllm_engine_client import _build_raw_vllm_prompt, _patch_vllm_logprobs_overflow

_patch_vllm_logprobs_overflow()


class VllmAsyncEngineVlmClient(VlmClient):
    def __init__(
        self,
        vllm_async_llm,  # vllm.v1.engine.async_llm.AsyncLLM instance
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        max_concurrency: int = 100,
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
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError:
            raise ImportError("Please install vllm to use VllmEngineVlmClient.")

        if not vllm_async_llm:
            raise ValueError("vllm_async_llm is None.")
        if not isinstance(vllm_async_llm, AsyncLLM):
            raise ValueError(f"vllm_async_llm must be an instance of {AsyncLLM}")

        self.vllm_async_llm = vllm_async_llm
        if vllm_async_llm.tokenizer is None:
            raise ValueError("vllm_async_llm.tokenizer is None.")

        tokenizer = vllm_async_llm.tokenizer
        if hasattr(tokenizer, "get_lora_tokenizer"):
            tokenizer = tokenizer.get_lora_tokenizer()  # type: ignore

        self.tokenizer = tokenizer
        self.model_max_length = vllm_async_llm.model_config.max_model_len
        self.VllmSamplingParams = SamplingParams
        self.VllmRequestOutputKind = RequestOutputKind
        self.max_concurrency = max_concurrency
        self.debug = debug

    def build_messages(self, prompt: str, num_images: int) -> list[dict]:
        prompt = prompt or self.prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if "<image>" in prompt:
            prompt_parts = prompt.split("<image>", num_images)
            user_messages = []
            for i in range(max(len(prompt_parts), num_images)):
                if i < len(prompt_parts) and prompt_parts[i]:
                    user_messages.append({"type": "text", "text": prompt_parts[i]})
                if i < num_images:
                    user_messages.append({"type": "image"})
        elif self.text_before_image:
            user_messages = [
                {"type": "text", "text": prompt},
                *({"type": "image"} for _ in range(num_images)),
            ]
        else:  # image before text, which is the default behavior.
            user_messages = [
                *({"type": "image"} for _ in range(num_images)),
                {"type": "text", "text": prompt},
            ]
        messages.append({"role": "user", "content": user_messages})
        return messages

    def build_vllm_sampling_params(self, sampling_params: SamplingParams | None):
        sp = self.build_sampling_params(sampling_params)

        vllm_sp_dict = {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "presence_penalty": sp.presence_penalty,
            "frequency_penalty": sp.frequency_penalty,
            "repetition_penalty": sp.repetition_penalty,
            # max_tokens should smaller than model max length
            "max_tokens": sp.max_new_tokens if sp.max_new_tokens is not None else self.model_max_length,
        }

        if sp.no_repeat_ngram_size is not None:
            vllm_sp_dict["extra_args"] = {
                "no_repeat_ngram_size": sp.no_repeat_ngram_size,
                "debug": self.debug,
            }

        return self.VllmSamplingParams(
            **{k: v for k, v in vllm_sp_dict.items() if v is not None},
            skip_special_tokens=False,
            output_kind=self.VllmRequestOutputKind.FINAL_ONLY,
        )

    def get_output_content(self, output: "RequestOutput") -> str:
        if not output.finished:
            raise ServerError("The output generation was not finished.")

        choices = output.outputs
        if not (isinstance(choices, list) and choices):
            raise ServerError("No choices found in the output.")

        finish_reason = choices[0].finish_reason
        if finish_reason is None:
            raise ServerError("Finish reason is None in the output.")
        if finish_reason == "length":
            if not self.allow_truncated_content:
                raise RequestError("The output was truncated due to length limit.")
            else:
                logger.warning("The output was truncated due to length limit.")
        elif finish_reason != "stop":
            raise RequestError(f"Unexpected finish reason: {finish_reason}")

        return choices[0].text

    async def _render_vllm_cmpl_input(self, raw_prompt: dict[str, Any]) -> dict[str, Any]:
        """优先使用新版 vLLM Renderer 异步预处理 prompt，旧版无 renderer 时保持原输入。"""
        renderer = getattr(self.vllm_async_llm, "renderer", None)
        if renderer is None:
            return raw_prompt

        render_cmpl_async = getattr(renderer, "render_cmpl_async", None)
        if callable(render_cmpl_async):
            rendered_inputs = await render_cmpl_async([raw_prompt])
            return rendered_inputs[0]

        render_cmpl = getattr(renderer, "render_cmpl", None)
        if callable(render_cmpl):
            return render_cmpl([raw_prompt])[0]

        return raw_prompt

    def predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        raise UnsupportedError(
            "Synchronous predict() is not supported in vllm-async-engine VlmClient(backend). "
            "Please use aio_predict() instead. If you intend to use synchronous client, "
            "please use vllm-engine VlmClient(backend)."
        )

    def batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        raise UnsupportedError(
            "Synchronous batch_predict() is not supported in vllm-async-engine VlmClient(backend). "
            "Please use aio_batch_predict() instead. If you intend to use synchronous client, "
            "please use vllm-engine VlmClient(backend)."
        )

    async def aio_predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        image = await aio_image_to_obj_list(image)

        chat_prompt: str = self.tokenizer.apply_chat_template(
            self.build_messages(prompt, len(image)),
            tokenize=False,
            add_generation_prompt=True,
        )

        vllm_sp = self.build_vllm_sampling_params(sampling_params)

        generate_kwargs = {}
        if priority is not None:
            generate_kwargs["priority"] = priority

        vllm_prompt = await self._render_vllm_cmpl_input(_build_raw_vllm_prompt(chat_prompt, image))

        last_output = None
        async for output in self.vllm_async_llm.generate(
            prompt=vllm_prompt,
            sampling_params=vllm_sp,
            request_id=str(uuid.uuid4()),
            **generate_kwargs,
        ):
            last_output = output

        if last_output is None:  # this should not happen
            raise ServerError("No output from the server.")

        return self.get_output_content(last_output)

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

    # --- scored predict (generation PPL) ---

    def _get_output_scored(self, output: "RequestOutput") -> ScoredOutput:
        text = self.get_output_content(output)
        choice = output.outputs[0]
        token_ids = list(choice.token_ids)
        logprobs_list: list[float] = []
        for i, tid in enumerate(token_ids):
            lp_dict = choice.logprobs[i]
            logprobs_list.append(lp_dict[tid].logprob)
        ppl, min_lp, std = compute_confidence_metrics(logprobs_list)
        return ScoredOutput(
            text=text,
            token_ids=token_ids,
            logprobs=logprobs_list,
            perplexity=ppl,
            min_logprob=min_lp,
            logprob_std=std,
        )

    async def aio_predict_scored(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> ScoredOutput:
        image = await aio_image_to_obj_list(image)

        chat_prompt: str = self.tokenizer.apply_chat_template(
            self.build_messages(prompt, len(image)),
            tokenize=False,
            add_generation_prompt=True,
        )

        vllm_sp = self.build_vllm_sampling_params(sampling_params)
        vllm_sp.logprobs = 0

        generate_kwargs = {}
        if priority is not None:
            generate_kwargs["priority"] = priority

        vllm_prompt = await self._render_vllm_cmpl_input(_build_raw_vllm_prompt(chat_prompt, image))

        last_output = None
        async for output in self.vllm_async_llm.generate(
            prompt=vllm_prompt,
            sampling_params=vllm_sp,
            request_id=str(uuid.uuid4()),
            **generate_kwargs,
        ):
            last_output = output

        if last_output is None:
            raise ServerError("No output from the server.")

        return self._get_output_scored(last_output)

    async def aio_batch_predict_scored(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm: bool = False,
        tqdm_desc: str | None = None,
    ) -> list[ScoredOutput]:
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

        async def predict_scored_with_semaphore(
            image: ImageType,
            prompt: str,
            sp: SamplingParams | None,
            prio: int | None,
        ):
            async with semaphore:
                return await self.aio_predict_scored(
                    image=image,
                    prompt=prompt,
                    sampling_params=sp,
                    priority=prio,
                )

        return await gather_tasks(
            tasks=[predict_scored_with_semaphore(*args) for args in zip(images, prompts, sampling_params, priority)],
            use_tqdm=use_tqdm,
            tqdm_desc=tqdm_desc,
        )

    # --- score (evaluation PPL / teacher forcing) ---

    def _build_score_prompt_pair(self, prompt: str, num_images: int, scored_text: str) -> tuple[str, int]:
        """Build prompt for scoring. Returns (full_prompt, scored_token_count)."""
        messages = self.build_messages(prompt, num_images)

        messages_with_assistant = messages + [{"role": "assistant", "content": scored_text}]
        full_prompt: str = self.tokenizer.apply_chat_template(
            messages_with_assistant, tokenize=False, add_generation_prompt=False
        )

        base_prompt: str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        full_token_ids = self.tokenizer.encode(full_prompt)
        base_token_ids = self.tokenizer.encode(base_prompt)
        scored_token_count = len(full_token_ids) - len(base_token_ids)
        if scored_token_count <= 0:
            raise RequestError(f"scored_text tokenizes to {scored_token_count} tokens, expected > 0.")

        return full_prompt, scored_token_count

    def _extract_prompt_logprobs(self, output: "RequestOutput", scored_token_count: int) -> ScoredOutput:
        prompt_logprobs = output.prompt_logprobs
        prompt_token_ids = output.prompt_token_ids

        token_ids: list[int] = []
        logprobs_list: list[float] = []

        start_idx = len(prompt_token_ids) - scored_token_count
        for i in range(start_idx, len(prompt_token_ids)):
            tid = prompt_token_ids[i]
            lp_dict = prompt_logprobs[i]
            if lp_dict is None:
                continue
            token_ids.append(tid)
            if tid in lp_dict:
                logprobs_list.append(lp_dict[tid].logprob)
            elif i > 0 and prompt_token_ids[i - 1] in lp_dict:
                # vLLM v1 async engine bug (chunked prefill): prompt_logprobs dict
                # key is off-by-one — stores ptids[i-1] instead of ptids[i] as key.
                # The logprob VALUE is still P(ptids[i]|context), just the key is wrong.
                logprobs_list.append(lp_dict[prompt_token_ids[i - 1]].logprob)
            else:
                # ultimate fallback: take whichever single logprob is in the dict
                logprobs_list.append(next(iter(lp_dict.values())).logprob)

        ppl, min_lp, std = compute_confidence_metrics(logprobs_list)
        return ScoredOutput(
            text="",  # placeholder, caller sets this
            token_ids=token_ids,
            logprobs=logprobs_list,
            perplexity=ppl,
            min_logprob=min_lp,
            logprob_std=std,
        )

    async def aio_score(
        self,
        image: ImageType,
        scored_text: str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> ScoredOutput:
        image_list = await aio_image_to_obj_list(image)

        full_prompt, scored_token_count = self._build_score_prompt_pair(prompt, len(image_list), scored_text)

        vllm_sp = self.build_vllm_sampling_params(sampling_params)
        vllm_sp.prompt_logprobs = 0
        vllm_sp.max_tokens = 1

        generate_kwargs = {}
        if priority is not None:
            generate_kwargs["priority"] = priority

        vllm_prompt = await self._render_vllm_cmpl_input(_build_raw_vllm_prompt(full_prompt, image_list))

        last_output = None
        async for output in self.vllm_async_llm.generate(
            prompt=vllm_prompt,
            sampling_params=vllm_sp,
            request_id=str(uuid.uuid4()),
            **generate_kwargs,
        ):
            last_output = output

        if last_output is None:
            raise ServerError("No output from the server.")

        scored_output = self._extract_prompt_logprobs(last_output, scored_token_count)
        scored_output.text = scored_text
        return scored_output

    async def aio_batch_score(
        self,
        images: Sequence[ImageType],
        scored_texts: Sequence[str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm: bool = False,
        tqdm_desc: str | None = None,
    ) -> list[ScoredOutput]:
        assert len(scored_texts) == len(images), "Length of scored_texts and images must match."
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

        async def score_with_semaphore(
            image: ImageType,
            scored_text: str,
            prompt: str,
            sp: SamplingParams | None,
            prio: int | None,
        ):
            async with semaphore:
                return await self.aio_score(
                    image=image,
                    scored_text=scored_text,
                    prompt=prompt,
                    sampling_params=sp,
                    priority=prio,
                )

        return await gather_tasks(
            tasks=[
                score_with_semaphore(*args)
                for args in zip(
                    images,
                    scored_texts,
                    prompts,
                    sampling_params,
                    priority,
                )
            ],
            use_tqdm=use_tqdm,
            tqdm_desc=tqdm_desc,
        )
