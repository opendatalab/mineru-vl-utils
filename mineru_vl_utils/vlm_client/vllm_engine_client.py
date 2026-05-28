import asyncio
from typing import TYPE_CHECKING, Any, Sequence

from loguru import logger

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams as VllmSamplingParams

from PIL import Image

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
from .utils import image_to_obj_list


def _patch_vllm_logprobs_overflow():
    """
    Workaround for vllm bug: multimodal image token IDs can be out of range
    for the tokenizer, causing OverflowError in convert_ids_list_to_tokens.
    Patches vllm.v1.engine.logprobs module in-place.
    """
    try:
        import vllm.v1.engine.logprobs as _logprobs_mod
        from vllm.transformers_utils.tokenizer import AnyTokenizer

        def _safe_convert_ids_list_to_tokens(
            tokenizer: AnyTokenizer,
            token_ids: list[int],
        ) -> list[str]:
            token_str_lst = []
            for token_id in token_ids:
                try:
                    token_str = tokenizer.decode([token_id])
                    if token_str is None:
                        token_str = ""
                except (OverflowError, ValueError):
                    token_str = ""
                token_str_lst.append(token_str)
            return token_str_lst

        _logprobs_mod.convert_ids_list_to_tokens = _safe_convert_ids_list_to_tokens
    except Exception:
        pass  # vllm not installed or structure changed, skip patch


_patch_vllm_logprobs_overflow()


def _build_raw_vllm_prompt(chat_prompt: str, image_list: list[Image.Image]) -> dict[str, Any]:
    """构造兼容旧版 vLLM 的 raw prompt，供 renderer 不可用时回退。"""
    vllm_prompt: dict[str, Any] = {"prompt": chat_prompt}
    if image_list:
        vllm_prompt["multi_modal_data"] = {"image": image_list}
    return vllm_prompt


class VllmEngineVlmClient(VlmClient):
    def __init__(
        self,
        vllm_llm,  # vllm.LLM instance
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        batch_size: int = 0,
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
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Please install vllm to use VllmEngineVlmClient.")

        if not vllm_llm:
            raise ValueError("vllm_llm is None.")
        if not isinstance(vllm_llm, LLM):
            raise ValueError("vllm_llm must be an instance of vllm.LLM.")

        self.vllm_llm = vllm_llm
        self.tokenizer = vllm_llm.get_tokenizer()
        self.model_max_length = vllm_llm.llm_engine.model_config.max_model_len
        self.VllmSamplingParams = SamplingParams
        self.batch_size = batch_size
        self.use_tqdm = use_tqdm
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

    def _render_vllm_cmpl_inputs(self, raw_prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """优先使用新版 vLLM Renderer 预处理 prompt，旧版无 renderer 时保持原输入。"""
        renderer = getattr(self.vllm_llm, "renderer", None)
        render_cmpl = getattr(renderer, "render_cmpl", None) if renderer is not None else None
        if callable(render_cmpl):
            return render_cmpl(raw_prompts)
        return raw_prompts

    def predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        return self.batch_predict([image], [prompt], [sampling_params])[0]

    def batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        if not isinstance(prompts, str):
            assert len(prompts) == len(images), "Length of prompts and images must match."
        if isinstance(sampling_params, Sequence):
            assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        if isinstance(priority, Sequence):
            assert len(priority) == len(images), "Length of priority and images must match."
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        image_lists = [image_to_obj_list(image) for image in images]

        chat_prompts: list[str] = [
            self.tokenizer.apply_chat_template(
                self.build_messages(prompt, len(image_list)),
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt, image_list in zip(prompts, image_lists)
        ]

        if not isinstance(sampling_params, Sequence):
            vllm_sp_list = [self.build_vllm_sampling_params(sampling_params)] * len(images)
        else:
            vllm_sp_list = [self.build_vllm_sampling_params(sp) for sp in sampling_params]

        outputs = []
        batch_size = self.batch_size if self.batch_size > 0 else len(images)
        batch_size = max(1, batch_size)

        for i in range(0, len(images), batch_size):
            batch_image_lists = image_lists[i : i + batch_size]
            batch_chat_prompts = chat_prompts[i : i + batch_size]
            batch_sp_list = vllm_sp_list[i : i + batch_size]
            batch_outputs = self._predict_one_batch(batch_image_lists, batch_chat_prompts, batch_sp_list)
            outputs.extend(batch_outputs)

        return outputs

    def _predict_one_batch(
        self,
        image_lists: list[list[Image.Image]],
        chat_prompts: list[str],
        vllm_sampling_params: list["VllmSamplingParams"],
    ):
        raw_prompts = [
            _build_raw_vllm_prompt(chat_prompt, image)
            for chat_prompt, image in zip(chat_prompts, image_lists)
        ]
        vllm_prompts = self._render_vllm_cmpl_inputs(raw_prompts)

        outputs = self.vllm_llm.generate(
            prompts=vllm_prompts,  # type: ignore
            sampling_params=vllm_sampling_params,
            use_tqdm=self.use_tqdm,
        )

        return [self.get_output_content(output) for output in outputs]

    # --- scored predict (generation PPL) ---

    def get_output_scored(self, output: "RequestOutput") -> ScoredOutput:
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

    def predict_scored(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> ScoredOutput:
        return self.batch_predict_scored(
            [image],
            [prompt],
            [sampling_params],
        )[0]

    def batch_predict_scored(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[ScoredOutput]:
        if not isinstance(prompts, str):
            assert len(prompts) == len(images), "Length of prompts and images must match."
        if isinstance(sampling_params, Sequence):
            assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        image_lists = [image_to_obj_list(image) for image in images]

        chat_prompts: list[str] = [
            self.tokenizer.apply_chat_template(
                self.build_messages(prompt, len(image_list)),
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt, image_list in zip(prompts, image_lists)
        ]

        if not isinstance(sampling_params, Sequence):
            vllm_sp_list = [self.build_vllm_sampling_params(sampling_params)] * len(images)
        else:
            vllm_sp_list = [self.build_vllm_sampling_params(sp) for sp in sampling_params]

        # Enable logprobs on all sampling params
        for vllm_sp in vllm_sp_list:
            vllm_sp.logprobs = 0

        results: list[ScoredOutput] = []
        batch_size = self.batch_size if self.batch_size > 0 else len(images)
        batch_size = max(1, batch_size)

        for i in range(0, len(images), batch_size):
            batch_image_lists = image_lists[i : i + batch_size]
            batch_chat_prompts = chat_prompts[i : i + batch_size]
            batch_sp_list = vllm_sp_list[i : i + batch_size]

            raw_prompts = [
                _build_raw_vllm_prompt(chat_prompt, image)
                for chat_prompt, image in zip(batch_chat_prompts, batch_image_lists)
            ]
            vllm_prompts = self._render_vllm_cmpl_inputs(raw_prompts)

            outputs = self.vllm_llm.generate(
                prompts=vllm_prompts,  # type: ignore
                sampling_params=batch_sp_list,
                use_tqdm=self.use_tqdm,
            )

            results.extend(self.get_output_scored(output) for output in outputs)

        return results

    # --- score (evaluation PPL / teacher forcing) ---

    def _build_score_prompt_pair(self, prompt: str, image_list: list[Image.Image], scored_text: str) -> tuple[str, int]:
        """Build prompt for scoring. Returns (full_prompt, scored_token_count)."""
        messages = self.build_messages(prompt, len(image_list))

        # Full prompt with assistant turn
        messages_with_assistant = messages + [{"role": "assistant", "content": scored_text}]
        full_prompt: str = self.tokenizer.apply_chat_template(
            messages_with_assistant, tokenize=False, add_generation_prompt=False
        )

        # Base prompt without assistant turn
        base_prompt: str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Compute token count for the scored_text portion.
        # Note: scored_token_count includes the <|im_end|> stop token AND the
        # trailing "\n" that apply_chat_template appends as a turn separator.
        # That trailing "\n" is not generated by the model during inference, so
        # score() reports one extra token compared to predict_scored(). See the
        # note in _extract_prompt_logprobs for the full explanation.
        full_token_ids = self.tokenizer.encode(full_prompt)
        base_token_ids = self.tokenizer.encode(base_prompt)
        scored_token_count = len(full_token_ids) - len(base_token_ids)
        if scored_token_count <= 0:
            raise RequestError(f"scored_text tokenizes to {scored_token_count} tokens, expected > 0.")

        return full_prompt, scored_token_count

    def _extract_prompt_logprobs(self, output: "RequestOutput", scored_token_count: int) -> ScoredOutput:
        """Extract logprobs for the scored_text portion from prompt_logprobs."""
        prompt_logprobs = output.prompt_logprobs
        prompt_token_ids = output.prompt_token_ids

        # NOTE: token_ids collected here (from prompt_token_ids) may NOT match
        # the token_ids returned by predict_scored() for the same text. This is
        # a known re-tokenization effect: vllm generates tokens autoregressively
        # (each token chosen in context), while score() re-encodes the decoded
        # text string through apply_chat_template + tokenizer.encode. BPE
        # tokenization is context-sensitive, so the same character sequence can
        # split into different tokens depending on the encoding path. As a result,
        # score() and predict_scored() may report slightly different token counts
        # and perplexity values for the same output text — this is expected and
        # not a bug.
        #
        # Additionally, apply_chat_template appends a trailing "\n" after the
        # <|im_end|> stop token as a turn separator. That trailing "\n" is
        # included in scored_token_count but is NOT generated by the model during
        # inference, so it slightly inflates the score() token count (+1) vs
        # predict_scored(). This is a minor cosmetic difference; the PPL impact
        # is negligible since "\n" after <|im_end|> has near-zero logprob.

        # Extract logprobs from the last scored_token_count positions.
        # prompt_logprobs[0] is always None (first token has no prior context).
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
            else:
                # vllm v1 bug: lp_dict key is incorrect (e.g. always 0 or overflowed
                # image token ID) when prompt_logprobs=0, but the single logprob value
                # is still the actual token's logprob at this position.
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

    def score(
        self,
        image: ImageType,
        scored_text: str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> ScoredOutput:
        return self.batch_score(
            [image],
            [scored_text],
            [prompt],
            [sampling_params],
        )[0]

    def batch_score(
        self,
        images: Sequence[ImageType],
        scored_texts: Sequence[str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[ScoredOutput]:
        assert len(scored_texts) == len(images), "Length of scored_texts and images must match."
        if not isinstance(prompts, str):
            assert len(prompts) == len(images), "Length of prompts and images must match."
        if isinstance(sampling_params, Sequence):
            assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        image_lists = [image_to_obj_list(image) for image in images]

        # Build score prompts and get scored token counts
        chat_prompts: list[str] = []
        scored_token_counts: list[int] = []
        for prompt, image_list, scored_text in zip(prompts, image_lists, scored_texts):
            full_prompt, scored_token_count = self._build_score_prompt_pair(prompt, image_list, scored_text)
            chat_prompts.append(full_prompt)
            scored_token_counts.append(scored_token_count)

        if not isinstance(sampling_params, Sequence):
            vllm_sp_list = [self.build_vllm_sampling_params(sampling_params)] * len(images)
        else:
            vllm_sp_list = [self.build_vllm_sampling_params(sp) for sp in sampling_params]

        # Enable prompt_logprobs and minimize generation
        for vllm_sp in vllm_sp_list:
            vllm_sp.prompt_logprobs = 0
            vllm_sp.max_tokens = 1

        results: list[ScoredOutput] = []
        batch_size = self.batch_size if self.batch_size > 0 else len(images)
        batch_size = max(1, batch_size)

        for i in range(0, len(images), batch_size):
            batch_image_lists = image_lists[i : i + batch_size]
            batch_chat_prompts = chat_prompts[i : i + batch_size]
            batch_sp_list = vllm_sp_list[i : i + batch_size]
            batch_scored_texts = scored_texts[i : i + batch_size]
            batch_scored_token_counts = scored_token_counts[i : i + batch_size]

            raw_prompts = [
                _build_raw_vllm_prompt(chat_prompt, image)
                for chat_prompt, image in zip(batch_chat_prompts, batch_image_lists)
            ]
            vllm_prompts = self._render_vllm_cmpl_inputs(raw_prompts)

            outputs = self.vllm_llm.generate(
                prompts=vllm_prompts,  # type: ignore
                sampling_params=batch_sp_list,
                use_tqdm=self.use_tqdm,
            )

            for output, scored_text, scored_token_count in zip(outputs, batch_scored_texts, batch_scored_token_counts):
                scored_output = self._extract_prompt_logprobs(output, scored_token_count)
                scored_output.text = scored_text
                results.append(scored_output)

        return results

    # --- async stubs (not supported for sync engine) ---

    async def aio_predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        raise UnsupportedError(
            "Asynchronous aio_predict() is not supported in vllm-engine VlmClient(backend). "
            "Please use predict() instead. If you intend to use asynchronous client, "
            "please use vllm-async-engine VlmClient(backend)."
        )

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
        raise UnsupportedError(
            "Asynchronous aio_batch_predict() is not supported in vllm-engine VlmClient(backend). "
            "Please use batch_predict() instead. If you intend to use asynchronous client, "
            "please use vllm-async-engine VlmClient(backend)."
        )
