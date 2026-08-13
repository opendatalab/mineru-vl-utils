import asyncio
import math
from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias

from PIL import Image

SingleImageType: TypeAlias = Image.Image | bytes | str
ImageType: TypeAlias = SingleImageType | Sequence[SingleImageType] | None

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_USER_PROMPT = "What is the text in the illustrate?"


class UnsupportedError(NotImplementedError):
    pass


class RequestError(ValueError):
    pass


class ServerError(RuntimeError):
    pass


@dataclass
class SamplingParams:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None  # not supported by hf
    frequency_penalty: float | None = None  # not supported by hf
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    max_new_tokens: int | None = None


@dataclass
class ScoredOutput:
    """Output with confidence metrics.

    Generation PPL: text is the model-generated text, logprobs/token_ids correspond to the generated token sequence.
    Evaluation PPL: text is the input scored_text returned as-is, logprobs/token_ids correspond to that text's token sequence.
    """

    text: str
    token_ids: list[int]
    logprobs: list[float]
    perplexity: float
    min_logprob: float
    logprob_std: float


def compute_confidence_metrics(
    logprobs: list[float],
) -> tuple[float, float, float]:
    """Compute confidence metrics. Returns (perplexity, min_logprob, logprob_std)."""
    if not logprobs:
        return (float("inf"), float("-inf"), 0.0)
    perplexity = math.exp(-sum(logprobs) / len(logprobs))
    min_lp = min(logprobs)
    mean = sum(logprobs) / len(logprobs)
    std = math.sqrt(sum((lp - mean) ** 2 for lp in logprobs) / len(logprobs))
    return (perplexity, min_lp, std)


class VlmClient:
    def __init__(
        self,
        *,
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
    ) -> None:
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.sampling_params = sampling_params
        self.text_before_image = text_before_image
        self.allow_truncated_content = allow_truncated_content

    def build_sampling_params(
        self,
        sampling_params: SamplingParams | None,
    ) -> SamplingParams:
        if self.sampling_params:
            temperature = self.sampling_params.temperature
            top_p = self.sampling_params.top_p
            top_k = self.sampling_params.top_k
            presence_penalty = self.sampling_params.presence_penalty
            frequency_penalty = self.sampling_params.frequency_penalty
            repetition_penalty = self.sampling_params.repetition_penalty
            no_repeat_ngram_size = self.sampling_params.no_repeat_ngram_size
            max_new_tokens = self.sampling_params.max_new_tokens
        else:
            temperature = None
            top_p = None
            top_k = None
            presence_penalty = None
            frequency_penalty = None
            repetition_penalty = None
            no_repeat_ngram_size = None
            max_new_tokens = None

        if sampling_params:
            if sampling_params.temperature is not None:
                temperature = sampling_params.temperature
            if sampling_params.top_p is not None:
                top_p = sampling_params.top_p
            if sampling_params.top_k is not None:
                top_k = sampling_params.top_k
            if sampling_params.presence_penalty is not None:
                presence_penalty = sampling_params.presence_penalty
            if sampling_params.frequency_penalty is not None:
                frequency_penalty = sampling_params.frequency_penalty
            if sampling_params.repetition_penalty is not None:
                repetition_penalty = sampling_params.repetition_penalty
            if sampling_params.no_repeat_ngram_size is not None:
                no_repeat_ngram_size = sampling_params.no_repeat_ngram_size
            if sampling_params.max_new_tokens is not None:
                max_new_tokens = sampling_params.max_new_tokens

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )

    def predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        raise NotImplementedError()

    def batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        raise NotImplementedError()

    async def aio_predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        raise NotImplementedError()

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
        raise NotImplementedError()

    # --- scored predict (generation PPL) ---

    def predict_scored(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> ScoredOutput:
        raise UnsupportedError(f"predict_scored() is not supported by {type(self).__name__}.")

    def batch_predict_scored(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[ScoredOutput]:
        raise UnsupportedError(f"batch_predict_scored() is not supported by {type(self).__name__}.")

    async def aio_predict_scored(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> ScoredOutput:
        raise UnsupportedError(f"aio_predict_scored() is not supported by {type(self).__name__}.")

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
        raise UnsupportedError(f"aio_batch_predict_scored() is not supported by {type(self).__name__}.")

    # --- score (evaluation PPL / teacher forcing) ---

    def score(
        self,
        image: ImageType,
        scored_text: str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> ScoredOutput:
        raise UnsupportedError(f"score() is not supported by {type(self).__name__}.")

    def batch_score(
        self,
        images: Sequence[ImageType],
        scored_texts: Sequence[str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[ScoredOutput]:
        raise UnsupportedError(f"batch_score() is not supported by {type(self).__name__}.")

    async def aio_score(
        self,
        image: ImageType,
        scored_text: str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> ScoredOutput:
        raise UnsupportedError(f"aio_score() is not supported by {type(self).__name__}.")

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
        raise UnsupportedError(f"aio_batch_score() is not supported by {type(self).__name__}.")


def new_vlm_client(
    backend: Literal[
        "http-client",
        "transformers",
        "mlx-engine",
        "lmdeploy-engine",
        "vllm-engine",
        "vllm-async-engine",
        "llama-cpp-engine",
    ],
    model_name: str | None = None,
    server_url: str | None = None,
    server_headers: dict[str, str] | None = None,
    model=None,  # transformers model
    processor=None,  # transformers processor
    lmdeploy_engine=None,  # lmdeploy.serve.vl_async_engine.VLAsyncEngine instance
    vllm_llm=None,  # vllm.LLM model
    vllm_async_llm=None,  # vllm.v1.engine.async_llm.AsyncLLM instance
    llama_cpp_engine=None,  # mineru_llama_cpp.Engine instance
    prompt: str = DEFAULT_USER_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    sampling_params: SamplingParams | None = None,
    text_before_image: bool = False,
    allow_truncated_content: bool = False,
    max_concurrency: int = 100,
    batch_size: int = 0,
    http_timeout: int = 600,
    connect_timeout: int = 10,
    max_connections: int | None = None,
    max_keepalive_connections: int | None = 20,
    keepalive_expiry: float | None = 5,
    use_tqdm: bool = True,
    debug: bool = False,
    max_retries: int = 3,
    retry_backoff_factor: float = 0.5,
    skip_model_name_checking: bool = False,
) -> VlmClient:
    if backend == "http-client":
        from .http_client import HttpVlmClient

        return HttpVlmClient(
            model_name=model_name,
            server_url=server_url,
            server_headers=server_headers,
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
            max_concurrency=max_concurrency,
            http_timeout=http_timeout,
            connect_timeout=connect_timeout,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
            debug=debug,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            skip_model_name_checking=skip_model_name_checking,
        )

    elif backend == "transformers":
        from .transformers_client import TransformersVlmClient

        return TransformersVlmClient(
            model=model,
            processor=processor,
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
            batch_size=batch_size,
            use_tqdm=use_tqdm,
        )

    elif backend == "mlx-engine":
        from .mlx_client import MlxVlmClient

        return MlxVlmClient(
            model=model,
            processor=processor,
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
            batch_size=batch_size,
            use_tqdm=use_tqdm,
        )

    elif backend == "lmdeploy-engine":
        from .lmdeploy_engine_client import LmdeployEngineVlmClient

        return LmdeployEngineVlmClient(
            lmdeploy_engine=lmdeploy_engine,
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            use_tqdm=use_tqdm,
            debug=debug,
        )

    elif backend == "vllm-engine":
        from .vllm_engine_client import VllmEngineVlmClient

        return VllmEngineVlmClient(
            vllm_llm=vllm_llm,
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
            batch_size=batch_size,
            use_tqdm=use_tqdm,
            debug=debug,
        )

    elif backend == "vllm-async-engine":
        from .vllm_async_engine_client import VllmAsyncEngineVlmClient

        return VllmAsyncEngineVlmClient(
            vllm_async_llm=vllm_async_llm,
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
            max_concurrency=max_concurrency,
            debug=debug,
        )

    elif backend == "llama-cpp-engine":
        from .llama_cpp_engine_client import LlamaCppEngineVlmClient

        return LlamaCppEngineVlmClient(
            llama_cpp_engine=llama_cpp_engine,
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
            max_concurrency=max_concurrency,
            debug=debug,
        )

    else:
        raise ValueError(f"Unsupported backend: {backend}")
