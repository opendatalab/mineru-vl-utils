import asyncio
import math
import os
import re
from concurrent.futures import Executor
from contextlib import nullcontext
from typing import Literal, Sequence

from loguru import logger
from PIL import Image

from .post_process import post_process
from .post_process.table_image_processor import (
    TABLE_IMAGE_TOKEN_MAP_KEY,
    build_table_image_map,
    cleanup_table_image_metadata,
    is_absorbed_table_image,
    mark_absorbed_table_images,
    mask_and_encode_table_image,
)
from .structs import BLOCK_TYPES, ContentBlock, ExtractResult, ExtractStr
from .vlm_client import DEFAULT_SYSTEM_PROMPT, SamplingParams, new_vlm_client
from .vlm_client.base_client import ImageType, ScoredOutput
from .vlm_client.utils import gather_tasks, get_png_bytes, get_rgb_image

_layout_re = (
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>"
    r"(?:(<\|rotate_(?:up|right|down|left)\|>))?"
    r"(.*?)(?=<\|box_start\|>|$)"
)


class MinerUSamplingParams(SamplingParams):
    def __init__(
        self,
        temperature: float | None = 0.0,
        top_p: float | None = 0.01,
        top_k: int | None = 1,
        presence_penalty: float | None = 0.0,
        frequency_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        no_repeat_ngram_size: int | None = 100,
        max_new_tokens: int | None = None,
    ):
        super().__init__(
            temperature,
            top_p,
            top_k,
            presence_penalty,
            frequency_penalty,
            repetition_penalty,
            no_repeat_ngram_size,
            max_new_tokens,
        )


DEFAULT_PROMPTS: dict[str, str] = {
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "image": "\nImage Analysis:",
    "chart": "\nImage Analysis:",
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Detection:",
    "[cross_page_table_merge]": "",  # prompt is dynamic, built from table content
}

DEFAULT_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "table": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.005),
    "equation": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "image": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "chart": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[default]": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]": MinerUSamplingParams(),
    "[cross_page_table_merge]": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
}

ANGLE_MAPPING: dict[str, Literal[0, 90, 180, 270]] = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}

IMAGE_ANALYSIS_TYPES = {"image", "chart"}
IMAGE_CAPTION_CONTAINER_TYPES = {"image", "chart", "image_block"}
INTERNAL_BLOCK_THRESHOLD = 0.9
IMAGE_ANALYSIS_MIN_BLOCK_SIZE = 0.1
IMAGE_ANALYSIS_MIN_BLOCK_AREA = 0.01


def _convert_bbox(bbox: Sequence[int] | Sequence[str]) -> list[float] | None:
    bbox = tuple(map(int, bbox))
    if any(coord < 0 or coord > 1000 for coord in bbox):
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return [num / 1000.0 for num in (x1, y1, x2, y2)]


def _parse_angle(tail: str) -> Literal[None, 0, 90, 180, 270]:
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            return angle
    return None


def _parse_merge_prev(tail: str) -> bool:
    return "txt_contd_tgt" in tail


class MinerUClientHelper:
    def __init__(
        self,
        backend: str,
        prompts: dict[str, str],
        sampling_params: dict[str, SamplingParams],
        layout_image_size: tuple[int, int],
        min_image_edge: int,
        max_image_edge_ratio: float,
        simple_post_process: bool,
        handle_equation_block: bool,
        abandon_list: bool,
        abandon_paratext: bool,
        image_analysis: bool,
        debug: bool,
        enable_table_formula_eq_wrap: bool = False,
        enable_cross_page_table_merge: bool = False,
    ) -> None:
        self.backend = backend
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.layout_image_size = layout_image_size
        self.min_image_edge = min_image_edge
        self.max_image_edge_ratio = max_image_edge_ratio
        self.simple_post_process = simple_post_process
        self.handle_equation_block = handle_equation_block
        self.abandon_list = abandon_list
        self.abandon_paratext = abandon_paratext
        self.image_analysis = image_analysis
        self.enable_table_formula_eq_wrap = enable_table_formula_eq_wrap
        self.enable_cross_page_table_merge = enable_cross_page_table_merge
        self.debug = debug

    @staticmethod
    def _bbox_intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return (x2 - x1) * (y2 - y1)

    @classmethod
    def _bbox_cover_ratio(cls, inner: Sequence[float], outer: Sequence[float]) -> float:
        inner_area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
        if inner_area == 0:
            return 0.0
        return cls._bbox_intersection_area(inner, outer) / inner_area

    @classmethod
    def _find_covered_block_indices(
        cls,
        blocks: Sequence[ContentBlock],
        candidate_types: set[str],
        container_types: set[str],
        threshold: float = INTERNAL_BLOCK_THRESHOLD,
    ) -> set[int]:
        container_indices = [idx for idx, block in enumerate(blocks) if block.type in container_types]
        if not container_indices:
            return set()

        covered_indices: set[int] = set()
        for idx, block in enumerate(blocks):
            if block.type not in candidate_types:
                continue
            for container_idx in container_indices:
                if idx == container_idx:
                    continue
                if cls._bbox_cover_ratio(block.bbox, blocks[container_idx].bbox) >= threshold:
                    covered_indices.add(idx)
                    break
        return covered_indices

    @staticmethod
    def _is_eligible_for_image_analysis(block: ContentBlock) -> bool:
        x1, y1, x2, y2 = block.bbox
        width = x2 - x1
        height = y2 - y1
        return (
            width > IMAGE_ANALYSIS_MIN_BLOCK_SIZE and height > IMAGE_ANALYSIS_MIN_BLOCK_SIZE
        ) or width * height > IMAGE_ANALYSIS_MIN_BLOCK_AREA

    def resize_by_need(self, image: Image.Image) -> Image.Image:
        edge_ratio = max(image.size) / min(image.size)
        if edge_ratio > self.max_image_edge_ratio:
            width, height = image.size
            if width > height:
                new_w, new_h = width, math.ceil(width / self.max_image_edge_ratio)
            else:  # width < height
                new_w, new_h = math.ceil(height / self.max_image_edge_ratio), height
            new_image = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
            new_image.paste(image, (int((new_w - width) / 2), int((new_h - height) / 2)))
            image = new_image
        if min(image.size) < self.min_image_edge:
            scale = self.min_image_edge / min(image.size)
            new_w, new_h = math.ceil(image.width * scale), math.ceil(image.height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return image

    def prepare_for_layout(self, image: Image.Image) -> Image.Image | bytes:
        image = get_rgb_image(image)
        image = image.resize(self.layout_image_size, Image.Resampling.BICUBIC)
        if self.backend == "http-client":
            return get_png_bytes(image)
        return image

    def parse_layout_output(self, output: str) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []
        matched = False
        for match in re.finditer(_layout_re, output, re.DOTALL):
            matched = True
            x1, y1, x2, y2, ref_type, rotate_token, tail = match.groups()
            bbox = _convert_bbox((x1, y1, x2, y2))
            if bbox is None:
                logger.warning("Invalid bbox in layout output line: {}", match.group(0))
                continue  # Skip invalid bbox
            ref_type = ref_type.lower()
            if ref_type == "unknown":
                ref_type = "image"
            if ref_type == "inline_formula":
                if self.debug:
                    logger.debug("Skipping inline formula block in layout output: {}", match.group(0))
                continue
            if ref_type not in BLOCK_TYPES:
                logger.warning("Unknown block type in layout output line: {}", match.group(0))
                continue  # Skip unknown block types
            angle = _parse_angle(rotate_token) if rotate_token else None
            if angle is None:
                logger.warning("No angle found in layout output line: {}", match.group(0))
            if ref_type == "text":
                merge_prev = _parse_merge_prev(tail)
                blocks.append(ContentBlock(ref_type, bbox, angle=angle, merge_prev=merge_prev))
            else:
                blocks.append(ContentBlock(ref_type, bbox, angle=angle))
        if not matched and output.strip():
            logger.warning("Layout output does not match expected format: {}", output)
        return self._filter_table_internal_layout_blocks(blocks)

    @classmethod
    def _filter_table_internal_layout_blocks(cls, blocks: list[ContentBlock]) -> list[ContentBlock]:
        internal_block_indices = cls._find_covered_block_indices(
            blocks,
            candidate_types={"text", "equation", "equation_block"},
            container_types={"table"},
        )
        if not internal_block_indices:
            return blocks
        return [block for idx, block in enumerate(blocks) if idx not in internal_block_indices]

    def prepare_for_extract(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
        not_extract_list: list[str] | None = None,
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]:
        internal_caption_indices = self._find_covered_block_indices(
            blocks,
            candidate_types={"image_caption"},
            container_types=IMAGE_CAPTION_CONTAINER_TYPES,
        )
        if internal_caption_indices:
            blocks[:] = [block for idx, block in enumerate(blocks) if idx not in internal_caption_indices]

        non_standalone_visual_indices = self._find_covered_block_indices(
            blocks,
            candidate_types=IMAGE_ANALYSIS_TYPES,
            container_types={"image_block"},
        )

        image = get_rgb_image(image)
        width, height = image.size
        block_images: list[Image.Image | bytes] = []
        prompts: list[str] = []
        sampling_params: list[SamplingParams | None] = []
        indices: list[int] = []
        skip_list = {"list", "equation_block", "image_block"}
        if not self.image_analysis:
            skip_list.update({"image", "chart"})
        if not_extract_list:
            for not_extract_type in not_extract_list:
                if not_extract_type in BLOCK_TYPES:
                    skip_list.add(not_extract_type)

        table_indices = [idx for idx, block in enumerate(blocks) if block.type == "table" and block.type not in skip_list]
        table_to_images = build_table_image_map(blocks, threshold=0.9, table_indices=table_indices)
        absorbed_image_indices = sorted(
            {image_idx for image_indices in table_to_images.values() for image_idx in image_indices}
        )
        mark_absorbed_table_images(blocks, absorbed_image_indices)

        for idx, block in enumerate(blocks):
            if block.type in skip_list:
                continue  # Skip blocks that should not be extracted.
            if block.type == "image" and is_absorbed_table_image(block):
                continue
            if block.type in IMAGE_ANALYSIS_TYPES:
                if idx in non_standalone_visual_indices:
                    continue
                if not self._is_eligible_for_image_analysis(block):
                    continue
            table_image_prepared = False
            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled_bbox)
            if block_image.width < 1 or block_image.height < 1:
                logger.warning("Cropped block image has invalid size {}", block_image.size)
                continue
            if block.type == "table":
                image_indices = table_to_images.get(idx, [])
                image_entries = [(image_idx, blocks[image_idx]) for image_idx in image_indices]
                block_image, token_map = mask_and_encode_table_image(image, block, image_entries, block_image)
                table_image_prepared = True
                if token_map:
                    block[TABLE_IMAGE_TOKEN_MAP_KEY] = token_map
            if not table_image_prepared and block.angle in [90, 180, 270]:
                block_image = block_image.rotate(block.angle, expand=True)
            block_image = self.resize_by_need(block_image)
            if self.backend == "http-client":
                block_image = get_png_bytes(block_image)
            block_images.append(block_image)
            prompt = self.prompts.get(block.type) or self.prompts["[default]"]
            prompts.append(prompt)
            params = self.sampling_params.get(block.type) or self.sampling_params.get("[default]")
            sampling_params.append(params)
            indices.append(idx)
        return block_images, prompts, sampling_params, indices

    def post_process(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        try:
            return post_process(
                blocks,
                simple_post_process=self.simple_post_process,
                handle_equation_block=self.handle_equation_block,
                abandon_list=self.abandon_list,
                abandon_paratext=self.abandon_paratext,
                enable_table_formula_eq_wrap=self.enable_table_formula_eq_wrap,
                debug=self.debug,
            )
        except Exception as e:
            logger.warning("Post-processing failed with error: {}", e)
            clean_blocks = [block for block in blocks if not (block.type == "image" and is_absorbed_table_image(block))]
            return cleanup_table_image_metadata(clean_blocks)

    def batch_prepare_for_layout(
        self,
        executor: Executor | None,
        images: list[Image.Image],
    ) -> list[Image.Image | bytes]:
        if executor is None:
            return [self.prepare_for_layout(im) for im in images]
        return list(executor.map(self.prepare_for_layout, images))

    def batch_parse_layout_output(
        self,
        executor: Executor | None,
        outputs: list[str],
    ) -> list[list[ContentBlock]]:
        if executor is None:
            return [self.parse_layout_output(output) for output in outputs]
        return list(executor.map(self.parse_layout_output, outputs))

    def batch_prepare_for_extract(
        self,
        executor: Executor | None,
        images: list[Image.Image],
        blocks_list: Sequence[list[ContentBlock]],
        not_extract_list: list[str] | None = None,
    ) -> list[tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]]:
        if executor is None:
            return [self.prepare_for_extract(im, bls, not_extract_list) for im, bls in zip(images, blocks_list)]
        return list(executor.map(self.prepare_for_extract, images, blocks_list, [not_extract_list] * len(images)))

    def batch_post_process(
        self,
        executor: Executor | None,
        blocks_list: Sequence[list[ContentBlock]],
    ) -> list[list[ContentBlock]]:
        if executor is None:
            return [self.post_process(blocks) for blocks in blocks_list]
        return list(executor.map(self.post_process, blocks_list))

    async def aio_prepare_for_layout(
        self,
        executor: Executor | None,
        image: Image.Image,
    ) -> Image.Image | bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.prepare_for_layout, image)

    async def aio_parse_layout_output(
        self,
        executor: Executor | None,
        output: str,
    ) -> list[ContentBlock]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.parse_layout_output, output)

    async def aio_prepare_for_extract(
        self,
        executor: Executor | None,
        image: Image.Image,
        blocks: list[ContentBlock],
        not_extract_list: list[str] | None = None,
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.prepare_for_extract, image, blocks, not_extract_list)

    async def aio_post_process(
        self,
        executor: Executor | None,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.post_process, blocks)


class _PredictResult:
    """Internal normalized result from predict / predict_scored."""

    __slots__ = ("text", "scored")

    def __init__(self, text: str, scored: ScoredOutput | None = None) -> None:
        self.text = text
        self.scored = scored


class MinerUClient:
    def __init__(
        self,
        backend: Literal[
            "http-client",
            "transformers",
            "mlx-engine",
            "lmdeploy-engine",
            "vllm-engine",
            "vllm-async-engine",
        ],
        model_name: str | None = None,
        server_url: str | None = None,
        server_headers: dict[str, str] | None = None,
        model=None,  # transformers model
        processor=None,  # transformers processor
        vllm_llm=None,  # vllm.LLM model
        vllm_async_llm=None,  # vllm.v1.engine.async_llm.AsyncLLM instance
        lmdeploy_engine=None,  # lmdeploy.serve.vl_async_engine.VLAsyncEngine instance
        model_path: str | None = None,
        prompts: dict[str, str] = DEFAULT_PROMPTS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: dict[str, SamplingParams] = DEFAULT_SAMPLING_PARAMS,
        layout_image_size: tuple[int, int] = (1036, 1036),
        min_image_edge: int = 28,
        max_image_edge_ratio: float = 50,
        simple_post_process: bool = False,
        handle_equation_block: bool = True,
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        image_analysis: bool = False,
        incremental_priority: bool = False,
        max_concurrency: int = 100,
        executor: Executor | None = None,
        batch_size: int = 0,  # for transformers and vllm-engine
        http_timeout: int = 600,  # for http-client backend only
        connect_timeout: int = 10,  # for http-client backend only
        max_connections: int | None = None,  # for http-client backend only
        max_keepalive_connections: int | None = 20,  # for http-client backend only
        keepalive_expiry: float | None = 5,  # for http-client backend only
        use_tqdm: bool = True,
        debug: bool = False,
        max_retries: int = 3,  # for http-client backend only
        retry_backoff_factor: float = 0.5,  # for http-client backend only
        skip_model_name_checking: bool = False,
        scored: bool = False,
        enable_table_formula_eq_wrap: bool = False,
        enable_cross_page_table_merge: bool = False,
    ) -> None:
        env_debug_value = os.getenv("MINERU_VL_DEBUG_ENABLE", "")
        if env_debug_value:
            if env_debug_value.lower() in ["true", "1", "yes"]:
                debug = True
            elif env_debug_value.lower() in ["false", "0", "no"]:
                debug = False
            else:
                logger.warning("unknown MINERU_VL_DEBUG_ENABLE config: {}, pass", env_debug_value)

        if backend == "transformers":
            if model is None or processor is None:
                if not model_path:
                    raise ValueError("model_path must be provided when model or processor is None.")

                try:
                    from transformers import (
                        AutoProcessor,
                        Qwen2VLForConditionalGeneration,
                    )
                    from transformers import __version__ as transformers_version
                except ImportError:
                    raise ImportError("Please install transformers to use the transformers backend.")

                if model is None:
                    dtype_key = "torch_dtype"
                    ver_parts = transformers_version.split(".")
                    if len(ver_parts) >= 2 and int(ver_parts[0]) >= 4 and int(ver_parts[1]) >= 56:
                        dtype_key = "dtype"
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        device_map="auto",
                        **{dtype_key: "auto"},  # type: ignore
                    )
                if processor is None:
                    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

        elif backend == "mlx-engine":
            if model is None or processor is None:
                if not model_path:
                    raise ValueError("model_path must be provided when model or processor is None.")
                from mineru_vl_utils.mlx_compat import load_mlx_model

                model, processor = load_mlx_model(model_path)

        elif backend == "lmdeploy-engine":
            if lmdeploy_engine is None:
                if not model_path:
                    raise ValueError("model_path must be provided when lmdeploy_engine is None.")

                try:
                    from lmdeploy.serve.vl_async_engine import VLAsyncEngine
                except ImportError:
                    raise ImportError("Please install lmdeploy to use the lmdeploy-engine backend.")

                lmdeploy_engine = VLAsyncEngine(
                    model_path,
                )

        elif backend == "vllm-engine":
            if vllm_llm is None:
                if not model_path:
                    raise ValueError("model_path must be provided when vllm_llm is None.")

                try:
                    import vllm
                except ImportError:
                    raise ImportError("Please install vllm to use the vllm-engine backend.")

                vllm_llm = vllm.LLM(model_path)

        elif backend == "vllm-async-engine":
            if vllm_async_llm is None:
                if not model_path:
                    raise ValueError("model_path must be provided when vllm_async_llm is None.")

                try:
                    from vllm.engine.arg_utils import AsyncEngineArgs
                    from vllm.v1.engine.async_llm import AsyncLLM
                except ImportError:
                    raise ImportError("Please install vllm to use the vllm-async-engine backend.")

                vllm_async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(model_path))

        self.client = new_vlm_client(
            backend=backend,
            model_name=model_name,
            server_url=server_url,
            server_headers=server_headers,
            model=model,
            processor=processor,
            lmdeploy_engine=lmdeploy_engine,
            vllm_llm=vllm_llm,
            vllm_async_llm=vllm_async_llm,
            system_prompt=system_prompt,
            allow_truncated_content=True,  # Allow truncated content for MinerU
            max_concurrency=max_concurrency,
            batch_size=batch_size,
            http_timeout=http_timeout,
            connect_timeout=connect_timeout,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
            use_tqdm=use_tqdm,
            debug=debug,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            skip_model_name_checking=skip_model_name_checking,
        )
        self.helper = MinerUClientHelper(
            backend=backend,
            prompts=prompts,
            sampling_params=sampling_params,
            layout_image_size=layout_image_size,
            min_image_edge=min_image_edge,
            max_image_edge_ratio=max_image_edge_ratio,
            simple_post_process=simple_post_process,
            handle_equation_block=handle_equation_block,
            abandon_list=abandon_list,
            abandon_paratext=abandon_paratext,
            image_analysis=image_analysis,
            enable_table_formula_eq_wrap=enable_table_formula_eq_wrap,
            enable_cross_page_table_merge=enable_cross_page_table_merge,
            debug=debug,
        )
        self.backend = backend
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.enable_table_formula_eq_wrap = enable_table_formula_eq_wrap
        self.incremental_priority = incremental_priority
        self.max_concurrency = max_concurrency
        self.executor = executor
        self.use_tqdm = use_tqdm
        self.debug = debug
        self.scored = scored

        if backend in ("http-client", "vllm-async-engine", "lmdeploy-engine"):
            self.batching_mode = "concurrent"
        else:  # backend in ("transformers", "vllm-engine")
            self.batching_mode = "stepping"

    # ------------------------------------------------------------------
    # Internal helpers: normalize predict / predict_scored into _PredictResult
    # ------------------------------------------------------------------

    def _resolve_scored(self, scored: bool | None) -> bool:
        return getattr(self, "scored", False) if scored is None else scored

    def _predict(
        self,
        image: ImageType,
        prompt: str,
        params: SamplingParams | None,
        priority: int | None,
        scored: bool | None,
    ) -> _PredictResult:
        if self._resolve_scored(scored):
            so = self.client.predict_scored(image, prompt, params, priority)
            return _PredictResult(so.text, so)
        return _PredictResult(self.client.predict(image, prompt, params, priority))

    def _batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str,
        params: Sequence[SamplingParams | None] | SamplingParams | None,
        priority: Sequence[int | None] | int | None,
        scored: bool | None,
    ) -> list[_PredictResult]:
        if self._resolve_scored(scored):
            return [_PredictResult(so.text, so) for so in self.client.batch_predict_scored(images, prompts, params, priority)]
        return [_PredictResult(t) for t in self.client.batch_predict(images, prompts, params, priority)]

    async def _aio_predict(
        self,
        image: ImageType,
        prompt: str,
        params: SamplingParams | None,
        priority: int | None,
        semaphore: asyncio.Semaphore | None,
        scored: bool | None,
    ) -> _PredictResult:
        async with semaphore if semaphore is not None else nullcontext():
            if self._resolve_scored(scored):
                so = await self.client.aio_predict_scored(image, prompt, params, priority)
                return _PredictResult(so.text, so)
            return _PredictResult(await self.client.aio_predict(image, prompt, params, priority))

    async def _aio_batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str,
        params: Sequence[SamplingParams | None] | SamplingParams | None,
        priority: Sequence[int | None] | int | None,
        semaphore: asyncio.Semaphore | None,
        scored: bool | None,
        use_tqdm: bool = False,
        tqdm_desc: str | None = None,
    ) -> list[_PredictResult]:
        if self._resolve_scored(scored):
            scored_outputs = await self.client.aio_batch_predict_scored(
                images,
                prompts,
                params,
                priority,
                semaphore=semaphore,
                use_tqdm=use_tqdm,
                tqdm_desc=tqdm_desc,
            )
            return [_PredictResult(so.text, so) for so in scored_outputs]
        else:
            texts = await self.client.aio_batch_predict(
                images,
                prompts,
                params,
                priority,
                semaphore=semaphore,
                use_tqdm=use_tqdm,
                tqdm_desc=tqdm_desc,
            )
            return [_PredictResult(t) for t in texts]

    @staticmethod
    def _flatten_prepared_inputs(
        prepared_inputs: list[tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]],
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[tuple[int, int]]]:
        all_images: list[Image.Image | bytes] = []
        all_prompts: list[str] = []
        all_params: list[SamplingParams | None] = []
        all_indices: list[tuple[int, int]] = []
        for img_idx, (block_images, prompts, params, indices) in enumerate(prepared_inputs):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])
        return all_images, all_prompts, all_params, all_indices

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        scored: bool | None = None,
    ) -> ExtractResult:
        layout_image = self.helper.prepare_for_layout(image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        output = self._predict(layout_image, prompt, params, priority, scored)
        blocks = self.helper.parse_layout_output(output.text)
        return ExtractResult(blocks, output.scored)

    def batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        scored: bool | None = None,
    ) -> list[ExtractResult]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        layout_images = self.helper.batch_prepare_for_layout(self.executor, images)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        outputs = self._batch_predict(layout_images, prompt, params, priority, scored)
        blocks_list = self.helper.batch_parse_layout_output(self.executor, [output.text for output in outputs])
        return [ExtractResult(blocks, output.scored) for blocks, output in zip(blocks_list, outputs)]

    async def aio_layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        scored: bool | None = None,
    ) -> ExtractResult:
        layout_image = await self.helper.aio_prepare_for_layout(self.executor, image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        output = await self._aio_predict(layout_image, prompt, params, priority, semaphore, scored)
        blocks = await self.helper.aio_parse_layout_output(self.executor, output.text)
        return ExtractResult(blocks, output.scored)

    async def aio_batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        scored: bool | None = None,
    ) -> list[ExtractResult]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        layout_images = await gather_tasks(
            tasks=[self.helper.aio_prepare_for_layout(self.executor, im) for im in images],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Layout Preparation",
        )
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        outputs = await self._aio_batch_predict(
            layout_images,
            prompt,
            params,
            priority,
            semaphore,
            scored,
            use_tqdm=self.use_tqdm,
            tqdm_desc="Layout Detection",
        )
        blocks_list = await gather_tasks(
            tasks=[self.helper.aio_parse_layout_output(self.executor, output.text) for output in outputs],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Layout Output Parsing",
        )
        return [ExtractResult(blocks, output.scored) for blocks, output in zip(blocks_list, outputs)]

    def content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        scored: bool | None = None,
    ) -> ExtractStr | None:
        blocks = [ContentBlock(type, [0.0, 0.0, 1.0, 1.0])]
        block_images, prompts, params, _ = self.helper.prepare_for_extract(image, blocks)
        if not (block_images and prompts and params):
            return None
        output = self._predict(block_images[0], prompts[0], params[0], priority, scored)
        blocks[0].content = output.text
        blocks = self.helper.post_process(blocks)
        content = blocks[0].content if blocks else None
        return ExtractStr(content, scored=output.scored) if content is not None else None

    def batch_content_extract(
        self,
        images: list[Image.Image],
        types: Sequence[str] | str = "text",
        priority: Sequence[int | None] | int | None = None,
        scored: bool | None = None,
    ) -> list[ExtractStr | None]:
        if isinstance(types, str):
            types = [types] * len(images)
        if len(types) != len(images):
            raise Exception("Length of types must match length of images")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        blocks_list = [[ContentBlock(type, [0.0, 0.0, 1.0, 1.0])] for type in types]
        prepared_inputs = self.helper.batch_prepare_for_extract(self.executor, images, blocks_list)
        all_images, all_prompts, all_params, all_indices = self._flatten_prepared_inputs(prepared_inputs)
        outputs = self._batch_predict(all_images, all_prompts, all_params, priority, scored)
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = output.text
            blocks_list[img_idx][idx].scored = output.scored
        blocks_list = self.helper.batch_post_process(self.executor, blocks_list)
        return [
            ExtractStr(blocks[0].content, scored=blocks[0].scored) if blocks and blocks[0].content is not None else None
            for blocks in blocks_list
        ]

    async def aio_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        scored: bool | None = None,
    ) -> ExtractStr | None:
        blocks = [ContentBlock(type, [0.0, 0.0, 1.0, 1.0])]
        block_images, prompts, params, _ = await self.helper.aio_prepare_for_extract(self.executor, image, blocks)
        if not (block_images and prompts and params):
            return None
        output = await self._aio_predict(block_images[0], prompts[0], params[0], priority, semaphore, scored)
        blocks[0].content = output.text
        blocks = await self.helper.aio_post_process(self.executor, blocks)
        content = blocks[0].content if blocks else None
        return ExtractStr(content, scored=output.scored) if content is not None else None

    async def aio_batch_content_extract(
        self,
        images: list[Image.Image],
        types: Sequence[str] | str = "text",
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        scored: bool | None = None,
    ) -> list[ExtractStr | None]:
        if isinstance(types, str):
            types = [types] * len(images)
        if len(types) != len(images):
            raise Exception("Length of types must match length of images")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        blocks_list = [[ContentBlock(type, [0.0, 0.0, 1.0, 1.0])] for type in types]
        prepared_inputs = await gather_tasks(
            tasks=[self.helper.aio_prepare_for_extract(self.executor, *args) for args in zip(images, blocks_list)],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extract Preparation",
        )
        all_images, all_prompts, all_params, all_indices = self._flatten_prepared_inputs(prepared_inputs)
        outputs = await self._aio_batch_predict(
            all_images,
            all_prompts,
            all_params,
            priority,
            semaphore,
            scored,
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extraction",
        )
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = output.text
            blocks_list[img_idx][idx].scored = output.scored
        blocks_list = await gather_tasks(
            tasks=[self.helper.aio_post_process(self.executor, blocks) for blocks in blocks_list],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Post Processing",
        )
        return [
            ExtractStr(blocks[0].content, scored=blocks[0].scored) if blocks and blocks[0].content is not None else None
            for blocks in blocks_list
        ]

    def two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        not_extract_list: list[str] | None = None,
        scored: bool | None = None,
    ) -> ExtractResult:
        layout_result = self.layout_detect(image, priority, scored)
        block_images, prompts, params, indices = self.helper.prepare_for_extract(image, layout_result, not_extract_list)
        outputs = self._batch_predict(block_images, prompts, params, priority, scored)
        for idx, output in zip(indices, outputs):
            layout_result[idx].content = output.text
            layout_result[idx].scored = output.scored
        return ExtractResult(self.helper.post_process(layout_result), layout_result.layout_scored)

    async def aio_two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        not_extract_list: list[str] | None = None,
        scored: bool | None = None,
    ) -> ExtractResult:
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        layout_result = await self.aio_layout_detect(image, priority, semaphore, scored)
        block_images, prompts, params, indices = await self.helper.aio_prepare_for_extract(
            self.executor,
            image,
            layout_result,
            not_extract_list,
        )
        outputs = await self._aio_batch_predict(block_images, prompts, params, priority, semaphore, scored)
        for idx, output in zip(indices, outputs):
            layout_result[idx].content = output.text
            layout_result[idx].scored = output.scored
        processed = await self.helper.aio_post_process(self.executor, layout_result)
        return ExtractResult(processed, layout_result.layout_scored)

    def concurrent_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        not_extract_list: list[str] | None = None,
        scored: bool | None = None,
    ) -> list[ExtractResult]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        task = self.aio_concurrent_two_step_extract(
            images,
            priority,
            not_extract_list,
            scored=scored,
        )

        if loop is not None:
            return loop.run_until_complete(task)
        else:
            return asyncio.run(task)

    async def aio_concurrent_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        not_extract_list: list[str] | None = None,
        semaphore: asyncio.Semaphore | None = None,
        scored: bool | None = None,
    ) -> list[ExtractResult]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        results = await gather_tasks(
            tasks=[self.aio_two_step_extract(*args, semaphore, not_extract_list, scored) for args in zip(images, priority)],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Two Step Extraction",
        )

        if self.helper.enable_cross_page_table_merge:
            from .post_process.cross_page_table import aio_detect_cross_page_cell_merge

            params = self.sampling_params.get("[cross_page_table_merge]")

            async def aio_batch_predict_fn(prompts: list[str]) -> list[str]:
                return await self.client.aio_batch_predict(
                    [None] * len(prompts), prompts, [params] * len(prompts),
                )

            await aio_detect_cross_page_cell_merge(results, aio_batch_predict_fn)

        return results

    def stepping_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        not_extract_list: list[str] | None = None,
        scored: bool | None = None,
    ) -> list[ExtractResult]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        layout_results = self.batch_layout_detect(images, priority, scored)
        prepared_inputs = self.helper.batch_prepare_for_extract(self.executor, images, layout_results, not_extract_list)
        all_images, all_prompts, all_params, all_indices = self._flatten_prepared_inputs(prepared_inputs)
        outputs = self._batch_predict(all_images, all_prompts, all_params, priority, scored)
        for (img_idx, idx), output in zip(all_indices, outputs):
            layout_results[img_idx][idx].content = output.text
            layout_results[img_idx][idx].scored = output.scored
        processed_list = self.helper.batch_post_process(self.executor, layout_results)
        results = [ExtractResult(blocks, layout.layout_scored) for layout, blocks in zip(layout_results, processed_list)]

        if self.helper.enable_cross_page_table_merge:
            from .post_process.cross_page_table import detect_cross_page_cell_merge

            params = self.sampling_params.get("[cross_page_table_merge]")

            def batch_predict_fn(prompts: list[str]) -> list[str]:
                return self.client.batch_predict(
                    [None] * len(prompts), prompts, [params] * len(prompts),
                )

            detect_cross_page_cell_merge(results, batch_predict_fn)

        return results

    async def aio_stepping_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        not_extract_list: list[str] | None = None,
        semaphore: asyncio.Semaphore | None = None,
        scored: bool | None = None,
    ) -> list[ExtractResult]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        layout_results = await self.aio_batch_layout_detect(images, priority, semaphore, scored)
        prepared_inputs = await gather_tasks(
            tasks=[
                self.helper.aio_prepare_for_extract(self.executor, *args, not_extract_list)
                for args in zip(images, layout_results)
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extract Preparation",
        )
        all_images, all_prompts, all_params, all_indices = self._flatten_prepared_inputs(prepared_inputs)
        outputs = await self._aio_batch_predict(
            all_images,
            all_prompts,
            all_params,
            priority,
            semaphore,
            scored,
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extraction",
        )
        for (img_idx, idx), output in zip(all_indices, outputs):
            layout_results[img_idx][idx].content = output.text
            layout_results[img_idx][idx].scored = output.scored
        processed_list = await gather_tasks(
            tasks=[self.helper.aio_post_process(self.executor, lr) for lr in layout_results],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Post Processing",
        )
        results = [ExtractResult(blocks, layout.layout_scored) for layout, blocks in zip(layout_results, processed_list)]

        if self.helper.enable_cross_page_table_merge:
            from .post_process.cross_page_table import aio_detect_cross_page_cell_merge

            params = self.sampling_params.get("[cross_page_table_merge]")

            async def aio_batch_predict_fn(prompts: list[str]) -> list[str]:
                return await self.client.aio_batch_predict(
                    [None] * len(prompts), prompts, [params] * len(prompts),
                )

            await aio_detect_cross_page_cell_merge(results, aio_batch_predict_fn)

        return results

    def batch_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        not_extract_list: list[str] | None = None,
        scored: bool | None = None,
    ) -> list[ExtractResult]:
        if self.batching_mode == "concurrent":
            return self.concurrent_two_step_extract(images, priority, not_extract_list, scored)
        else:  # self.batching_mode == "stepping"
            return self.stepping_two_step_extract(images, priority, not_extract_list, scored)

    async def aio_batch_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        not_extract_list: list[str] | None = None,
        semaphore: asyncio.Semaphore | None = None,
        scored: bool | None = None,
    ) -> list[ExtractResult]:
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        if self.batching_mode == "concurrent":
            return await self.aio_concurrent_two_step_extract(images, priority, not_extract_list, semaphore, scored)
        else:  # self.batching_mode == "stepping"
            return await self.aio_stepping_two_step_extract(images, priority, not_extract_list, semaphore, scored)
