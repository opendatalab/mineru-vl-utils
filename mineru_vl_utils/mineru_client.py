import asyncio
import math
import os
import re
from concurrent.futures import Executor
from typing import Literal, Sequence

from PIL import Image
from loguru import logger

from .post_process import post_process
from .structs import BLOCK_TYPES, ContentBlock
from .table_image_processor import build_table_image_map, mask_and_crop_table_image
from .vlm_client import DEFAULT_SYSTEM_PROMPT, SamplingParams, new_vlm_client
from .vlm_client.utils import gather_tasks, get_png_bytes, get_rgb_image

_layout_re = r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*?)(?=<\|box_start\|>|$)"


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
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Detection:",
}

DEFAULT_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "table": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.005),
    "equation": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[default]": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]": MinerUSamplingParams(),
}

ANGLE_MAPPING: dict[str, Literal[0, 90, 180, 270]] = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}


def _convert_bbox(bbox: Sequence[int] | Sequence[str]) -> list[float] | None:
    bbox = tuple(map(int, bbox))
    if any(coord < 0 or coord > 1000 for coord in bbox):
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return list(map(lambda num: num / 1000.0, (x1, y1, x2, y2)))


def _parse_angle(tail: str) -> Literal[None, 0, 90, 180, 270]:
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            return angle
    return None


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
        debug: bool,
        enable_image_in_table_process: bool = True,
        table_image_overlap_threshold: float = 0.9,
        table_image_token_format: str = "[{idx}]",
        table_image_output_dir: str | None = None,
        table_image_rel_base: str | None = None,
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
        self.debug = debug
        self.enable_image_in_table_process = enable_image_in_table_process
        self.table_image_overlap_threshold = table_image_overlap_threshold
        self.table_image_token_format = table_image_token_format
        self.table_image_output_dir = table_image_output_dir
        self.table_image_rel_base = table_image_rel_base



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
        for match in re.finditer(_layout_re, output, re.DOTALL):
            x1, y1, x2, y2, ref_type, tail = match.groups()
            bbox = _convert_bbox((x1, y1, x2, y2))
            if bbox is None:
                print(f"Warning: invalid bbox in match: {match.group(0)}")
                continue  # Skip invalid bbox
            ref_type = ref_type.lower()
            angle = _parse_angle(tail)
            if angle is None:
                angle = 0
            blocks.append(ContentBlock(ref_type, bbox, angle=angle))
        
        # Post-process blocks to handle image inside table
        blocks = self._post_process_content_blocks(blocks)
        return blocks

    def _post_process_content_blocks(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        """
        Post-process content blocks to handle image inside table.
        For images inside tables, create image blocks with UIDs.
        """
        # Find all table blocks and image blocks
        table_indices = [i for i, b in enumerate(blocks) if b.type == "table"]
        
        # For each table, find images that are inside it
        for table_idx in table_indices:
            table_block = blocks[table_idx]
            tbox = table_block.bbox
            
            # Check all existing image blocks to see if they are inside this table
            for block in blocks:
                if block.type == "image":
                    # Calculate overlap ratio
                    overlap = self._calculate_overlap(block.bbox, tbox)
                    if overlap >= 0.9:  # Threshold for image being inside table
                        # Ensure image block has a UID
                        if not block.uid:
                            block.uid = self._generate_uid()
        
        return blocks

    def _calculate_overlap(self, inner: list[float], outer: list[float]) -> float:
        """Calculate the overlap ratio of inner box inside outer box."""
        x1 = max(inner[0], outer[0])
        y1 = max(inner[1], outer[1])
        x2 = min(inner[2], outer[2])
        y2 = min(inner[3], outer[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter_area = (x2 - x1) * (y2 - y1)
        inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
        
        return inter_area / inner_area if inner_area > 0 else 0.0

    def _generate_uid(self, length: int = 4) -> str:
        """Generate a short unique ID using uppercase letters and numbers."""
        import random
        letters = 'ACDGHKTWXYZ'
        numbers = '2345678'
        chars = letters + numbers
        return ''.join(random.choices(chars, k=length))

    def prepare_for_extract(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
        not_extract_list: list[str] | None = None,
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]:
        image = get_rgb_image(image)
        width, height = image.size
        block_images: list[Image.Image | bytes] = []
        prompts: list[str] = []
        sampling_params: list[SamplingParams | None] = []
        indices: list[int] = []
        skip_list = {"image", "list", "equation_block"}

        # get images in table after layout detection
        table_to_images = {}
        if self.enable_image_in_table_process:
             table_to_images = build_table_image_map(blocks, self.table_image_overlap_threshold)

        if not_extract_list:
            for not_extract_type in not_extract_list:
                if not_extract_type in BLOCK_TYPES:
                    skip_list.add(not_extract_type)
        for idx, block in enumerate(blocks):
            if block.type in skip_list:
                continue  # Skip blocks that should not be extracted.
            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled_bbox)
            if block_image.width < 1 or block_image.height < 1:
                print(f"Warning: cropped block image has invalid size {block_image.size}")
                continue

            # mask and crop table_image with image inside
            if block.type == "table" and self.enable_image_in_table_process:
                image_indices = table_to_images.get(idx, []) # (block_id,[])
                if image_indices:
                    image_blocks = [blocks[i] for i in image_indices]
                    block_image, token_map = mask_and_crop_table_image(
                        page_image=image,
                        table_block=block,
                        image_blocks=image_blocks,
                        table_image=block_image,
                        token_format=self.table_image_token_format,
                        output_root=self.table_image_output_dir,
                        rel_base=self.table_image_rel_base,
                        table_idx=idx,
                    )
                    block["table_image_token_map"] = token_map # mask_token->url
            # print(block["table_image_token_map"])
            if block.angle in [90, 180, 270]:
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
            blocks = post_process(
                blocks,
                simple_post_process=self.simple_post_process,
                handle_equation_block=self.handle_equation_block,
                abandon_list=self.abandon_list,
                abandon_paratext=self.abandon_paratext,
                debug=self.debug,
            )

            if self.enable_image_in_table_process:
                 for block in blocks:
                     if block.type == "table" and block.get("table_image_token_map") and block.content:
                         for token, url in block["table_image_token_map"].items():
                             # block.content = block.content.replace(token, f'<img src="{url}"/>')
                             # Use regex to allow spaces within brackets, e.g. [ AB23 ]
                             token_inner = token[1:-1] # Remove outer brackets
                             pattern = r'\[\s*' + re.escape(token_inner) + r'\s*\]'
                             block.content = re.sub(pattern, f'<img src="{url}"/>', block.content)

            return blocks
        except Exception as e:
            print(f"Warning: post-processing failed with error: {e}")
            return blocks

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
        blocks_list: list[list[ContentBlock]],
        not_extract_list: list[str] | None = None,
    ) -> list[tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]]:
        if executor is None:
            return [self.prepare_for_extract(im, bls, not_extract_list) for im, bls in zip(images, blocks_list)]
        return list(executor.map(self.prepare_for_extract, images, blocks_list, [not_extract_list] * len(images)))

    def batch_post_process(
        self,
        executor: Executor | None,
        blocks_list: list[list[ContentBlock]],
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
        enable_image_in_table_process: bool = True,
        table_image_overlap_threshold: float = 0.9,
        table_image_token_format: str = "[{idx}]",
        table_image_output_dir: str | None = None,
        table_image_rel_base: str | None = None,
    ) -> None:
        env_debug_value = os.getenv('MINERU_VL_DEBUG_ENABLE', '')
        if env_debug_value:
            if env_debug_value.lower() in ['true', '1', 'yes']:
                debug = True
            elif env_debug_value.lower() in ['false', '0', 'no']:
                debug = False
            else:
                logger.warning(f'unknown MINERU_VL_DEBUG_ENABLE config: {env_debug_value}, pass')

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

                try:
                    from mlx_vlm import load as mlx_load
                except ImportError:
                    raise ImportError("Please install mlx-vlm to use the mlx-engine backend.")
                model, processor = mlx_load(model_path)

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
            debug=debug,
            enable_image_in_table_process=enable_image_in_table_process,
            table_image_overlap_threshold=table_image_overlap_threshold,
            table_image_token_format=table_image_token_format,
            table_image_output_dir=table_image_output_dir,
            table_image_rel_base=table_image_rel_base,
        )
        self.backend = backend
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.incremental_priority = incremental_priority
        self.max_concurrency = max_concurrency
        self.executor = executor
        self.use_tqdm = use_tqdm
        self.debug = debug
        self.enable_image_in_table_process = enable_image_in_table_process
        self.table_image_overlap_threshold = table_image_overlap_threshold
        self.table_image_token_format = table_image_token_format
        self.table_image_output_dir = table_image_output_dir
        self.table_image_rel_base = table_image_rel_base

        if backend in ("http-client", "vllm-async-engine", "lmdeploy-engine"):
            self.batching_mode = "concurrent"
        else:  # backend in ("transformers", "vllm-engine")
            self.batching_mode = "stepping"

    def layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
    ) -> list[ContentBlock]:
        layout_image = self.helper.prepare_for_layout(image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        output = self.client.predict(layout_image, prompt, params, priority)
        return self.helper.parse_layout_output(output)

    def batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        layout_images = self.helper.batch_prepare_for_layout(self.executor, images)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        outputs = self.client.batch_predict(layout_images, prompt, params, priority)
        return self.helper.batch_parse_layout_output(self.executor, outputs)

    async def aio_layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        layout_image = await self.helper.aio_prepare_for_layout(self.executor, image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        if semaphore is None:
            output = await self.client.aio_predict(layout_image, prompt, params, priority)
        else:
            async with semaphore:
                output = await self.client.aio_predict(layout_image, prompt, params, priority)
        return await self.helper.aio_parse_layout_output(self.executor, output)

    async def aio_batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
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
        outputs = await self.client.aio_batch_predict(
            layout_images,
            prompt,
            params,
            priority,
            semaphore=semaphore,
            use_tqdm=self.use_tqdm,
            tqdm_desc="Layout Detection",
        )
        return await gather_tasks(
            tasks=[self.helper.aio_parse_layout_output(self.executor, out) for out in outputs],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Layout Output Parsing",
        )

    def content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
    ) -> str | None:
        blocks = [ContentBlock(type, [0.0, 0.0, 1.0, 1.0])]
        block_images, prompts, params, _ = self.helper.prepare_for_extract(image, blocks)
        if not (block_images and prompts and params):
            return None
        output = self.client.predict(block_images[0], prompts[0], params[0], priority)
        blocks[0].content = output
        blocks = self.helper.post_process(blocks)
        return blocks[0].content if blocks else None

    def batch_content_extract(
        self,
        images: list[Image.Image],
        types: Sequence[str] | str = "text",
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str | None]:
        if isinstance(types, str):
            types = [types] * len(images)
        if len(types) != len(images):
            raise Exception("Length of types must match length of images")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        blocks_list = [[ContentBlock(type, [0.0, 0.0, 1.0, 1.0])] for type in types]
        all_images: list[Image.Image | bytes] = []
        all_prompts: list[str] = []
        all_params: list[SamplingParams | None] = []
        all_indices: list[tuple[int, int]] = []
        prepared_inputs = self.helper.batch_prepare_for_extract(self.executor, images, blocks_list)
        for img_idx, (block_images, prompts, params, indices) in enumerate(prepared_inputs):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])
        outputs = self.client.batch_predict(all_images, all_prompts, all_params, priority)
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = output
        blocks_list = self.helper.batch_post_process(self.executor, blocks_list)
        return [blocks[0].content if blocks else None for blocks in blocks_list]

    async def aio_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> str | None:
        blocks = [ContentBlock(type, [0.0, 0.0, 1.0, 1.0])]
        block_images, prompts, params, _ = await self.helper.aio_prepare_for_extract(self.executor, image, blocks)
        if not (block_images and prompts and params):
            return None
        if semaphore is None:
            output = await self.client.aio_predict(block_images[0], prompts[0], params[0], priority)
        else:
            async with semaphore:
                output = await self.client.aio_predict(block_images[0], prompts[0], params[0], priority)
        blocks[0].content = output
        blocks = await self.helper.aio_post_process(self.executor, blocks)
        return blocks[0].content if blocks else None

    async def aio_batch_content_extract(
        self,
        images: list[Image.Image],
        types: Sequence[str] | str = "text",
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[str | None]:
        if isinstance(types, str):
            types = [types] * len(images)
        if len(types) != len(images):
            raise Exception("Length of types must match length of images")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        blocks_list = [[ContentBlock(type, [0.0, 0.0, 1.0, 1.0])] for type in types]
        all_images: list[Image.Image | bytes] = []
        all_prompts: list[str] = []
        all_params: list[SamplingParams | None] = []
        all_indices: list[tuple[int, int]] = []
        prepared_inputs = await gather_tasks(
            tasks=[self.helper.aio_prepare_for_extract(self.executor, *args) for args in zip(images, blocks_list)],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extract Preparation",
        )
        for img_idx, (block_images, prompts, params, indices) in enumerate(prepared_inputs):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])
        outputs = await self.client.aio_batch_predict(
            all_images,
            all_prompts,
            all_params,
            priority,
            semaphore=semaphore,
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extraction",
        )
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = output
        blocks_list = await gather_tasks(
            tasks=[self.helper.aio_post_process(self.executor, blocks) for blocks in blocks_list],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Post Processing",
        )
        return [blocks[0].content if blocks else None for blocks in blocks_list]

    def two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        not_extract_list: list[str] | None = None,
    ) -> list[ContentBlock]:
        blocks = self.layout_detect(image, priority)
        block_images, prompts, params, indices = self.helper.prepare_for_extract(image, blocks, not_extract_list)
        outputs = self.client.batch_predict(block_images, prompts, params, priority)
        for idx, output in zip(indices, outputs):
            blocks[idx].content = output
        return self.helper.post_process(blocks)

    async def aio_two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        not_extract_list: list[str] | None = None,
    ) -> list[ContentBlock]:
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        blocks = await self.aio_layout_detect(image, priority, semaphore)
        block_images, prompts, params, indices = await self.helper.aio_prepare_for_extract(
            self.executor,
            image,
            blocks,
            not_extract_list,
        )
        outputs = await self.client.aio_batch_predict(block_images, prompts, params, priority, semaphore=semaphore)
        for idx, output in zip(indices, outputs):
            blocks[idx].content = output
        return await self.helper.aio_post_process(self.executor, blocks)

    def concurrent_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        not_extract_list: list[str] | None = None,
    ) -> list[list[ContentBlock]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        task = self.aio_concurrent_two_step_extract(images, priority, not_extract_list)

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
    ) -> list[list[ContentBlock]]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        return await gather_tasks(
            tasks=[self.aio_two_step_extract(*args, semaphore, not_extract_list) for args in zip(images, priority)],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Two Step Extraction",
        )

    def stepping_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        not_extract_list: list[str] | None = None,
    ) -> list[list[ContentBlock]]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        blocks_list = self.batch_layout_detect(images, priority)
        all_images: list[Image.Image | bytes] = []
        all_prompts: list[str] = []
        all_params: list[SamplingParams | None] = []
        all_indices: list[tuple[int, int]] = []
        prepared_inputs = self.helper.batch_prepare_for_extract(
            self.executor,
            images,
            blocks_list,
            not_extract_list,
        )
        for img_idx, (block_images, prompts, params, indices) in enumerate(prepared_inputs):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])
        outputs = self.client.batch_predict(all_images, all_prompts, all_params, priority)
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = output
        return self.helper.batch_post_process(self.executor, blocks_list)

    async def aio_stepping_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        not_extract_list: list[str] | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        blocks_list = await self.aio_batch_layout_detect(images, priority, semaphore)
        all_images: list[Image.Image | bytes] = []
        all_prompts: list[str] = []
        all_params: list[SamplingParams | None] = []
        all_indices: list[tuple[int, int]] = []
        prepared_inputs = await gather_tasks(
            tasks=[
                self.helper.aio_prepare_for_extract(
                    self.executor,
                    *args,
                    not_extract_list,
                )
                for args in zip(images, blocks_list)
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extract Preparation",
        )
        for img_idx, (block_images, prompts, params, indices) in enumerate(prepared_inputs):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])
        outputs = await self.client.aio_batch_predict(
            all_images,
            all_prompts,
            all_params,
            priority,
            semaphore=semaphore,
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extraction",
        )
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = output
        return await gather_tasks(
            tasks=[self.helper.aio_post_process(self.executor, blocks) for blocks in blocks_list],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Post Processing",
        )

    def batch_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        not_extract_list: list[str] | None = None,
    ) -> list[list[ContentBlock]]:
        if self.batching_mode == "concurrent":
            return self.concurrent_two_step_extract(images, priority, not_extract_list)
        else:  # self.batching_mode == "stepping"
            return self.stepping_two_step_extract(images, priority, not_extract_list)

    async def aio_batch_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        not_extract_list: list[str] | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        if self.batching_mode == "concurrent":
            return await self.aio_concurrent_two_step_extract(images, priority, not_extract_list, semaphore)
        else:  # self.batching_mode == "stepping"
            return await self.aio_stepping_two_step_extract(images, priority, not_extract_list, semaphore)
