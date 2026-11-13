import asyncio
import os
import re
from base64 import b64decode, b64encode
from collections.abc import Coroutine
from io import BytesIO
from typing import Any, Callable, ParamSpec, Sequence, TypeVar

import aiofiles
import httpx
from PIL import Image
from tqdm import tqdm

from .base_client import EngineUnavailableError, ImageType, RequestError, SingleImageType

T = TypeVar("T")
P = ParamSpec("P")
VLM_PREDICT_DESC = "VLM Predict"
# 内部清理取消携带独立标记，不能与调用方主动取消混为一谈。
_BATCH_FAILURE_CANCEL = object()


def _is_batch_failure_cancel(error: BaseException) -> bool:
    """只识别批次收集器发出的取消，并允许标记穿过嵌套批次。"""
    return isinstance(error, asyncio.CancelledError) and any(arg is _BATCH_FAILURE_CANCEL for arg in error.args)


def vlm_predict_progress(*args: Any, **kwargs: Any) -> tqdm:
    """适配引擎的进度工厂，覆盖默认文案并保留计数、速度及显示选项。"""
    kwargs["desc"] = VLM_PREDICT_DESC
    return tqdm(*args, **kwargs)


async def run_in_thread_until_complete(function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """取消时等待同步线程结束，避免调用方释放仍被推理线程使用的模型租约。"""
    work = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        await asyncio.wait([work])
        return work.result()
    except asyncio.CancelledError as cancellation:
        while not work.done():
            try:
                await asyncio.wait([work])
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not work.cancelled():
            error = work.exception()
            if error is not None and _is_batch_failure_cancel(cancellation) and EngineUnavailableError.find_in(error):
                # 同批失败引起的取消仍要上报真实死亡；调用方主动取消保留原有语义。
                raise error
        raise


_timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
_file_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf")
_data_uri_regex = re.compile(r"^data:[^;,]+;base64,")


def load_resource(uri: str) -> bytes:
    if uri.startswith("http://") or uri.startswith("https://"):
        response = httpx.get(uri, timeout=_timeout)
        return response.content
    if uri.startswith("file://"):
        with open(uri[len("file://") :], "rb") as file:
            return file.read()
    if uri.lower().endswith(_file_exts):
        with open(uri, "rb") as file:
            return file.read()
    if re.match(_data_uri_regex, uri):
        return b64decode(uri.split(",")[1])
    return b64decode(uri)


async def aio_load_resource(uri: str) -> bytes:
    if uri.startswith("http://") or uri.startswith("https://"):
        async with httpx.AsyncClient(timeout=_timeout) as client:
            response = await client.get(uri)
            return response.content
    if uri.startswith("file://"):
        async with aiofiles.open(uri[len("file://") :], "rb") as file:
            return await file.read()
    if uri.lower().endswith(_file_exts):
        async with aiofiles.open(uri, "rb") as file:
            return await file.read()
    if re.match(_data_uri_regex, uri):
        return b64decode(uri.split(",")[1])
    return b64decode(uri)


def get_png_bytes(image: Image.Image) -> bytes:
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def get_image_format(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if image_bytes.startswith(b"\x89PNG"):
        return "png"
    if image_bytes.startswith(b"GIF8"):
        return "gif"
    if image_bytes.startswith(b"BM"):
        return "bmp"
    if image_bytes[0:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    if image_bytes.startswith(b"II\x2a\x00") or image_bytes.startswith(b"MM\x00\x2a"):
        return "tiff"
    raise RequestError("Unsupported image format.")


def get_image_data_url(image_bytes: bytes, image_format: str | None) -> str:
    image_base64 = b64encode(image_bytes).decode("utf-8")
    if not image_format:
        image_format = get_image_format(image_bytes)
    return f"data:image/{image_format};base64,{image_base64}"


def get_rgb_image(image: Image.Image) -> Image.Image:
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def image_to_seq(image: ImageType) -> Sequence[SingleImageType]:
    if image is None:
        image = []
    elif isinstance(image, SingleImageType):
        image = [image]
    return image


def image_to_obj_list(image: ImageType) -> list[Image.Image]:
    image_objs: list[Image.Image] = []
    for image_item in image_to_seq(image):
        if isinstance(image_item, str):
            image_item = load_resource(image_item)
        if not isinstance(image_item, Image.Image):
            image_item = Image.open(BytesIO(image_item))
        image_item = get_rgb_image(image_item)
        image_objs.append(image_item)
    return image_objs


async def aio_image_to_obj_list(image: ImageType) -> list[Image.Image]:
    image_objs: list[Image.Image] = []
    for image_item in image_to_seq(image):
        if isinstance(image_item, str):
            image_item = await aio_load_resource(image_item)
        if not isinstance(image_item, Image.Image):
            image_item = Image.open(BytesIO(image_item))
        image_item = get_rgb_image(image_item)
        image_objs.append(image_item)
    return image_objs


def image_to_bytes_list_and_format(image: ImageType):
    image_bytes: list[bytes] = []
    image_format: str | None = "png"
    for image_item in image_to_seq(image):
        if isinstance(image_item, Image.Image):
            image_item = get_png_bytes(image_item)
        else:  # image is not PIL Image
            image_format = None
        if isinstance(image_item, str):
            image_item = load_resource(image_item)
        image_bytes.append(image_item)
    return image_bytes, image_format


async def aio_image_to_bytes_list_and_format(image: ImageType):
    image_bytes: list[bytes] = []
    image_format: str | None = "png"
    for image_item in image_to_seq(image):
        if isinstance(image_item, Image.Image):
            image_item = await run_in_thread_until_complete(get_png_bytes, image_item)
        else:  # image is not PIL Image
            image_format = None
        if isinstance(image_item, str):
            image_item = await aio_load_resource(image_item)
        image_bytes.append(image_item)
    return image_bytes, image_format


async def gather_tasks(
    tasks: list[Coroutine[Any, Any, T]],
    use_tqdm=False,
    tqdm_desc: str | None = None,
) -> list[T]:
    """按完成顺序更新进度并还原结果顺序；空任务不创建进度条。"""
    if not tasks:
        return []

    # 直接调度原协程，避免包装任务在首次执行前取消时遗留未 await 的内层协程。
    task_list = [asyncio.create_task(task) for task in tasks]
    task_set = set(task_list)
    pending = set(task_set)
    try:
        with tqdm(total=len(tasks), desc=tqdm_desc, disable=not use_tqdm) as pbar:
            while len(pending) > 0:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for done_task in done:
                    done_task.result()
                pbar.update(len(done))
    except (Exception, asyncio.CancelledError) as primary:
        # 子任务失败或父协程取消时，先取消未完成任务，再等待各任务完成自身清理。
        failure_cleanup = isinstance(primary, Exception) or _is_batch_failure_cancel(primary)
        for task in pending:
            if failure_cleanup:
                task.cancel(_BATCH_FAILURE_CANCEL)
            else:
                task.cancel()
        cleanup = asyncio.gather(*task_set, return_exceptions=True)
        while not cleanup.done():
            try:
                await asyncio.wait([cleanup])
            except asyncio.CancelledError:
                continue
        results = cleanup.result()
        if failure_cleanup and EngineUnavailableError.find_in(primary) is None:
            # 清理期间才到达的明确死亡优先于普通请求错误；取消仍保留调用方原意。
            for result in results:
                if isinstance(result, Exception) and EngineUnavailableError.find_in(result) is not None:
                    raise result
        raise

    return [task.result() for task in task_list]
