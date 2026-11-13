"""普通批量预测的可选块级降级策略，不改变单次和评分接口。"""

from dataclasses import dataclass
from typing import Sequence

from loguru import logger

from .base_client import ClientClosedError, EngineUnavailableError, HttpResponseError, ServerError


@dataclass
class BlockPredictionResult:
    """分别记录文本和成功状态，避免把合法空文本误认为失败。"""

    text: str = ""
    error: Exception | None = None


def capture_block_error(error: Exception, *, enabled: bool, backend: str, index: int) -> BlockPredictionResult:
    """仅在显式开启容错且错误可局部降级时记录该块失败。"""
    fatal_http = isinstance(error, HttpResponseError) and (
        error.status_code in {401, 403, 404, 407, 429} or error.status_code >= 500
    )
    # 未分类的运行时故障可能表示引擎已停止；明确的服务响应异常另按状态判断。
    fatal_runtime = isinstance(error, RuntimeError) and not isinstance(error, ServerError)
    if (
        not enabled
        or fatal_http
        or fatal_runtime
        or EngineUnavailableError.find_in(error) is not None
        or isinstance(
            error, (ClientClosedError, MemoryError, AssertionError, AttributeError, TypeError, LookupError, NotImplementedError)
        )
    ):
        raise error
    logger.error("{} block {} prediction failed; result left empty: {}: {}", backend, index, type(error).__name__, error)
    return BlockPredictionResult(error=error)


def collect_block_results(results: Sequence[BlockPredictionResult]) -> list[str]:
    """保持输入顺序；非空批次全部失败时重抛第一项原异常。"""
    if results and all(result.error is not None for result in results):
        raise results[0].error
    return [result.text for result in results]
