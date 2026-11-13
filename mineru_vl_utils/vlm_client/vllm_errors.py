"""将 vLLM 明确的实例死亡转换为公共异常，不加载模型或依赖消息匹配。"""

from collections.abc import Iterator
from contextlib import contextmanager

from .base_client import EngineUnavailableError
from .exception_chain import iter_exception_chain


@contextmanager
def vllm_error_scope() -> Iterator[None]:
    """仅在 vLLM 调用失败时惰性导入真实异常类，保留完整原始异常链。"""
    try:
        yield
    except Exception as error:
        if EngineUnavailableError.find_in(error) is not None:
            raise
        from vllm.v1.engine.exceptions import EngineDeadError

        fatal = next((item for item in iter_exception_chain(error) if isinstance(item, EngineDeadError)), None)
        if fatal is not None:
            raise EngineUnavailableError(str(fatal), error_message=str(fatal)) from error
        raise


__all__ = ["vllm_error_scope"]
