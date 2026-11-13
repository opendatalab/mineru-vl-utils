"""遍历异常链和异常组，不把消息文本当作故障分类依据。"""

import builtins
import sys
from collections.abc import Iterator


def iter_exception_chain(error: BaseException) -> Iterator[BaseException]:
    """保留显式 cause 和未隐藏的 context，并防止循环引用造成死循环。"""
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        yield current
        if sys.version_info >= (3, 11) and isinstance(current, builtins.BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None and not current.__suppress_context__:
            pending.append(current.__context__)


__all__ = ["iter_exception_chain"]
