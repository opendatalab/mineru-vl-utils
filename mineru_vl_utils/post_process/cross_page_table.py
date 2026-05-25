import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from loguru import logger

from ..structs import ContentBlock, ExtractResult

try:
    from mineru.utils.table_merge import (
        build_table_state_from_html,
        build_row_rendered_cell_segments,
        can_merge_by_structure,
        calculate_row_rendered_segments,
        detect_table_headers,
    )

    _HAS_TABLE_MERGE = True
except ImportError:
    _HAS_TABLE_MERGE = False

SKIP_BETWEEN_TABLE_TYPES = {
    "table_caption",
    "table_footnote",
    "image_caption",
    "image_footnote",
    "header",
    "footer",
    "page_number",
    "page_footnote",
}


@dataclass(frozen=True)
class _BoundaryRowContext:
    header_count: int
    previous_last_row_metrics: Any
    current_first_data_row_metrics: Any
    previous_last_row_rendered_segments: int
    current_first_data_row_rendered_segments: int
    previous_last_row_segments: list[Any]
    current_first_data_row_segments: list[Any]

    @property
    def expanded_col_count(self) -> int:
        return max((segment.end_col for segment in self.previous_last_row_segments), default=0)

    @property
    def segment_count(self) -> int:
        return len(self.previous_last_row_segments)


@dataclass(frozen=True)
class _MergeTask:
    prompt: str
    prev_page_idx: int
    prev_block_idx: int
    curr_page_idx: int
    curr_block_idx: int
    expected_segment_count: int
    expected_expanded_col_count: int
    previous_segment_col_ranges: list[tuple[int, int]]


def _find_last_table_index(blocks: list[ContentBlock]) -> int | None:
    """从末尾扫描，跳过 caption/footnote/header/footer 等，找到最后一个 table block 的索引。"""
    for i in range(len(blocks) - 1, -1, -1):
        if blocks[i].type == "table":
            return i
        if blocks[i].type not in SKIP_BETWEEN_TABLE_TYPES:
            return None
    return None


def _find_first_table_index(blocks: list[ContentBlock]) -> int | None:
    """从开头扫描，跳过 caption/footnote/header/footer 等，找到第一个 table block 的索引。"""
    for i in range(len(blocks)):
        if blocks[i].type == "table":
            return i
        if blocks[i].type not in SKIP_BETWEEN_TABLE_TYPES:
            return None
    return None


def find_cross_page_table_pairs(
    results: Sequence[ExtractResult],
) -> list[tuple[int, int, int, int]]:
    """查找相邻页面中可能跨页的表格对。

    返回 [(prev_page_idx, prev_table_block_idx, curr_page_idx, curr_table_block_idx), ...]
    """
    pairs: list[tuple[int, int, int, int]] = []
    for page_idx in range(1, len(results)):
        prev_blocks = results[page_idx - 1]
        curr_blocks = results[page_idx]
        if not prev_blocks or not curr_blocks:
            continue

        prev_table_idx = _find_last_table_index(prev_blocks)
        if prev_table_idx is None:
            continue

        curr_table_idx = _find_first_table_index(curr_blocks)
        if curr_table_idx is None:
            continue

        prev_block = prev_blocks[prev_table_idx]
        curr_block = curr_blocks[curr_table_idx]
        if not prev_block.content or not curr_block.content:
            continue

        pairs.append((page_idx - 1, prev_table_idx, page_idx, curr_table_idx))

    return pairs


def _build_table_states(html1: str, html2: str) -> tuple[Any, Any] | None:
    """构建两个表格的 TableMergeState。"""
    if not _HAS_TABLE_MERGE:
        return None

    state1 = build_table_state_from_html(html1)
    state2 = build_table_state_from_html(html2)
    if state1 is None or state2 is None:
        return None

    return state1, state2


def can_tables_merge_by_structure(
    block1: ContentBlock,
    block2: ContentBlock,
) -> bool:
    """基于表格结构判断两个 ContentBlock 中的表格是否可合并。"""
    if not _HAS_TABLE_MERGE:
        logger.warning("mineru package not available, cannot check table merge structure")
        return False

    states = _build_table_states(block1.content, block2.content)
    if states is None:
        return False
    state1, state2 = states

    bbox1 = tuple(block1.bbox)
    bbox2 = tuple(block2.bbox)

    return can_merge_by_structure(state2, state1, current_bbox=bbox2, previous_bbox=bbox1)


def _build_boundary_row_context(html1: str, html2: str) -> _BoundaryRowContext | None:
    """构建跨页表格边界行的上下文信息。"""
    states = _build_table_states(html1, html2)
    if states is None:
        return None
    state1, state2 = states

    header_count, _, _ = detect_table_headers(state1, state2)
    previous_last_row_metrics = state1.last_data_row_metrics
    current_first_data_row_metrics = state2.front_first_data_row_metrics.get(header_count)
    if previous_last_row_metrics is None or current_first_data_row_metrics is None:
        return None

    previous_last_row_segments = build_row_rendered_cell_segments(state1.rows, previous_last_row_metrics.row_idx)
    current_first_data_row_segments = build_row_rendered_cell_segments(state2.rows, current_first_data_row_metrics.row_idx)
    if not previous_last_row_segments or not current_first_data_row_segments:
        return None

    previous_last_row_rendered_segments = calculate_row_rendered_segments(state1.rows, previous_last_row_metrics.row_idx)
    current_first_data_row_rendered_segments = calculate_row_rendered_segments(
        state2.rows, current_first_data_row_metrics.row_idx
    )

    return _BoundaryRowContext(
        header_count=header_count,
        previous_last_row_metrics=previous_last_row_metrics,
        current_first_data_row_metrics=current_first_data_row_metrics,
        previous_last_row_rendered_segments=previous_last_row_rendered_segments,
        current_first_data_row_rendered_segments=current_first_data_row_rendered_segments,
        previous_last_row_segments=previous_last_row_segments,
        current_first_data_row_segments=current_first_data_row_segments,
    )


def build_cell_merge_prompt(
    context: _BoundaryRowContext,
) -> str | None:
    """构建跨页表格单元格合并的 VLM prompt。

    Args:
        context: 跨页表格边界行上下文

    Returns:
        格式化的 prompt 字符串，或 None（无法提取有效数据时）
    """
    last_row_texts = [segment.text for segment in context.previous_last_row_segments]
    first_data_row_texts = [segment.text for segment in context.current_first_data_row_segments]

    # 按渲染单元格段无法对齐时，跳过 VLM 调用
    if len(last_row_texts) != len(first_data_row_texts):
        logger.debug(
            "Skipping cell merge prompt: rendered boundary segment count mismatch ({} vs {})",
            len(last_row_texts), len(first_data_row_texts),
        )
        return None

    last_row_repr = repr(last_row_texts)
    first_data_row_repr = repr(first_data_row_texts)

    prompt = rf"""Please merge the next two tables.

## Table 1 (Previous Page - Last Table)

**Caption:** (No caption)
**Last Row(s) Data:**
[{last_row_repr}]

---

## Table 2 (Current Page - First Table)

**Caption:** (No caption)
**First Data Row(s):**
[{first_data_row_repr}]"""

    logger.debug(f"\nPrompt cell merge prompt: {prompt}")
    return prompt


def parse_cell_merge_response(response: str) -> list[int] | None:
    """解析 VLM 返回的 cell_merge 列表。

    Returns:
        包含 0 和 1 的列表，或 None（解析失败时）
    """
    match = re.search(r"\[[\s\d,]+\]", response)
    if not match:
        return None

    try:
        result = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(result, list):
        return None
    if not all(isinstance(v, int) and v in (0, 1) for v in result):
        return None
    if not result:
        return None

    return result


def _prepare_merge_tasks(
    results: Sequence[ExtractResult],
    pairs: list[tuple[int, int, int, int]],
) -> list[_MergeTask]:
    """为可合并的跨页表格对准备 VLM prompts。

    Returns:
        [_MergeTask(...), ...]
    """
    tasks: list[_MergeTask] = []
    for prev_page_idx, prev_block_idx, curr_page_idx, curr_block_idx in pairs:
        prev_block = results[prev_page_idx][prev_block_idx]
        curr_block = results[curr_page_idx][curr_block_idx]

        if not can_tables_merge_by_structure(prev_block, curr_block):
            continue

        context = _build_boundary_row_context(prev_block.content, curr_block.content)
        if context is None:
            continue

        prev_rendered_segments = context.previous_last_row_rendered_segments
        curr_rendered_segments = context.current_first_data_row_rendered_segments
        if prev_rendered_segments != curr_rendered_segments:
            logger.debug(
                "Skipping cell merge prompt: boundary rendered segment mismatch ({} vs {})",
                prev_rendered_segments, curr_rendered_segments,
            )
            continue

        prompt = build_cell_merge_prompt(context)
        if prompt is None:
            continue

        tasks.append(
            _MergeTask(
                prompt=prompt,
                prev_page_idx=prev_page_idx,
                prev_block_idx=prev_block_idx,
                curr_page_idx=curr_page_idx,
                curr_block_idx=curr_block_idx,
                expected_segment_count=context.segment_count,
                expected_expanded_col_count=context.expanded_col_count,
                previous_segment_col_ranges=[
                    (segment.start_col, segment.end_col) for segment in context.previous_last_row_segments
                ],
            )
        )
    return tasks


def _apply_merge_results(
    results: Sequence[ExtractResult],
    tasks: list[_MergeTask],
    responses: list[str],
) -> None:
    """将 VLM batch 返回结果应用到对应的 block 上。"""
    if len(tasks) != len(responses):
        logger.warning(
            "Task/response count mismatch: {} tasks but {} responses, skipping merge results",
            len(tasks), len(responses),
        )
        return
    for task, response in zip(tasks, responses):
        cell_merge = parse_cell_merge_response(response)
        if cell_merge is None:
            continue

        if len(cell_merge) != task.expected_segment_count:
            logger.debug(
                "Skipping cross-page table merge result: rendered boundary segment count mismatch for "
                "page {} block {} -> page {} block {} ({} vs {})",
                task.prev_page_idx, task.prev_block_idx, task.curr_page_idx, task.curr_block_idx,
                len(cell_merge), task.expected_segment_count,
            )
            continue

        expanded_cell_merge = [0] * task.expected_expanded_col_count
        for merge_flag, (start_col, end_col) in zip(cell_merge, task.previous_segment_col_ranges):
            if merge_flag == 1:
                for col_idx in range(start_col, min(end_col, task.expected_expanded_col_count)):
                    expanded_cell_merge[col_idx] = 1

        logger.debug(
            "Cross-page table merge detected: page {} block {} -> page {} block {}, cell_merge={}",
            task.prev_page_idx, task.prev_block_idx, task.curr_page_idx, task.curr_block_idx, expanded_cell_merge,
        )
        results[task.curr_page_idx][task.curr_block_idx]["cell_merge"] = expanded_cell_merge


def detect_cross_page_cell_merge(
    results: Sequence[ExtractResult],
    batch_predict_fn: Callable[[list[str]], list[str]],
) -> None:
    """检测跨页表格并通过 VLM 批量判断单元格合并语义。

    对于可合并的跨页表格对，收集所有 prompts 后一次性调用 batch_predict_fn，
    并将 cell_merge 列表存储到当前页首表 block 上。

    Args:
        results: 各页的提取结果列表
        batch_predict_fn: 同步批量预测函数，接受 prompt 列表，返回模型输出列表
    """
    if not _HAS_TABLE_MERGE:
        logger.warning("mineru package not available, skipping cross-page table merge detection")
        return

    pairs = find_cross_page_table_pairs(results)
    if not pairs:
        return

    tasks = _prepare_merge_tasks(results, pairs)
    if not tasks:
        return

    prompts = [t.prompt for t in tasks]
    try:
        responses = batch_predict_fn(prompts)
    except Exception as e:
        logger.warning("VLM batch predict failed for cross-page table merge: {}", e)
        return

    _apply_merge_results(results, tasks, responses)


async def aio_detect_cross_page_cell_merge(
    results: Sequence[ExtractResult],
    aio_batch_predict_fn: Callable,
) -> None:
    """异步版本的跨页表格单元格合并检测。

    收集所有 prompts 后一次性调用 aio_batch_predict_fn 进行批量预测。

    Args:
        results: 各页的提取结果列表
        aio_batch_predict_fn: 异步批量预测函数，接受 prompt 列表，返回模型输出列表
    """
    if not _HAS_TABLE_MERGE:
        logger.warning("mineru package not available, skipping cross-page table merge detection")
        return

    pairs = find_cross_page_table_pairs(results)
    if not pairs:
        return

    tasks = _prepare_merge_tasks(results, pairs)
    if not tasks:
        return

    prompts = [t.prompt for t in tasks]
    try:
        responses = await aio_batch_predict_fn(prompts)
    except Exception as e:
        logger.warning("VLM batch predict failed for cross-page table merge: {}", e)
        return

    _apply_merge_results(results, tasks, responses)
