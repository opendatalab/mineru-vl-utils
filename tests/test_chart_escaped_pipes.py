"""验证图表修复与校验保留单元格中的转义分隔符。"""

import pytest

from mineru_vl_utils.post_process.chart_invalid_repetition import (
    clean_table_cell,
    extract_largest_legal_markdown_table,
    normalize_table_source_line,
    repair_content_if_repetition,
    split_markdown_row,
    try_fix_chart_invalid_repetition,
    validate_content_repetition,
)
from mineru_vl_utils.post_process.image_analysis_postprocess import (
    _count_markdown_table_columns,
    convert_markdown_table_to_html,
    has_malformed_markdown_table,
    process_image_or_chart,
)


def tagged_chart(content):
    """按真实图表响应协议封装待校验内容。"""
    return (
        "<|class_start|>chart<|class_end|>"
        "<|sub_class_start|>bar<|sub_class_end|>"
        "<|caption_start|>Channel throughput<|caption_end|>"
        f"<|content_start|>{content}<|content_end|>"
    )


@pytest.mark.parametrize("outer_pipes", [False, True])
@pytest.mark.parametrize("label", [r"A\|B", r"A\\\|B", r"`A\|B`", r"end\|"])
def test_valid_escaped_cells_survive_the_complete_chart_handler(outer_pipes, label):
    """合法转义单元格经过完整图表处理链后应保持原文。"""
    rows = ["Value | Channel", "--- | ---", f"10 | {label}", "20 | other", "30 | E"]
    if outer_pipes:
        rows = [f"| {row} |" for row in rows]
    content = "\n".join(rows)

    assert validate_content_repetition(content) == []
    assert try_fix_chart_invalid_repetition(content) == content
    assert not has_malformed_markdown_table(content)
    assert convert_markdown_table_to_html(content) is not None
    result = process_image_or_chart(tagged_chart(content))
    assert result["content"] == content
    assert result["caption"] == "Channel throughput"


def test_header_and_all_data_rows_are_preserved():
    """表头和全部数据行不得因单元格中的竖线而丢失。"""
    content = r"""| Chan\|nel | Throughput (Mbit/s) |
| --- | --- |
| A \| B | 10 |
| C \| D | 20 |
| E | 30 |"""
    assert process_image_or_chart(tagged_chart(content))["content"] == content


@pytest.mark.parametrize("slash_count", range(1, 7))
def test_column_count_respects_backslash_parity(slash_count):
    """列数判断应区分奇数和偶数个反斜线形成的转义状态。"""
    row = "| A" + "\\" * slash_count + "|B | 10 |"
    expected = 2 if slash_count % 2 else 3
    assert _count_markdown_table_columns(row) == expected
    content = "| Name | Value |\n| --- | --- |\n" + row
    assert has_malformed_markdown_table(content) == (expected != 2)


@pytest.mark.parametrize("cell", [r"A\|B", r"A\\\|B", r"`A\|B`"])
def test_repair_does_not_double_escape_existing_cell_pipes(cell):
    """修复序列化不得对已转义的单元格竖线重复转义。"""
    assert clean_table_cell(cell) == cell
    content = f"| Name | Value |\n| --- | --- |\n| {cell} | 10 |\n| E | 30 |"
    assert extract_largest_legal_markdown_table(content) == content


def test_an_unrelated_repair_preserves_escaped_cell_content():
    """其他重复内容触发修复时仍应保留单元格原有转义。"""
    table = "| Name | Value |\n| --- | --- |\n| A\\|B | 10 |\n| E | 30 |"
    noisy = "intro\n\n\n\n\n\n" + table
    assert any("newline" in error for error in validate_content_repetition(noisy))
    assert try_fix_chart_invalid_repetition(noisy) == table


def test_plain_tables_and_actual_extra_columns_keep_their_behavior():
    """普通表格保持原样，确实多列的坏表仍按原策略拒绝。"""
    valid = "| Name | Value |\n| --- | --- |\n| A | 10 |"
    invalid = "| Name | Value |\n| --- | --- |\n| A | B | 10 |"
    assert process_image_or_chart(tagged_chart(valid))["content"] == valid
    assert has_malformed_markdown_table(invalid)
    assert process_image_or_chart(tagged_chart(invalid))["content"] == ""


def test_normalization_and_row_parsing_keep_a_trailing_escaped_pipe():
    """没有外侧分隔符时，行尾转义竖线仍属于最后一个单元格。"""
    row = r"10 | end\|"
    assert normalize_table_source_line(row) == row
    assert split_markdown_row(row) == ["10", r"end\|"]


def test_existing_fullwidth_delimiter_normalization_is_retained():
    """保留全角和相似竖线分隔符的既有归一化行为。"""
    assert normalize_table_source_line("｜ A ¦ 10 ｜") == "| A | 10 |"


@pytest.mark.parametrize("outer_pipes", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("cell", ["A", r"A\|B", r"A\\\|B", r"`A\|B`"])
def test_whole_table_overescaping_is_recovered_without_damaging_cell_escapes(outer_pipes, newline, cell):
    """整表仅移除一层额外转义，并保留单元格转义和换行格式。"""
    rows = ["Name | Value", ":--- | ---:", f"{cell} | 10", "E | 30"]
    if outer_pipes:
        rows = [f"| {row} |" for row in rows]
    expected = newline.join(rows)
    escaped = expected.replace("|", r"\|")

    assert try_fix_chart_invalid_repetition(escaped) == expected
    repaired, errors = repair_content_if_repetition(escaped)
    assert repaired == expected
    assert "has_pipe_escape" in errors
    assert process_image_or_chart(tagged_chart(escaped))["content"] == expected
    assert try_fix_chart_invalid_repetition(expected) == expected


def test_whole_table_recovery_preserves_surrounding_text_and_a_valid_neighbor():
    """整表转义恢复不得删除周边文本或重写相邻合法表格。"""
    valid = "| Channel | Value |\n| --- | --- |\n| A\\|B | 10 |"
    table = "| Name | Rate |\n| --- | --- |\n| X | 20 |"
    escaped = table.replace("|", r"\|")
    original = "Measured data:\n" + valid + "\n\nSecond table:\n" + escaped + "\nEnd.\n"
    expected = original.replace(escaped, table)
    assert try_fix_chart_invalid_repetition(original) == expected
    assert process_image_or_chart(tagged_chart(original))["content"] == expected.rstrip("\n")


@pytest.mark.parametrize(
    "content",
    [
        r"A\|B is a literal label.",
        "Name\\|Value\nX\\|10",  # No separator row: no evidence of a complete table.
        "Name\\|Value\n---\\|---",  # No data row.
        "Name\\|Value\n---\\|---\nX\\|10\\|extra",  # Conflicting column counts.
    ],
)
def test_ambiguous_escaped_text_is_not_reinterpreted_as_a_table(content):
    """缺少完整表格证据时保留转义文本，不猜测表格结构。"""
    assert try_fix_chart_invalid_repetition(content) == content
    assert repair_content_if_repetition(content) == (content, [])


def test_whole_table_recovery_still_runs_existing_repetition_repair():
    """整表转义恢复后仍执行既有的重复内容修复。"""
    table = "| Name | Value |\n| --- | --- |\n| X | 10 |"
    escaped = "intro\n\n\n\n\n\n" + table.replace("|", r"\|")
    restored, errors = repair_content_if_repetition(escaped)
    assert restored == table
    assert "has_pipe_escape" in errors
    assert any("newline" in error for error in errors)
