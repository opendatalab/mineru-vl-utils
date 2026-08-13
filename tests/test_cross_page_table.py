import importlib
import sys
import types

import pytest

from mineru_vl_utils.post_process.cross_page_table import (
    _prepare_merge_tasks,
    detect_cross_page_cell_merge,
    find_cross_page_table_pairs,
)
from mineru_vl_utils.structs import ContentBlock, ExtractResult


PREVIOUS_HTML = (
    "<table>"
    "<tr><td>合计</td><td>--</td><td>49,031.73</td><td>50,199.76</td><td>7,973.94</td>"
    "<td>30,191.52</td><td>--</td><td>--</td><td>3,207.97</td><td>--</td><td>--</td></tr>"
    "<tr><td>分项目说明未达到计划进度、预计收益的情况和原因(含</td><td colspan=\"10\">不适用</td></tr>"
    "</table>"
)

CURRENT_HTML = (
    "<table>"
    "<tr><td>“是否达到预计效益”选择“不适用”的原因)</td><td></td></tr>"
    "<tr><td>项目可行性发生重大变化的情况说明</td><td>无</td></tr>"
    "</table>"
)


@pytest.fixture(autouse=True)
def _restore_cross_page_table_module():
    """每个用例结束后重载真实模块，避免 fake import 状态污染后续测试。"""
    yield
    import mineru_vl_utils.post_process.cross_page_table as cross_page_table

    importlib.reload(cross_page_table)


def _table_block(html: str) -> ContentBlock:
    """构造归一化 bbox 的最小 table block。"""
    return ContentBlock(type="table", bbox=[0.1, 0.1, 0.9, 0.9], angle=0, content=html)


def _results(previous_html: str = PREVIOUS_HTML, current_html: str = CURRENT_HTML) -> list[ExtractResult]:
    """构造两页跨页表格检测输入。"""
    return [ExtractResult([_table_block(previous_html)]), ExtractResult([_table_block(current_html)])]


def _fake_table_merge_module(name: str, marker: str) -> types.ModuleType:
    """构造带完整跨页表格辅助函数接口的假模块。"""
    module = types.ModuleType(name)
    module.build_table_state_from_html = lambda html: marker
    module.build_row_rendered_cell_segments = lambda rows, idx: []
    module.can_merge_by_structure = lambda *args, **kwargs: False
    module.calculate_row_rendered_segments = lambda rows, idx: 0
    module.detect_table_headers = lambda *args, **kwargs: (0, False, [])
    return module


def _reload_cross_page_table(monkeypatch, backend_module=None, legacy_module=None):
    """按测试指定模块状态重载跨页表格后处理模块。"""
    backend_path = "mineru.backend.utils.table_merge"
    legacy_path = "mineru.utils.table_merge"

    if backend_module is None:
        monkeypatch.setitem(sys.modules, backend_path, None)
    else:
        monkeypatch.setitem(sys.modules, backend_path, backend_module)

    if legacy_module is None:
        monkeypatch.setitem(sys.modules, legacy_path, None)
    else:
        monkeypatch.setitem(sys.modules, legacy_path, legacy_module)

    import mineru_vl_utils.post_process.cross_page_table as cross_page_table

    return importlib.reload(cross_page_table)


def test_table_merge_helpers_prefer_backend_module(monkeypatch):
    """backend 新路径和 legacy 旧路径同时存在时，应优先使用 backend。"""
    backend_module = _fake_table_merge_module("mineru.backend.utils.table_merge", "backend")
    legacy_module = _fake_table_merge_module("mineru.utils.table_merge", "legacy")

    cross_page_table = _reload_cross_page_table(monkeypatch, backend_module, legacy_module)

    assert cross_page_table._HAS_TABLE_MERGE is True
    assert cross_page_table.build_table_state_from_html("<table></table>") == "backend"


def test_table_merge_helpers_fallback_to_legacy_module(monkeypatch):
    """backend 新路径不可用时，应回退到 legacy 旧路径。"""
    legacy_module = _fake_table_merge_module("mineru.utils.table_merge", "legacy")

    cross_page_table = _reload_cross_page_table(monkeypatch, backend_module=None, legacy_module=legacy_module)

    assert cross_page_table._HAS_TABLE_MERGE is True
    assert cross_page_table.build_table_state_from_html("<table></table>") == "legacy"


def test_table_merge_unavailable_warning_describes_import_paths(monkeypatch):
    """两个路径都不可用时，warning 应说明缺失的是表格合并辅助接口。"""
    cross_page_table = _reload_cross_page_table(monkeypatch, backend_module=None, legacy_module=None)
    messages = []
    monkeypatch.setattr(cross_page_table.logger, "warning", lambda message, *args: messages.append(message.format(*args)))

    cross_page_table.detect_cross_page_cell_merge([], lambda prompts: [])

    assert cross_page_table._HAS_TABLE_MERGE is False
    assert messages
    assert "MinerU table merge helpers are unavailable" in messages[0]
    assert "mineru.backend.utils.table_merge" in messages[0]
    assert "mineru.utils.table_merge" in messages[0]
    assert "last import error" in messages[0]


def test_prepare_merge_task_uses_rendered_segments_for_colspan_boundary():
    """边界行视觉上都是 2 段时，不应因上一页 colspan 展开成 11 列而跳过。"""
    results = _results()
    pairs = find_cross_page_table_pairs(results)

    tasks = _prepare_merge_tasks(results, pairs)

    assert len(tasks) == 1
    assert tasks[0].expected_segment_count == 2
    assert tasks[0].expected_expanded_col_count == 11


def test_detect_cross_page_cell_merge_expands_segment_flags_to_visual_columns():
    """VLM 返回段级 cell_merge 后，应展开为 Magic-PDF 需要的视觉列级列表。"""
    results = _results()

    detect_cross_page_cell_merge(results, lambda prompts: ["[1, 0]"])

    assert results[1][0]["cell_merge"] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_detect_cross_page_cell_merge_skips_wrong_segment_count_response():
    """VLM 返回长度与渲染段数不一致时，应跳过写入 cell_merge。"""
    results = _results()

    detect_cross_page_cell_merge(results, lambda prompts: ["[1, 0, 1]"])

    assert "cell_merge" not in results[1][0]


def test_prepare_merge_task_skips_real_rendered_segment_mismatch():
    """边界行真实渲染段数不一致时，应继续跳过跨页单元格合并 prompt。"""
    previous_html = "<table><tr><td>A</td><td>B</td><td>C</td></tr></table>"
    current_html = "<table><tr><td>A</td><td>B</td></tr></table>"
    results = _results(previous_html, current_html)

    tasks = _prepare_merge_tasks(results, find_cross_page_table_pairs(results))

    assert tasks == []
