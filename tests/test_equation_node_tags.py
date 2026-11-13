"""Cover node/tag alignment and delimiter repair after non-array separators."""

import pytest

from mineru_vl_utils.post_process import post_process
from mineru_vl_utils.post_process.equation_left_right import (
    clean_span,
    count_left,
    count_right,
    split_with_delimiters,
    tag_array,
    tag_element,
    try_match_equation_left_right,
)
from mineru_vl_utils.structs import ContentBlock


# The no-array path must also preserve the node/tag cardinality contract.
@pytest.mark.parametrize("node_count", [0, 1, 2, 7, 25])
def test_no_array_returns_one_none_tag_per_node(node_count):
    nodes = [f"node{index}" for index in range(node_count)]
    original = list(nodes)
    assert tag_element(nodes, []) == [None] * node_count
    assert nodes == original


def test_empty_nodes_can_pass_through_clean_span():
    assert clean_span([], tag_element([], [])) == []


# Inspect later nodes without changing the existing delimiter insertion positions.
@pytest.mark.parametrize("separator", ["&", r"\\"])
@pytest.mark.parametrize(
    "source, expected",
    [(r"\left(x", r"\left( \right.x"), (r"x\right)", r"x\left. \right)")],
)
def test_clean_span_repairs_nodes_after_separator(separator, source, expected):
    nodes = ["a=b", separator, source]
    assert clean_span(nodes, tag_element(nodes, [])) == ["a=b", separator, expected]
    assert nodes == ["a=b", separator, source]


LATER_NODE_REPAIRS = [
    (
        r"\begin{aligned}a &= b \\ c &= \left(d+e\end{aligned}",
        r"\begin{aligned}a&= b\\c&= \left( \right.d+e\end{aligned}",
    ),
    (
        r"\begin{aligned}a &= b \\ c &= d+e\right)\end{aligned}",
        r"\begin{aligned}a&= b\\c&= d+e\left. \right)\end{aligned}",
    ),
    (
        r"\begin{aligned}a &= \left(b \\ c &= d\end{aligned}",
        r"\begin{aligned}a&= \left( \right.b\\c&= d\end{aligned}",
    ),
    (
        r"\begin{gathered}a=b \\ c=\left[d+e\end{gathered}",
        r"\begin{gathered}a=b\\c=\left[ \right.d+e\end{gathered}",
    ),
    (
        r"\begin{aligned}a &= \left(b\right) \\ c &= \left[d\end{aligned}",
        r"\begin{aligned}a&= \left(b\right)\\c&= \left[ \right.d\end{aligned}",
    ),
    (
        r"\begin{aligned}a &= \left(b \\ c &= \left[d\end{aligned}",
        r"\begin{aligned}a&= \left( \right.b\\c&= \left[ \right.d\end{aligned}",
    ),
]


# Exercise the public postprocessor without mocking tokenization or repair.
@pytest.mark.parametrize("source, expected", LATER_NODE_REPAIRS)
def test_later_nodes_are_repaired_through_full_postprocess(source, expected):
    assert try_match_equation_left_right(source) == expected
    assert count_left(expected) == count_right(expected)
    block = ContentBlock("equation", [0.1, 0.1, 0.9, 0.3], content=source)
    result = post_process(
        [block],
        simple_post_process=False,
        handle_equation_block=False,
        abandon_list=False,
        abandon_paratext=False,
    )
    assert len(result) == 1
    assert result[0].content == "\\[\n" + expected + "\n\\]"


# Preserve balanced formulas exactly, not just their delimiter counts.
@pytest.mark.parametrize(
    "source",
    [
        "",
        "x+1",
        r"\left(x+1\right)",
        r"\begin{aligned}a &= b \\ c &= \left(d+e\right)\end{aligned}",
        r"\begin{aligned}a &= \left(b\right) \\ c &= \left[d\right]\end{aligned}",
        r"\begin{gathered}a=b \\ c=\left[d+e\right]\end{gathered}",
        r"\begin{cases}x&x>0\\-x&x<0\end{cases}",
        r"\left(\begin{array}{cc}a&b\\c&d\end{array}\right)",
    ],
)
def test_balanced_formulas_remain_unchanged(source):
    assert try_match_equation_left_right(source) == source


# Keep the existing single-node, array, and nested-array repair behavior.
@pytest.mark.parametrize(
    "source, expected",
    [
        (r"\left(x+1", r"\left( \right.x+1"),
        (r"x+1\right)", r"x+1\left. \right)"),
        (r"\left[x+1", r"\left[ \right.x+1"),
        (r"x+1\right]", r"x+1\left. \right]"),
        (r"\begin{array}{c}a\\\left(b\end{array}", r"\begin{array}{c}a\\\left( \right.b\end{array}"),
        (r"\begin{array}{c}a\\b\right)\end{array}", r"\begin{array}{c}a\\b\left. \right)\end{array}"),
        (r"\left(\begin{array}{c}a\\b\end{array}", r"\left( \right.\begin{array}{c}a\\b\end{array}"),
        (
            r"\left(\begin{array}{cc}a&\begin{array}{c}\left[b\\c\end{array}\\d&e\end{array}",
            r"\left( \right.\begin{array}{cc}a&\begin{array}{c}\left[ \right.b\\c\end{array}\\d&e\end{array}",
        ),
    ],
)
def test_existing_single_node_and_array_repairs(source, expected):
    assert try_match_equation_left_right(source) == expected


@pytest.mark.parametrize(
    "nodes, arrays, expected",
    [
        ([r"\begin{array}{c}", "a", r"\end{array}"], [[1, 0, 2, []]], [(1, 0), (1, 0), (1, 0)]),
        (
            [r"\left(", r"\begin{array}{cc}", "a", "&", "b", r"\end{array}", r"\right)"],
            [[1, 1, 5, []]],
            [None, (1, 0), (1, 0), None, (1, 1), (1, 1), None],
        ),
    ],
)
def test_existing_array_tags_preserve_indices(nodes, arrays, expected):
    assert tag_element(nodes, arrays) == expected


def test_nested_array_tags_preserve_length_and_inner_ownership():
    nodes = [r"\begin{array}{c}", "a", r"\\", r"\begin{array}{c}", "b", r"\end{array}", r"\end{array}"]
    arrays = sorted(tag_array(nodes), key=lambda array: len(array[-1]), reverse=True)
    assert tag_element(nodes, arrays) == [(1, 0), (1, 0), None, (2, 0), (2, 0), (2, 0), (1, 1)]


@pytest.mark.parametrize(
    "source",
    [r"\begin{array}{c}a\\\left(b", r"\left(b\end{array}", r"\end{array}\left(b\begin{array}{c}"],
)
def test_malformed_array_is_still_preserved(source):
    assert try_match_equation_left_right(source) == source


def test_simple_mode_does_not_start_repairing_formulas():
    source = LATER_NODE_REPAIRS[0][0]
    block = ContentBlock("equation", [0.1, 0.1, 0.9, 0.3], content=source)
    result = post_process(
        [block],
        simple_post_process=True,
        handle_equation_block=False,
        abandon_list=False,
        abandon_paratext=False,
    )
    assert result[0].content == source


def test_aligned_fixture_has_no_array_but_multiple_nodes():
    nodes = [node.strip() for node in split_with_delimiters(LATER_NODE_REPAIRS[0][0]) if node]
    assert len(nodes) == 7
    assert tag_array(nodes) == []
    assert tag_element(nodes, []) == [None] * 7
