"""验证定界符识别，同时保留既有缺失定界符的修复行为。"""

import re

import pytest

from mineru_vl_utils.post_process import post_process
from mineru_vl_utils.post_process.equation_left_right import (
    LEFT_TOKEN_LIST,
    RIGHT_TOKEN_LIST,
    VALID_LEFT_TOKEN_LIST,
    VALID_RIGHT_TOKEN_LIST,
    count_left,
    count_right,
    split_with_left_right,
    try_match_equation_left_right,
)
from mineru_vl_utils.structs import ContentBlock


# Prevent the matching patterns from drifting away from the pairing vocabulary.
@pytest.mark.parametrize(
    "actual, valid", [(LEFT_TOKEN_LIST, VALID_LEFT_TOKEN_LIST), (RIGHT_TOKEN_LIST, VALID_RIGHT_TOKEN_LIST)]
)
def test_patterns_are_escaped_valid_tokens(actual, valid):
    """识别正则应直接来自合法定界符列表，防止两者漂移。"""
    assert actual == [re.escape(token) for token in valid]


# Every delimiter already declared by the matcher must be recognized in full.
@pytest.mark.parametrize("token", VALID_LEFT_TOKEN_LIST)
def test_each_declared_left_delimiter_is_recognized(token):
    """每种已声明的左定界符都应被完整识别。"""
    assert re.fullmatch("|".join(LEFT_TOKEN_LIST), token)
    assert count_left(token) == 1
    assert count_right(token) == 0


@pytest.mark.parametrize("token", VALID_RIGHT_TOKEN_LIST)
def test_each_declared_right_delimiter_is_recognized(token):
    """每种已声明的右定界符都应被完整识别。"""
    assert re.fullmatch("|".join(RIGHT_TOKEN_LIST), token)
    assert count_right(token) == 1
    assert count_left(token) == 0


VALID_FORMULAS = [
    r"\left/ x \right/",
    r"\left/ \frac {a + b}{c + d} \right/",
    r"\left( a+b \right) \left/ \left( c+d \right) \right.",
    r"\left) x \right(",
    r"\left( x+1 \right)",
    r"\left[ x+1 \right]",
    r"\left] 0,1 \right[",
    r"\left| x \right|",
    r"\left\{ x \right\}",
    r"\left\lfloor x \right\rfloor",
    r"\left\langle x \right\rangle",
    r"\left. f(x) \right|_0^1",
    r"\left(\left[x\right]+1\right)",
    r"\left( a \rightarrow b \right)",
    r"\left( a \rightleftharpoons b \right)",
    r"\left(\begin{array}{cc} a&b\\c&d\end{array}\right)",
    r"\left/\begin{array}{cc} a&b\\c&d\end{array}\right/",
    r"\begin{array}{cc}\left(a\right)&\left[b\right]\\c&d\end{array}",
    r"a/b",
    r"I = \frac{x^3}{3}\big|_0^1",
    # The previous wildcard incidentally covered these unlisted or spaced delimiters.
    r"\left. x \right\rceil",
    r"\left( x \right )",
    r"\left ( x \right)",
    r"\left\lceil x \right\rceil",
    r"\left\lparen x \right\rparen",
    r"\left. x \right\backslash",
    r"\left. x \right\uparrow",
    r"\left. x \right\Downarrow",
]


@pytest.mark.parametrize("latex", VALID_FORMULAS)
def test_valid_formula_survives_matching_and_full_postprocess(latex):
    """合法公式经过匹配器和公开后处理入口后不得被误改。"""
    assert try_match_equation_left_right(latex) == latex
    block = ContentBlock("equation", [0.1, 0.1, 0.9, 0.3], content=latex)
    result = post_process(
        [block],
        simple_post_process=False,
        handle_equation_block=False,
        abandon_list=False,
        abandon_paratext=False,
    )
    assert result[0].content == "\\[\n" + latex + "\n\\]"


# A literal dot must not match unrelated commands sharing the right prefix.
@pytest.mark.parametrize(
    "command", [r"\rightarrow", r"\rightleftharpoons", r"\rightrightarrows", r"\rightsquigarrow", r"\rightX"]
)
def test_other_right_commands_are_not_delimiters(command):
    """同前缀的箭头等命令不应被当作右定界符。"""
    assert count_right(command) == 0
    assert split_with_left_right(command) == [command]
    assert try_match_equation_left_right(command) == command


def test_slash_tokenization_keeps_both_delimiters():
    """斜线定界符分词应同时保留左右两端。"""
    tokens = [token for token in split_with_left_right(r"\left/ x \right/") if token]
    assert tokens == [r"\left/", " x ", r"\right/"]


# Preserve exact insertion positions, not just balanced delimiter counts.
@pytest.mark.parametrize(
    "source, expected",
    [
        (r"\left( x", r"\left( \right. x"),
        (r"x \right)", r"x \left. \right)"),
        (r"\left[ x", r"\left[ \right. x"),
        (r"x \right]", r"x \left. \right]"),
        (r"\left. x", r"\left. \right. x"),
        (r"x \right.", r"x \left. \right."),
        (r"\left| x", r"\left| \right. x"),
        (r"x \right|", r"x \left. \right|"),
        (r"\left( x+\left[y\right]", r"\left( \right. x+\left[y\right]"),
    ],
)
def test_existing_missing_delimiter_repairs_are_preserved(source, expected):
    """保留既有修复的准确插入位置，而不只检查左右数量。"""
    assert try_match_equation_left_right(source) == expected
    assert count_left(expected) == count_right(expected)


@pytest.mark.parametrize("latex", [r"\left( x \end{array}", r"\left( x \begin{array}{c}"])
def test_incomplete_array_still_returns_original(latex):
    """数组环境不完整时继续保留原文，避免扩大改写范围。"""
    assert try_match_equation_left_right(latex) == latex


def test_simple_mode_keeps_its_existing_behavior():
    """简单后处理模式维持原有行为，不新增公式修复。"""
    source = r"\left/ x \right/"
    block = ContentBlock("equation", [0.1, 0.1, 0.9, 0.3], content=source)
    result = post_process(
        [block],
        simple_post_process=True,
        handle_equation_block=False,
        abandon_list=False,
        abandon_paratext=False,
    )
    assert result[0].content.removeprefix("\\[").removesuffix("\\]").strip() == source


# Arrow commands must not disable repairs of recognized delimiters.
@pytest.mark.parametrize(
    "source, expected",
    [
        (r"\left( a \rightarrow b", r"\left( \right. a \rightarrow b"),
        (r"a \rightarrow b \right)", r"a \rightarrow b \left. \right)"),
    ],
)
def test_missing_repairs_with_arrow_commands(source, expected):
    """箭头命令不应阻止对已识别定界符执行既有修复。"""
    assert try_match_equation_left_right(source) == expected
