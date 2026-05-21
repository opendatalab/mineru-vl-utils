import argparse
import ast
import json
import re
import sys
from loguru import logger
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

SEP_CELL_RE = re.compile(r"^:?-{3,}:?$")
REPEAT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*|[^\s]", re.UNICODE)
REPEATED_CHAR_RUN_RE = re.compile(r"([A-Za-z0-9])\1{19,}")
FRAGMENT_SPLIT_RE = re.compile(r"\s*(?:::|;|,|\|)\s*")
FRAGMENT_REPEAT_MIN = 8
CELL_MAX_LEN = 300
CELL_MAX_DOUBLE_COLON = 20
CELL_MAX_SEMICOLON = 30

def content_without_table_separator_lines(content: str) -> str:
    return "\n".join(
        line for line in content.splitlines() if not is_markdown_table_separator_line(line)
    )

def is_separator_row(cells: List[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        compact = cell.replace(" ", "")
        if not SEP_CELL_RE.fullmatch(compact):
            return False
    return True

def split_markdown_row(line: str) -> Optional[List[str]]:
    text = line.strip()
    if not text or "|" not in text:
        return None
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]

    cells = []
    buf = []
    escaped = False
    in_code = False
    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            buf.append(ch)
            continue
        if ch == "`":
            in_code = not in_code
            buf.append(ch)
            continue
        if ch == "|" and not in_code:
            cells.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    cells.append("".join(buf).strip())
    return cells

def is_markdown_table_separator_line(line: str) -> bool:
    cells = split_markdown_row(line)
    return is_separator_row(cells)

def repeated_whole_content(content: str) -> Optional[Tuple[str, int]]:
    compact = re.sub(r"\s+", " ", content).strip()
    size = len(compact)
    if size < 80:
        return None
    period = (compact + compact).find(compact, 1)
    if 0 < period < size and size % period == 0:
        count = size // period
        if count >= 2:
            return compact[:period], count
    return None

def has_pipe_escape(s: str) -> bool:
    return r'\|' in s  # 用 raw string 避免自身触发警告


def normalize_fragment(fragment: str) -> str:
    return re.sub(r"\s+", " ", fragment).strip().lower()


def detect_cell_repetition_issue(cell_text: str) -> Optional[str]:
    double_colon_count = cell_text.count("::")
    semicolon_count = cell_text.count(";")
    if (
        len(cell_text) > CELL_MAX_LEN
        or double_colon_count > CELL_MAX_DOUBLE_COLON
        or semicolon_count > CELL_MAX_SEMICOLON
    ):
        return (
            "content_repetition_cell_overlength: "
            f"len={len(cell_text)}, double_colon={double_colon_count}, semicolon={semicolon_count}"
        )

    fragment_counter: Dict[str, int] = {}
    for fragment in FRAGMENT_SPLIT_RE.split(cell_text):
        normalized = normalize_fragment(fragment)
        if len(normalized) < 3:
            continue
        fragment_counter[normalized] = fragment_counter.get(normalized, 0) + 1
        if fragment_counter[normalized] >= FRAGMENT_REPEAT_MIN:
            return (
                "content_repetition_phrase_segment: "
                f"fragment {normalized!r} repeats {fragment_counter[normalized]} times"
            )
    return None


def detect_cell_repetition_pattern(content: str) -> Optional[str]:
    lines = content.splitlines()
    in_table_block = False
    separator_seen = False

    for line_no, line in enumerate(lines, 1):
        row = split_markdown_row(line)
        if row is None:
            in_table_block = False
            separator_seen = False
            continue

        if not in_table_block:
            in_table_block = True
            separator_seen = False
            continue

        if not separator_seen:
            if is_separator_row(row):
                separator_seen = True
            continue

        for cell_idx, cell in enumerate(row, 1):
            cell_text = cell.strip()
            if not cell_text:
                continue
            issue = detect_cell_repetition_issue(cell_text)
            if issue:
                return f"{issue} at line {line_no} cell {cell_idx}"
    return None


def is_noisy_table_row(cells: List[str]) -> bool:
    if is_separator_row(cells):
        return False
    for cell in cells:
        cell_text = cell.strip()
        if not cell_text:
            continue
        if detect_cell_repetition_issue(cell_text):
            return True
    return False

def validate_content_repetition(content: str) -> List[str]:
    errors = []

    if has_pipe_escape(content):
        errors.append(f"has_pipe_escape")
    
    for match in re.finditer(r"\n{5,}", content):
        line_no = content[: match.start()].count("\n") + 1
        errors.append(f"content_repetition_newline: {len(match.group(0))} consecutive newlines at line {line_no}")
        break

    whole_repeat = repeated_whole_content(content)
    if whole_repeat:
        unit, count = whole_repeat
        errors.append(
            f"content_repetition_whole_content: content repeats {count} times, repeated_unit_len={len(unit)}"
        )

    for match in REPEATED_CHAR_RUN_RE.finditer(content):
        line_no = content[: match.start()].count("\n") + 1
        repeated_char = match.group(1)
        errors.append(
            f"content_repetition_repeated_char: character {repeated_char!r} repeats {len(match.group(0))} times at line {line_no}"
        )
        break

    cell_repetition_error = detect_cell_repetition_pattern(content)
    if cell_repetition_error:
        errors.append(cell_repetition_error)

    lines = [line.strip() for line in content.strip("\n").splitlines()]
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue
        next_index = index + 1
        while next_index < len(lines) and lines[next_index] == line:
            next_index += 1
        count = next_index - index
        if (count >= 2 and len(line) >= 30) or count >= 3:
            errors.append(
                f"content_repetition_consecutive_line: line {index + 1} repeats {count} times"
            )
            break
        index = next_index

    line_positions: Dict[str, List[int]] = {}
    for line_no, line in enumerate(lines, 1):
        if len(line) >= 30 and not is_markdown_table_separator_line(line):
            line_positions.setdefault(line, []).append(line_no)
    for positions in line_positions.values():
        if len(positions) >= 4:
            shown = ",".join(str(pos) for pos in positions[:10])
            errors.append(
                f"content_repetition_same_line_many_times: same line appears {len(positions)} times at lines {shown}"
            )
            break

    token_text = content_without_table_separator_lines(content)
    tokens = REPEAT_TOKEN_RE.findall(token_text)
    index = 0
    while index < len(tokens):
        next_index = index + 1
        lower_token = tokens[index].lower()
        while next_index < len(tokens) and tokens[next_index].lower() == lower_token:
            next_index += 1
        count = next_index - index
        if count >= 8:
            errors.append(
                f"content_repetition_single_token: token {tokens[index]!r} repeats {count} times at token {index}"
            )
            break
        index = next_index

    return errors


def has_pipe_boundaries(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def normalize_table_source_line(line: str) -> str:
    """Normalize common model table syntax mistakes before parsing."""
    normalized = line.strip().replace("\uff5c", "|").replace("\u00a6", "|")
    return normalized.replace(r"\|", "|")


def split_repair_markdown_row(line: str) -> Optional[List[str]]:
    normalized = normalize_table_source_line(line)
    if "|" not in normalized:
        return None
    cells = split_markdown_row(normalized)
    if cells is None or len(cells) < 2:
        return None
    return cells


def clean_table_cell(cell: str) -> str:
    cleaned = re.sub(r"\s+", " ", cell.replace("\t", " ")).strip()
    if cleaned.count("`") % 2 == 1:
        cleaned += "`"
    return cleaned.replace("|", r"\|")


def format_markdown_row(cells: List[str]) -> str:
    return "| " + " | ".join(clean_table_cell(cell) for cell in cells) + " |"


def format_markdown_separator(column_count: int) -> str:
    return "| " + " | ".join("---" for _ in range(column_count)) + " |"


def parsed_table_blocks(content: str) -> List[List[List[str]]]:
    blocks: List[List[List[str]]] = []
    current: List[List[str]] = []
    separator_seen = False

    for line in content.splitlines():
        cells = split_repair_markdown_row(line)
        if cells is None:
            if current:
                blocks.append(current)
                current = []
            separator_seen = False
            continue

        if not current:
            current.append(cells)
            separator_seen = False
            continue

        if not separator_seen:
            current.append(cells)
            if is_separator_row(cells):
                separator_seen = True
            continue

        if is_noisy_table_row(cells):
            continue
        current.append(cells)

    if current:
        blocks.append(current)
    return blocks


def build_table_candidate(rows: List[List[str]], separator_index: int) -> Optional[str]:
    header_index = separator_index - 1
    if header_index < 0:
        return None

    header = rows[header_index]
    if len(header) < 2 or is_separator_row(header):
        return None

    expected_cols = len(header)
    data_rows: List[List[str]] = []
    for row in rows[separator_index + 1 :]:
        if is_separator_row(row):
            break
        if len(row) != expected_cols:
            break
        data_rows.append(row)

    if not data_rows:
        return None

    fixed_lines = [format_markdown_row(header), format_markdown_separator(expected_cols)]
    fixed_lines.extend(format_markdown_row(row) for row in data_rows)
    return "\n".join(fixed_lines)


def build_fallback_table_candidates(rows: List[List[str]]) -> Iterator[str]:
    index = 0
    while index < len(rows):
        if is_separator_row(rows[index]) or len(rows[index]) < 2:
            index += 1
            continue

        expected_cols = len(rows[index])
        run = [rows[index]]
        next_index = index + 1
        while next_index < len(rows):
            row = rows[next_index]
            if is_separator_row(row) or len(row) != expected_cols:
                break
            run.append(row)
            next_index += 1

        if len(run) >= 2:
            fixed_lines = [format_markdown_row(run[0]), format_markdown_separator(expected_cols)]
            fixed_lines.extend(format_markdown_row(row) for row in run[1:])
            yield "\n".join(fixed_lines)
        index = max(next_index, index + 1)


def table_score(table: str) -> Tuple[int, int, int]:
    rows = table.splitlines()
    first_row = split_markdown_row(rows[0]) or []
    return (len(rows), len(first_row), len(table))


def validate_repaired_markdown_table(table: str) -> bool:
    lines = table.strip().splitlines()
    if len(lines) < 3:
        return False
    header = split_markdown_row(lines[0])
    separator = split_markdown_row(lines[1])
    if header is None or separator is None or not is_separator_row(separator):
        return False
    if not has_pipe_boundaries(lines[0]) or not has_pipe_boundaries(lines[1]):
        return False
    if len(header) != len(separator):
        return False
    expected_cols = len(header)
    for line in lines[2:]:
        row = split_markdown_row(line)
        if row is None or not has_pipe_boundaries(line) or len(row) != expected_cols:
            return False
    return True


def extract_largest_legal_markdown_table(content: str) -> str:
    """Extract and normalize the largest syntactically valid markdown table."""
    candidates: List[str] = []
    for rows in parsed_table_blocks(content):
        for index, row in enumerate(rows):
            if not is_separator_row(row):
                continue
            candidate = build_table_candidate(rows, index)
            if candidate and validate_repaired_markdown_table(candidate):
                candidates.append(candidate)
        for candidate in build_fallback_table_candidates(rows):
            if validate_repaired_markdown_table(candidate):
                candidates.append(candidate)

    if not candidates:
        return ""
    return max(candidates, key=table_score)


def repair_content_if_repetition(content: str) -> Tuple[str, List[str]]:
    errors = validate_content_repetition(content)
    if not errors:
        return content, errors

    fixed_table = extract_largest_legal_markdown_table(content)
    return fixed_table or content, errors


def try_fix_chart_invalid_repetition(content: str, debug: bool = False) -> str:

    errors = validate_content_repetition(content)
    for error in errors:
        logger.debug(error)

    if errors:
        fixed_content = extract_largest_legal_markdown_table(content)
        if not fixed_content:
            fixed_content = content
    else:
        fixed_content = content
        
    if debug and fixed_content != content:
        logger.debug("Fixed chart markdown from: {} to: {}", fixed_content, content)
        
    return fixed_content
