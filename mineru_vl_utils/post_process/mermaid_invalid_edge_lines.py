import re


_VALID_EDGE_LABEL_RE = re.compile(r"-->\s*\|[^|\n]+\|\s*\S")
_SUSPECT_SINGLE_PIPE_RE = re.compile(r"-->\s*[^|\n]*\|\s*[^|\n]+$")


def _count_unquoted_pipes(line: str) -> int:
    in_single = False
    in_double = False
    escape = False
    count = 0

    for ch in line:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "|" and not in_single and not in_double:
            count += 1
    return count


def _is_invalid_single_pipe_edge_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("%%"):
        return False
    if "-->" not in line or "|" not in line:
        return False

    # 正确写法: A -->|label| B
    if _VALID_EDGE_LABEL_RE.search(line):
        return False

    # 错误写法: G -->node(DRAM)| H（仅有单侧 |）
    return _count_unquoted_pipes(line) == 1 and _SUSPECT_SINGLE_PIPE_RE.search(stripped) is not None


def try_fix_mermaid_invalid_edge_lines(content: str, debug: bool = False) -> str:
    """Remove malformed Mermaid edge lines with unmatched single pipe.

    Example removed line:
        G -->node(DRAM)| H

    Keep valid edge labels untouched:
        A -->|label| B
    """
    kept_lines: list[str] = []
    for idx, line in enumerate(content.splitlines(), start=1):
        if _is_invalid_single_pipe_edge_line(line):
            if debug:
                print(f"[mermaid_invalid_edge] drop L{idx}: {line}")
            continue
        kept_lines.append(line)

    result = "\n".join(kept_lines)
    if content.endswith("\n"):
        result += "\n"
    return result


if __name__ == "__main__":
    mermaid = r'''graph TD
  A -->|ok| B
  G -->node(DRAM)| H
  C --> D
  E -->|v1.0| F
'''
    print(try_fix_mermaid_invalid_edge_lines(mermaid, debug=True))
