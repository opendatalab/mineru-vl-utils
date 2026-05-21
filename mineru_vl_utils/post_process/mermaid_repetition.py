import re
from dataclasses import dataclass


# Match node labels like: A["..."] / node_id["..."]
_NODE_LABEL_RE = re.compile(
    r'(?P<node>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*\["(?P<label>(?:[^"\\]|\\.)*)"\]',
    re.DOTALL,
)

# Match simple edge lines like: A --> B
_EDGE_LINE_RE = re.compile(
    r"^\s*(?P<src>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*-->"
    r"\s*(?:\|[^|\n]*\|\s*)?"
    r"(?P<dst>[A-Za-z0-9_\u4e00-\u9fa5\-]+)"
    r"(?:\s*(?:\[[^\]]*\]|\([^\)]*\)|\{[^\}]*\}))?\s*$"
)

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
# Edge line with label on destination node
_EDGE_LABELED_RE = re.compile(
    r"^\s*(?P<src>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*-->\s*"
    r"(?P<dst>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*"
    r"\[\"(?P<label>(?:[^\"\\]|\\\\.)*)\"\]\s*$"
)
# CamelCase-style 2-char tokens inside concatenated node ids (An, Ao, Ap, ...)
_NODE_ID_PAIR_RE = re.compile(r"[A-Z][a-z]|[A-Z]{2}")

DEFAULT_MAX_NODE_ID_LEN = 128



@dataclass
class RepetitionIssue:
    node_id: str
    reason: str
    repeat_count: int
    snippet: str


def _normalize_escaped_newlines(text: str) -> str:
    # Handle both "\n" and "\\n" styles.
    text = re.sub(r"\\+n", "\n", text)
    return _BR_RE.sub("\n", text)


def _max_consecutive_repeat(lines: list[str]) -> tuple[int, str]:
    if not lines:
        return 0, ""

    best_count = 1
    best_line = lines[0]
    curr_count = 1
    curr_line = lines[0]

    for line in lines[1:]:
        if line == curr_line:
            curr_count += 1
        else:
            if curr_count > best_count:
                best_count = curr_count
                best_line = curr_line
            curr_line = line
            curr_count = 1

    if curr_count > best_count:
        best_count = curr_count
        best_line = curr_line

    return best_count, best_line


def _normalize_free_text_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r'^[\[\]\(\)"\'\s>]+', '', line)
    line = re.sub(r'[\[\]\(\)"\'\s,.;:]+$', '', line)
    return re.sub(r"\s+", " ", line).strip()


def _is_mermaid_structure_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True

    lowered = stripped.lower()
    if lowered.startswith("graph ") or lowered.startswith("flowchart "):
        return True
    if lowered.startswith("subgraph ") or lowered == "end":
        return True
    if lowered.startswith("direction ") or lowered.startswith("%%"):
        return True
    if "-->" in stripped:
        return True

    return False


def _detect_raw_line_repetition(
    content: str,
    min_consecutive_repeats: int,
    min_total_line_repeats: int,
) -> list[RepetitionIssue]:
    """Fallback detection for malformed Mermaid where label regex may fail."""
    normalized = _normalize_escaped_newlines(content)
    lines: list[str] = []

    for raw in normalized.splitlines():
        if _is_mermaid_structure_line(raw):
            continue
        cleaned = _normalize_free_text_line(raw)
        if cleaned:
            lines.append(cleaned)

    if not lines:
        return []

    issues: list[RepetitionIssue] = []
    max_run, run_line = _max_consecutive_repeat(lines)
    if max_run >= min_consecutive_repeats:
        issues.append(
            RepetitionIssue(
                node_id="GLOBAL",
                reason="raw_consecutive_line_repetition",
                repeat_count=max_run,
                snippet=run_line[:120],
            )
        )

    freq: dict[str, int] = {}
    for line in lines:
        freq[line] = freq.get(line, 0) + 1

    line_text, max_count = max(freq.items(), key=lambda x: x[1])
    if max_count >= min_total_line_repeats:
        issues.append(
            RepetitionIssue(
                node_id="GLOBAL",
                reason="raw_overall_line_repetition",
                repeat_count=max_count,
                snippet=line_text[:120],
            )
        )

    return issues


def _node_id_token_counts(node_id: str) -> dict[str, int]:
    """Count 2-char tokens in a node id (handles concatenated AnAoApAq blobs)."""
    freq: dict[str, int] = {}
    for token in _NODE_ID_PAIR_RE.findall(node_id):
        freq[token] = freq.get(token, 0) + 1
    if freq:
        return freq

    # Fallback: sliding 2-char window on long alphanumeric runs
    for chunk in re.findall(r"[A-Za-z]{8,}", node_id):
        for i in range(len(chunk) - 1):
            pair = chunk[i : i + 2]
            freq[pair] = freq.get(pair, 0) + 1
    return freq


def _detect_node_id_issues(
    node_id: str,
    min_token_repeats: int,
    max_node_id_len: int,
) -> list[RepetitionIssue]:
    """Detect oversized node ids and repeated short tokens (e.g. AnAnAoApAq)."""
    issues: list[RepetitionIssue] = []
    if not node_id:
        return issues

    if len(node_id) > max_node_id_len:
        issues.append(
            RepetitionIssue(
                node_id=node_id[:80],
                reason="oversized_node_id",
                repeat_count=len(node_id),
                snippet=node_id[:120],
            )
        )

    freq = _node_id_token_counts(node_id)
    if freq:
        token, count = max(freq.items(), key=lambda x: x[1])
        if count >= min_token_repeats:
            issues.append(
                RepetitionIssue(
                    node_id=node_id[:80],
                    reason="node_id_token_repetition",
                    repeat_count=count,
                    snippet=token,
                )
            )

    return issues


def _detect_same_label_edge_repetition(
    content: str,
    min_same_label_edges: int,
) -> list[RepetitionIssue]:
    """Detect many edges sharing the same destination label (chain badcase)."""
    normalized = _normalize_escaped_newlines(content)
    label_counter: dict[str, int] = {}
    consecutive_label: str | None = None
    consecutive_run = 0
    best_run = 0
    best_label = ""

    for raw_line in normalized.splitlines():
        match = _EDGE_LABELED_RE.match(raw_line)
        if not match:
            consecutive_label = None
            consecutive_run = 0
            continue

        label = match.group("label").strip()
        if not label:
            continue

        label_counter[label] = label_counter.get(label, 0) + 1

        if label == consecutive_label:
            consecutive_run += 1
        else:
            consecutive_label = label
            consecutive_run = 1

        if consecutive_run > best_run:
            best_run = consecutive_run
            best_label = label

    issues: list[RepetitionIssue] = []
    for label, count in label_counter.items():
        if count >= min_same_label_edges:
            issues.append(
                RepetitionIssue(
                    node_id="EDGE_LABEL",
                    reason="same_label_edge_repetition",
                    repeat_count=count,
                    snippet=label[:120],
                )
            )

    if best_run >= min_same_label_edges:
        issues.append(
            RepetitionIssue(
                node_id="EDGE_LABEL_CHAIN",
                reason="consecutive_same_label_edges",
                repeat_count=best_run,
                snippet=best_label[:120],
            )
        )

    return issues


def _extract_edge_endpoint_id(segment: str) -> str:
    """Extract the node id from one side of an edge, preserving valid chained links."""
    cleaned = segment.strip()
    if not cleaned:
        return ""

    # Drop an edge label from destination segments like: |label| NodeId
    cleaned = re.sub(r"^\|[^|\n]*\|\s*", "", cleaned)

    # Keep only the leading node id before any label/shape suffix, e.g. A[Start].
    match = re.match(r"^([A-Za-z0-9_\u4e00-\u9fa5\-]+)", cleaned)
    if match:
        return match.group(1)

    return ""


def _detect_malformed_edge_line_issues(
    content: str,
    min_token_repeats: int,
    max_node_id_len: int,
) -> list[RepetitionIssue]:
    """Detect repetition inside edge endpoint ids without rejecting valid chained links."""
    normalized = _normalize_escaped_newlines(content)
    issues: list[RepetitionIssue] = []

    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        if "-->" not in stripped:
            continue

        if stripped.count("-->") >= 2:
            # Mermaid allows valid chains like A --> B --> C. Split and inspect
            # endpoint ids instead of treating chaining itself as repetition.
            endpoint_ids = [
                _extract_edge_endpoint_id(segment)
                for segment in stripped.split("-->")
            ]
            for endpoint_id in endpoint_ids:
                issues.extend(
                    _detect_node_id_issues(
                        endpoint_id,
                        min_token_repeats=min_token_repeats,
                        max_node_id_len=max_node_id_len,
                    )
                )
            continue

        match = _EDGE_LINE_RE.match(raw_line)
        if not match:
            continue

        for part in (match.group("src"), match.group("dst")):
            issues.extend(
                _detect_node_id_issues(
                    part,
                    min_token_repeats=min_token_repeats,
                    max_node_id_len=max_node_id_len,
                )
            )

    return issues

def detect_mermaid_repetition_issues(
    content: str,
    min_consecutive_repeats: int = 8,
    min_total_line_repeats: int = 20,
    max_label_length: int = 2000,
    min_same_edge_repeats: int = 8,
    max_node_id_len: int = DEFAULT_MAX_NODE_ID_LEN,
    debug: bool = False,
) -> list[RepetitionIssue]:
    """Detect suspicious repetition in Mermaid labels, edges and raw text."""
    issues: list[RepetitionIssue] = []

    # 1) Label-level detection (works when labels are syntactically valid).
    for match in _NODE_LABEL_RE.finditer(content):
        node_id = match.group("node")
        issues.extend(
            _detect_node_id_issues(
                node_id,
                min_token_repeats=min_consecutive_repeats,
                max_node_id_len=max_node_id_len,
            )
        )
        label = _normalize_escaped_newlines(match.group("label"))

        lines = [line.strip() for line in label.splitlines() if line.strip()]
        if not lines:
            continue

        max_run, run_line = _max_consecutive_repeat(lines)
        if max_run >= min_consecutive_repeats:
            issues.append(
                RepetitionIssue(
                    node_id=node_id,
                    reason="consecutive_line_repetition",
                    repeat_count=max_run,
                    snippet=run_line[:120],
                )
            )
            continue

        freq: dict[str, int] = {}
        for line in lines:
            freq[line] = freq.get(line, 0) + 1
        line_text, max_count = max(freq.items(), key=lambda x: x[1])
        if max_count >= min_total_line_repeats:
            issues.append(
                RepetitionIssue(
                    node_id=node_id,
                    reason="overall_line_repetition",
                    repeat_count=max_count,
                    snippet=line_text[:120],
                )
            )
            continue

        unique_ratio = len(set(lines)) / max(len(lines), 1)
        if len(label) >= max_label_length and unique_ratio <= 0.2:
            issues.append(
                RepetitionIssue(
                    node_id=node_id,
                    reason="long_low_diversity_label",
                    repeat_count=len(lines),
                    snippet=lines[0][:120],
                )
            )

    # 2) Edge-level repetition (after escaped newline normalization).
    normalized_for_edges = _normalize_escaped_newlines(content)
    edge_counter: dict[str, int] = {}
    for raw_line in normalized_for_edges.splitlines():
        match = _EDGE_LINE_RE.match(raw_line)
        if not match:
            continue
        edge_key = f"{match.group('src')}-->{match.group('dst')}"
        edge_counter[edge_key] = edge_counter.get(edge_key, 0) + 1

    for edge_key, count in edge_counter.items():
        if count >= min_same_edge_repeats:
            issues.append(
                RepetitionIssue(
                    node_id=edge_key,
                    reason="same_edge_repetition",
                    repeat_count=count,
                    snippet=edge_key,
                )
            )

    # 2b) Malformed edge lines (chained arrows, oversized swallowed node ids).
    issues.extend(
        _detect_malformed_edge_line_issues(
            content,
            min_token_repeats=min_consecutive_repeats,
            max_node_id_len=max_node_id_len,
        )
    )

    # 2c) Many edges sharing the same label text (chain badcase).
    issues.extend(
        _detect_same_label_edge_repetition(
            content,
            min_same_label_edges=min_same_edge_repeats,
        )
    )

    # 3) Raw-text fallback for malformed labels.
    issues.extend(
        _detect_raw_line_repetition(
            content,
            min_consecutive_repeats=min_consecutive_repeats,
            min_total_line_repeats=min_total_line_repeats,
        )
    )

    if debug:
        for issue in issues:
            print(
                f"[mermaid_repetition] node={issue.node_id} reason={issue.reason} "
                f"repeats={issue.repeat_count} snippet={issue.snippet!r}"
            )

    return issues


def detect_mermaid_label_repetition_issues(
    content: str,
    min_consecutive_repeats: int = 8,
    min_total_line_repeats: int = 20,
    max_label_length: int = 2000,
    min_same_edge_repeats: int = 20,
) -> list[RepetitionIssue]:
    """Backward-compatible alias."""
    return detect_mermaid_repetition_issues(
        content,
        min_consecutive_repeats=min_consecutive_repeats,
        min_total_line_repeats=min_total_line_repeats,
        max_label_length=max_label_length,
        min_same_edge_repeats=min_same_edge_repeats,
        debug=False,
    )


if __name__ == "__main__":
    # Example 1: malformed label with repeated '> top_valent' lines.
    sample_top_valent = r'''graph TD
  A["[Chrome] 云平台自动删除\\n60%用户 + 2 转换 + 20%条件验证"] --> B["[Clean] OpenClean Cross 视频 finance_malestate_agent.py\\n> top_valent\\n> top_valent\\n> top_valent\\n> top_valent\\n> top_valent\\n> top_valent\\n> top_valent\\n> top_valent\\n> top_valent\\n> top_valent\\n]
'''

    # Example 2: valid label with repeated Chinese lines.
    sample_register = r'''graph TD
  A["注册登记\\n完整版登录www.leadleo.com"] --> B["注册登记\\n搜索《中国摩托车出海展望》"]
  B --> C["注册登记\\n完成注册登记\\n完成注册登记\\n完成注册登记\\n完成注册登记\\n完成注册登记\\n完成注册登记\\n完成注册登记\\n完成注册登记\\n完成注册登记\\n完成注册登记"]
'''

    # Example 3: repeated edge lines.
    sample_edges = r'''graph LR
  A["平台"] --> B["教师培训"]
  B --> C["竞赛"]
  C --> D["社区"]
  D --> E["硬件"]
  E --> F["前沿内容"]
  F --> G["课程"]
  G --> H["教学装备"]
  H --> I["成长社区"]
  I --> J["成长社区"]
  J --> K["成长社区"]
  K --> L["成长社区"]
  L --> M["成长社区"]
  M --> N["成长社区"]
  N --> O["成长社区"]
  O --> P["成长社区"]
  P --> Q["成长社区"]
  Q --> R["成长社区"]
  R --> S["成长社区"]
  S --> T["成长社区"]
  T --> U["成长社区"]
  U --> V["成长社区"]
  V --> W["成长社区"]
  W --> X["成长社区"]
  X --> Y["成长社区"]
  Y --> Z["成长社区"]
  Z --> AA["成长社区"]
  AA --> AB["成长社区"]
  AB --> AC["成长社区"]
  AC --> AD["成长社区"]
  AD --> AE["成长社区"]
  AE --> AF["成长社区"]
  AF --> AG["成长社区"]
  AG --> AH["成长社区"]
  AH --> AI["成长社区"]
  AI --> AJ["成长社区"]
  AJ --> AK["成长社区"]
  AK --> AL["成长社区"]
  AL --> AM["成长社区"]
  AM --> AN["成长社区"]
  AN --> AO["成长社区"]
  AO --> AP["成长社区"]
  AP --> AQ["成长社区"]
  AQ --> AR["成长社区"]
  AR --> AS["成长社区"]
  AS --> AT["成长社区"]
  AT --> AU["成长社区"]
  AU --> AV["成长社区"]
  AV --> AW["成长社区"]
  AW --> AX["成长社区"]
  AX --> AY["成长社区"]
  AY --> AZ["成长社区"]
  AZ --> BA["成长社区"]
  BA --> BA --> AC["成长社区"]
  AC --> AD["成长社区"]
  AD --> AE["成长社区"]
  AE --> AF["成长社区"]
  AF --> AG["成长社区"]
  AG --> AH --> AI["成长社区"]
  AI --> AJAkAlAmAnAoApAqAqAlAmAnAoApAqAlAnAoApAqAlAnAoApAqAlAnAoApAqAlAnAoApAqAlAnAoApAqAlAnAoApAqAlAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAnAoApAqAnAn["AJ --> AK --> AL --> AM --> AN --> AO --> AP --> AQ --> AQ --> AL --> AM --> AN --> AO --> AP --> AQ --> AL --> AN --> AO --> AP --> AQ --> AL --> AN --> AO --> AP --> AQ --> AL --> AN --> AO --> AP --> AQ --> AL --> AN --> AO --> AP --> AQ --> AL --> AN --> AO --> AP --> AQ --> AL --> AN --> AO --> AP --> AQ --> AL --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN --> AO --> AP --> AQ --> AN --> AN"]
'''

    print("[demo] sample_top_valent")
    detect_mermaid_repetition_issues(sample_top_valent, debug=True)

    print("[demo] sample_register")
    detect_mermaid_repetition_issues(sample_register, debug=True)

    print("[demo] sample_edges")
    detect_mermaid_repetition_issues(sample_edges, min_same_edge_repeats=8, debug=True)
