import re


_SUBGRAPH_LINE_RE = re.compile(r"^(\s*)subgraph\s+(.+?)\s*$")
_VALID_SUBGRAPH_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sanitize_subgraph_id(raw: str) -> str:
    sid = re.sub(r"[^A-Za-z0-9_]+", "_", raw)
    sid = re.sub(r"_+", "_", sid).strip("_")
    if not sid:
        sid = "sg"
    if sid[0].isdigit():
        sid = f"sg_{sid}"
    return sid


def _is_already_explicit_subgraph(line_body: str) -> bool:
    # e.g. subgraph id["Title"] or subgraph id("Title")
    if "[" in line_body or "(" in line_body:
        first = line_body.split(None, 1)[0] if line_body.strip() else ""
        return _VALID_SUBGRAPH_ID_RE.match(first) is not None
    return False


def _build_unique_id(base_id: str, seen_ids: set[str]) -> str:
    if base_id not in seen_ids:
        seen_ids.add(base_id)
        return base_id

    idx = 2
    while f"{base_id}_{idx}" in seen_ids:
        idx += 1
    new_id = f"{base_id}_{idx}"
    seen_ids.add(new_id)
    return new_id


def try_fix_mermaid_invalid_subgraph_lines(content: str, debug: bool = False) -> str:
    """Fix Mermaid subgraph lines with invalid IDs.

    Example fixed line:
        subgraph 1) Trigger
    =>
        subgraph sg_1_Trigger["1) Trigger"]
    """
    fixed_lines: list[str] = []
    seen_subgraph_ids: set[str] = set()

    for idx, line in enumerate(content.splitlines(), start=1):
        m = _SUBGRAPH_LINE_RE.match(line)
        if not m:
            fixed_lines.append(line)
            continue

        indent, body = m.groups()

        if _is_already_explicit_subgraph(body):
            first = body.split(None, 1)[0]
            seen_subgraph_ids.add(first)
            fixed_lines.append(line)
            continue

        parts = body.split(None, 1)
        if not parts:
            fixed_lines.append(line)
            continue

        first_token = parts[0]
        if _VALID_SUBGRAPH_ID_RE.match(first_token):
            seen_subgraph_ids.add(first_token)
            fixed_lines.append(line)
            continue

        title = body
        base_id = _sanitize_subgraph_id(title)
        sid = _build_unique_id(base_id, seen_subgraph_ids)
        escaped_title = title.replace('"', '\\"')
        new_line = f'{indent}subgraph {sid}["{escaped_title}"]'

        if debug:
            print(f"[mermaid_invalid_subgraph] fix L{idx}: {line} -> {new_line}")

        fixed_lines.append(new_line)

    result = "\n".join(fixed_lines)
    if content.endswith("\n"):
        result += "\n"
    return result


if __name__ == "__main__":
    mermaid = r'''graph LR
  subgraph 1) Trigger
    Webhook["Webhook"]
    Schedule["Schedule"]
    Email["Email"]
    Form["Form"]
  end

  subgraph 2)_n8n_WorkFlow_Engine
    WorkflowEngine["n8n Workflow Engine\nwith visual node-based editor"]
  end

  subgraph 3)_iApp_AI_APIs
    AIA["iApp AI APIs"]
    OCR["OCR"]
    FaceVerification["Face Verification"]
    Translation["Translation"]
  end

  subgraph 4) Actions
    Database["Database"]
    Email["Email"]
    Slack["Slack"]
    GoogleSheets["Google Sheets"]
    CRM["CRM"]
  end

  %% Connections from 1) Trigger to 2) WorkflowEngine
  %% Connections from 2) WorkflowEngine to 3) iApp_AI_APIs
  %% Connections from 3) iApp_AI_APIs to 4) Actions
'''
    print(try_fix_mermaid_invalid_subgraph_lines(mermaid, debug=True))
