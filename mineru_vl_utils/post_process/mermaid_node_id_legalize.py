import re


_VALID_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXISTING_DECL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[\[\(\{]")


def _sanitize_node_id(raw_id: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", raw_id)
    if parts:
        first = parts[0]
        tail = "".join(part.capitalize() for part in parts[1:])
        candidate = f"{first}{tail}"
    else:
        candidate = re.sub(r"[^A-Za-z0-9_]", "", raw_id)

    if not candidate:
        candidate = "node"
    if candidate[0].isdigit():
        candidate = f"n{candidate}"
    return candidate


def _sanitize_edge_label(raw_label: str) -> str:
    # Some Mermaid parsers choke on parentheses inside |...| edge labels.
    # Remove them conservatively to keep labels parse-safe.
    label = raw_label.replace("(", "").replace(")", "")
    label = re.sub(r" {2,}", " ", label)
    return label.strip()


def _looks_plain_endpoint(token: str) -> bool:
    if not token:
        return False
    return not any(ch in token for ch in '[](){}"|')


def _convert_endpoint(token: str, id_map: dict[str, str], declared_ids: set[str]) -> str:
    raw = token.strip()
    if not _looks_plain_endpoint(raw):
        return token

    if _VALID_ID_RE.fullmatch(raw):
        return raw

    fixed = id_map.get(raw)
    if fixed is None:
        fixed = _sanitize_node_id(raw)
        base = fixed
        idx = 2
        while fixed in declared_ids or fixed in id_map.values():
            fixed = f"{base}{idx}"
            idx += 1
        id_map[raw] = fixed

    if fixed not in declared_ids:
        declared_ids.add(fixed)
        # First appearance uses an explicit label to preserve original text.
        return f'{fixed}["{raw}"]'

    return fixed


def try_fix_mermaid_node_id_legalization(content: str, debug: bool = False) -> str:
    lines = content.splitlines()
    declared_ids: set[str] = set()

    for line in lines:
        match = _EXISTING_DECL_RE.match(line.strip())
        if match:
            declared_ids.add(match.group(1))

    id_map: dict[str, str] = {}
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Keep bidirectional arrows unchanged; this fixer only normalizes single-direction edges.
        if re.search(r"<\s*-\s*-\s*>", line):
            out.append(line)
            continue

        if "-->" not in line or stripped.startswith("%%"):
            out.append(line)
            continue

        arrow_idx = line.find("-->")
        left = line[:arrow_idx]
        right = line[arrow_idx + 3 :]

        src = left.strip()
        src_fixed = _convert_endpoint(src, id_map, declared_ids)

        right_stripped = right.strip()
        label_match = re.match(r"^\|(?P<label>.*?)\|\s*(?P<dst>.+?)\s*$", right_stripped)
        if label_match:
            raw_label = label_match.group("label")
            label = _sanitize_edge_label(raw_label)
            dst = label_match.group("dst")
            dst_fixed = _convert_endpoint(dst, id_map, declared_ids)
            new_line = f"  {src_fixed} -->|{label}| {dst_fixed}"
        else:
            dst_fixed = _convert_endpoint(right_stripped, id_map, declared_ids)
            new_line = f"  {src_fixed} --> {dst_fixed}"

        if debug and new_line != line:
            print(f"[mermaid_node_id_legalize] {line} -> {new_line}")
        out.append(new_line)

    fixed = "\n".join(out)
    if content.endswith("\n"):
        fixed += "\n"
    return fixed


if __name__ == "__main__":
    mermaid = r'''graph TD
  Sunlight --> |Textured front surface\nAnti-reluctiv coating (SiNx)| Top Layer
  Sunlight --> |Holes| Top Layer
  Top Layer --> |Front silver fingers| Top Layer
  Top Layer --> |Tunnel oxide-2 nm| Bottom Layer
  Bottom Layer --> |Passivated contact reduces recombination| Top Layer
  Bottom Layer --> |TOPCon stack\nn+ poly_Si (~100-200 nm) (electron-tunleling)| Bottom Layer
  Bottom Layer --> |Electrons| Bottom Layer
  Bottom Layer --> |Rear metal contact\nBusbars| Bottom Layer
  Bottom Layer --> |Electrons| Bottom Layer
'''

    print(try_fix_mermaid_node_id_legalization(mermaid, debug=True))
