import re
import subprocess
import tempfile
import os

IMAGE_CHART_FIELD_TAGS = {
    "class": ("<|class_start|>", "<|class_end|>"),
    "sub_class": ("<|sub_class_start|>", "<|sub_class_end|>"),
    "caption": ("<|caption_start|>", "<|caption_end|>"),
    "content": ("<|content_start|>", "<|content_end|>"),
}


def _extract_tagged_field(text: str, start_tag: str, end_tag: str) -> str:
    start_idx = text.find(start_tag)
    if start_idx < 0:
        return ""
    start_idx += len(start_tag)
    end_idx = text.find(end_tag, start_idx)
    if end_idx < 0:
        return ""
    return text[start_idx:end_idx].strip()



def _count_markdown_table_columns(line: str) -> int:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return len([cell for cell in stripped.split("|")])


def _is_markdown_table_separator_line(line: str) -> bool:
    # Example: | --- | :---: | ---: |
    return re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", line) is not None


def _is_markdown_table_row_candidate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return "|" in stripped and _count_markdown_table_columns(stripped) >= 2


def has_malformed_markdown_table(content: str) -> bool:
    lines = content.splitlines()
    separator_indices = [i for i, line in enumerate(lines) if _is_markdown_table_separator_line(line)]
    row_candidate_indices = [i for i, line in enumerate(lines) if _is_markdown_table_row_candidate(line)]
    found_valid_table = False

    for sep_idx in separator_indices:
        # 找 separator 前最近一行非空作为表头
        header_idx = sep_idx - 1
        while header_idx >= 0 and not lines[header_idx].strip():
            header_idx -= 1
        if header_idx < 0 or not _is_markdown_table_row_candidate(lines[header_idx]):
            return True

        header_cols = _count_markdown_table_columns(lines[header_idx])
        sep_cols = _count_markdown_table_columns(lines[sep_idx])
        if header_cols < 2 or sep_cols != header_cols:
            return True

        # 检查数据行列数是否一致（直到遇到空行或非表格行）
        row_idx = sep_idx + 1
        while row_idx < len(lines):
            row_line = lines[row_idx]
            if not row_line.strip():
                break
            if not _is_markdown_table_row_candidate(row_line):
                break
            if _count_markdown_table_columns(row_line) != header_cols:
                return True
            row_idx += 1
        found_valid_table = True

    # 有明显表格行但没有合法表格结构（常见于缺失/损坏 separator）
    if len(row_candidate_indices) >= 2 and not found_valid_table:
        return True

    return False


def extract_and_validate_mermaid_strict(content: str) -> str:
    content = content.strip()
    if not content:
        return ""

    # 2. 使用临时文件调用 mmdc 进行严格校验
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mmd", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_mmd = f.name

    temp_svg = temp_mmd + ".svg"

    try:
        # 调用 mmdc 尝试将其编译为 svg (静默模式)
        # 如果格式有语法错误，mmdc 会返回非 0 的退出码
        result = subprocess.run(
            ["mmdc", "-i", temp_mmd, "-o", temp_svg],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return content  # 编译成功，格式完全正确
        else:
            # print("Mermaid Syntax Error:", result.stderr) # 调试时可取消注释查看具体报错
            return ""  # 编译失败，返回空

    except FileNotFoundError:
        return ""
    finally:
        # 清理生成的临时文件
        if os.path.exists(temp_mmd):
            os.remove(temp_mmd)
        if os.path.exists(temp_svg):
            os.remove(temp_svg)


def process_image_or_chart(content: str) -> str:
    values = {
        field: _extract_tagged_field(content, tags[0], tags[1]) for field, tags in IMAGE_CHART_FIELD_TAGS.items()
    }

    class_name = values["class"].strip().lower()
    normalized_content = values["content"]

    # 1) markdown 表格语法有误或不闭合：content 置空
    if normalized_content and has_malformed_markdown_table(normalized_content):
        normalized_content = ""

    # 2) chemical 类别：content 置空
    if class_name == "chemical":
        normalized_content = ""
    
    

    # 3) flowchart 类别：严格校验并提取 mermaid
    # if class_name == "flowchart":
    #     normalized_content = extract_and_validate_mermaid_strict(normalized_content)

    values["content"] = normalized_content

    return "\n".join(
        [
            f"**class:**\n{values['class']}",
            f"**sub_class:**\n{values['sub_class']}",
            f"**caption:**\n{values['caption']}",
            f"**content:**\n{values['content']}",
        ]
    )


if __name__ == "__main__":
    content = """
<|class_start|>flowchart<|class_end|>
<|sub_class_start|>flowchart<|sub_class_end|>
<|caption_start|>Text processing flowchart showing data flow from text through validation and refiners to final text after processing<|caption_end|>
<|content_start|>

```mermaid
graph TD
    A[text] --> B[text valid]
    A --> C[text Refiner]
    B --> D[Text Unifier]
    C --> E[Text Unifier]
    D --> F[文本后处理]
    E --> F
```

<|content_end|>
    """
    print(process_image_or_chart(content))