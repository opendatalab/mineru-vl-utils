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
    """
    严格提取并校验 Mermaid flowchart 代码，修复常见的格式和语法问题。
    支持修复：
    1. 不规范的 Markdown 代码块
    2. 声明头部拼写错误 (如 grap -> graph)
    3. 错误的连线箭头 (如 ->, - -> 修正为 -->)
    4. 节点 ID 中的空格 (替换为下划线)
    5. 节点文本中的嵌套双引号 (转义为 &quot;) 和换行符问题
    """
    if not content or not content.strip():
        return ""

    content = content.strip()

    # ==========================================
    # 步骤 1: 提取核心的 mermaid 代码
    # ==========================================
    mermaid_match = re.search(r"```mermaid\s*(.*?)\s*```", content, re.DOTALL)
    if mermaid_match:
        code = mermaid_match.group(1).strip()
    else:
        code_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = content.strip()

    if code.lower().startswith("mermaid"):
        code = code[7:].strip()

    # ==========================================
    # 步骤 2: 修复图表声明拼写错误
    # ==========================================
    # 修复常见的大小写或拼写错误
    code = re.sub(r"^(grap|grapg|graphh)\b", "graph", code, flags=re.IGNORECASE)
    code = re.sub(r"^(flowchar|flowchartt)\b", "flowchart", code, flags=re.IGNORECASE)

    # ==========================================
    # 步骤 3: 修复不规范的连线箭头
    # ==========================================
    # 修复带有意外空格的箭头: - ->, -- >, -  -> 等，统一替换为 -->
    code = re.sub(r"-\s+->", "-->", code)
    code = re.sub(r"--\s+>", "-->", code)
    code = re.sub(r"-\s+-\s+>", "-->", code)
    # 修复漏写减号的箭头 -> (使用负向后瞻 (?<![-=]) 确保前面不是 - 或 =，防止误伤正常箭头)
    code = re.sub(r"(?<![-=])\s+->\s+", " --> ", code)

    # ==========================================
    # 步骤 4: 修复节点 ID 和 嵌套双引号
    # ==========================================

    def node_fixer(match):
        raw_node_id = match.group(1).strip()
        # 1. 修复节点 ID 里的空格：仅将水平空格/制表符替换为下划线，不影响换行符
        node_id = re.sub(r"[ \t]+", "_", raw_node_id)

        raw_text = match.group(2)

        # 2. 如果文本为空，直接返回
        if not raw_text:
            return f'{node_id}[]'

        # 3. 剥离原有的外层双引号
        if raw_text.startswith('"') and raw_text.endswith('"'):
            raw_text = raw_text[1:-1]

        # 4. 转义文本内部残留的双引号！
        safe_text = raw_text.replace('"', '&quot;')

        # 5. [新增] 将物理换行符替换为 Mermaid 友好的 <br> 标签，防止渲染崩溃
        safe_text = safe_text.replace('\n', '<br>')

        # 6. 重新用标准的双引号包裹
        return f'{node_id}["{safe_text}"]'

    # 正则解析修改：
    # 将原来的 \s 替换为 [ \t]，严格限制节点 ID 不能跨行！
    processed_code = re.sub(
        r'([a-zA-Z0-9_\u4e00-\u9fa5\-]+(?:[ \t]+[a-zA-Z0-9_\u4e00-\u9fa5\-]+)*)[ \t]*\[(.*?)\]', 
        node_fixer, 
        code, 
        flags=re.DOTALL
    )

    # ==========================================
    # 步骤 5: 返回标准格式
    # ==========================================
    return f"```mermaid\n{processed_code}\n```"


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
    if class_name == "flowchart":
        normalized_content = extract_and_validate_mermaid_strict(normalized_content)

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
<|caption_start|>Formula处理流程图，展示formula输入经formula valid与Formula Refiner（自带Unify效果）处理并输出公式后处理的过程<|caption_end|>
<|content_start|>

graph TD
    A[formula] --> B[formula valid]
    A --> C[formula valid]
    B --> D[Formula Unifier]
    C --> E[Formula Refiner (自带Unify效果)]
    D --> F[公式后处理]
    E --> F

<|content_end|>

    """
    print(process_image_or_chart(content))