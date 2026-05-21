from loguru import logger

from ..structs import BlockType, ContentBlock
from .equation_big import try_fix_equation_big
from .equation_block import do_handle_equation_block
from .equation_delimeters import try_fix_equation_delimeters
from .equation_double_subscript import try_fix_equation_double_subscript
from .equation_fix_eqqcolon import try_fix_equation_eqqcolon
from .equation_left_right import try_match_equation_left_right
from .equation_leq import try_fix_equation_leq
from .equation_unbalanced_braces import try_fix_unbalanced_braces
from .image_analysis_postprocess import convert_markdown_table_to_html, process_image_or_chart
from .json2markdown import json2md
from .otsl2html import convert_otsl_to_html
from .table_image_processor import (
    TABLE_IMAGE_TOKEN_MAP_KEY,
    cleanup_table_image_metadata,
    is_absorbed_table_image,
    replace_table_formula_delimiters,
    replace_table_image_tokens,
)
from .text_display2inline import try_convert_display_to_inline
from .text_inline_spacing import try_fix_macro_spacing_in_markdown
from .text_move_underscores_outside import try_move_underscores_outside

PARATEXT_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "unknown",
}

NON_TEXT_PLACEHOLDER = "[Non-Text]"

NON_TEXT_PLACEHOLDER_KEEP_TYPES = {
    BlockType.HEADER,
    BlockType.FOOTER,
}

NON_TEXT_PLACEHOLDER_PROTECTED_TYPES = {
    BlockType.IMAGE,
    BlockType.CHART,
    BlockType.TABLE,
    BlockType.EQUATION,
    BlockType.LIST,
    BlockType.IMAGE_BLOCK,
    BlockType.EQUATION_BLOCK,
}

_OTSL_TABLE_TOKENS = ("<nl>", "<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>")


def _process_equation(content: str, debug: bool) -> str:
    content = try_fix_equation_delimeters(content, debug=debug)
    content = try_match_equation_left_right(content, debug=debug)
    content = try_fix_equation_double_subscript(content, debug=debug)
    content = try_fix_equation_eqqcolon(content, debug=debug)
    content = try_fix_equation_big(content, debug=debug)
    content = try_fix_equation_leq(content, debug=debug)
    content = try_fix_unbalanced_braces(content, debug=debug)
    return content


def _add_equation_brackets(content: str) -> str:
    content = content.strip()
    if not content.startswith("\\["):
        content = f"\\[\n{content}"
    if not content.endswith("\\]"):
        content = f"{content}\n\\]"
    return content


def _convert_pure_table_content_to_html(content: str) -> str:
    if not content or not content.strip():
        return ""

    stripped_content = content.strip()
    if stripped_content.lower().startswith("<table") and stripped_content.lower().endswith("</table>"):
        return stripped_content

    markdown_html = convert_markdown_table_to_html(content)
    if markdown_html is not None:
        return markdown_html

    if any(token in content for token in _OTSL_TABLE_TOKENS):
        try:
            otsl_html = convert_otsl_to_html(content)
        except Exception as e:
            logger.warning("Failed to convert pure_table OTSL to HTML: {}; content: {}", e, content)
            return ""

        if not otsl_html or not otsl_html.strip():
            logger.warning("Failed to convert pure_table OTSL to HTML: {}", content)
            return ""

        return otsl_html

    logger.warning("Failed to recognize pure_table format: {}", content)
    return ""


def _cleanup_non_text_placeholder_blocks(blocks: list[ContentBlock]) -> list[ContentBlock]:
    """清理模型输出的[Non-Text]占位块，同时保留页眉页脚的版面框。"""
    out_blocks: list[ContentBlock] = []
    for block in blocks:
        content = block.content
        if content is None or content.strip() != NON_TEXT_PLACEHOLDER:
            out_blocks.append(block)
            continue

        if block.type in NON_TEXT_PLACEHOLDER_KEEP_TYPES:
            block.content = ""
            out_blocks.append(block)
            continue

        if block.type in NON_TEXT_PLACEHOLDER_PROTECTED_TYPES:
            out_blocks.append(block)

    return out_blocks


def simple_process(
    blocks: list[ContentBlock],
    enable_table_formula_eq_wrap: bool = False,
) -> list[ContentBlock]:
    for block in blocks:
        if block.type == "table" and block.content:
            content = block.content
            try:
                content = convert_otsl_to_html(content)
            except Exception as e:
                logger.warning("Failed to convert OTSL to HTML: {}; content: {}", e, block.content)
            content = replace_table_image_tokens(content, block.get(TABLE_IMAGE_TOKEN_MAP_KEY))
            block.content = replace_table_formula_delimiters(content, enabled=enable_table_formula_eq_wrap)
        if block.type in {"image", "chart"} and block.content:
            try:
                block_image_analysis_result = process_image_or_chart(block.content)
                class_name = block_image_analysis_result["class"]
                content = block_image_analysis_result["content"]
                if class_name == "pure_table":
                    block.type = "table"
                    table_html = _convert_pure_table_content_to_html(content)
                    if table_html:
                        block.content = replace_table_formula_delimiters(
                            table_html,
                            enabled=enable_table_formula_eq_wrap,
                        )
                    else:
                        block.content = ""
                elif class_name == "pure_formula":
                    block.type = "equation"
                    block.content = content
                elif class_name == "chart":
                    block.type = "chart"
                    block["sub_type"] = block_image_analysis_result["sub_class"]
                    block.content = content
                else:
                    block.type = "image"
                    block["sub_type"] = class_name
                    if class_name == "natural_image" or not content:
                        block.content = block_image_analysis_result["caption"]
                    else:
                        block.content = content

            except Exception as e:
                logger.warning("Failed to process image/chart: {}; content: {}", e, block.content)
                block.content = None  # or keep original content, depending on your preference
    return blocks


def _finalize_simple_blocks(blocks: list[ContentBlock]) -> list[ContentBlock]:
    out_blocks = [block for block in blocks if not (block.type == "image" and is_absorbed_table_image(block))]
    return cleanup_table_image_metadata(out_blocks)


def post_process(
    blocks: list[ContentBlock],
    simple_post_process: bool,
    handle_equation_block: bool,
    abandon_list: bool,
    abandon_paratext: bool,
    enable_table_formula_eq_wrap: bool = False,
    debug: bool = False,
) -> list[ContentBlock]:
    blocks = simple_process(blocks, enable_table_formula_eq_wrap=enable_table_formula_eq_wrap)

    for block in blocks:
        if block.type == "list_item":
            block.type = "text"

    blocks = _cleanup_non_text_placeholder_blocks(blocks)

    if simple_post_process:
        return _finalize_simple_blocks(blocks)

    for block in blocks:
        if block.type == "equation" and block.content:
            try:
                block.content = _process_equation(block.content, debug=debug)
            except Exception as e:
                logger.warning("Failed to process equation: {}; content: {}", e, block.content)

        elif block.type == "text" and block.content:
            try:
                block.content = try_convert_display_to_inline(block.content, debug=debug)
                block.content = try_fix_macro_spacing_in_markdown(block.content, debug=debug)
                block.content = try_move_underscores_outside(block.content, debug=debug)
            except Exception as e:
                logger.warning("Failed to process text: {}; content: {}", e, block.content)

    if handle_equation_block:
        blocks = do_handle_equation_block(blocks, debug=debug)

    for block in blocks:
        if block.type == "equation" and block.content:
            block.content = _add_equation_brackets(block.content)

    out_blocks: list[ContentBlock] = []
    for block in blocks:
        if block.type == "equation_block":  # drop equation_block anyway
            continue
        if block.type == "image" and is_absorbed_table_image(block):
            continue
        if abandon_list and block.type == "list":
            continue
        if abandon_paratext and block.type in PARATEXT_TYPES:
            continue
        out_blocks.append(block)

    return cleanup_table_image_metadata(out_blocks)
