from ..structs import ContentBlock
from .equation_big import try_fix_equation_big
from .equation_block import do_handle_equation_block
from .equation_double_subscript import try_fix_equation_double_subscript
from .equation_fix_eqqcolon import try_fix_equation_eqqcolon
from .equation_left_right import try_match_equation_left_right
from .equation_leq import try_fix_equation_leq
from .equation_unbalanced_braces import try_fix_unbalanced_braces
from .equation_delimeters import try_fix_equation_delimeters
from .text_inline_spacing import try_fix_macro_spacing_in_markdown
from .text_display2inline import try_convert_display_to_inline
from .text_move_underscores_outside import try_move_underscores_outside
from .image_analysis_postprocess import process_image_or_chart
from .otsl2html import convert_otsl_to_html
from .table_image_processor import (
    cleanup_table_image_metadata,
    is_absorbed_table_image,
    replace_table_formula_delimiters,
    replace_table_image_tokens,
    TABLE_IMAGE_TOKEN_MAP_KEY,
)

PARATEXT_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "unknown",
}


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
                print("Warning: Failed to convert OTSL to HTML: ", e)
                print("Content: ", block.content)
            content = replace_table_image_tokens(content, block.get(TABLE_IMAGE_TOKEN_MAP_KEY))
            block.content = replace_table_formula_delimiters(content, enabled=enable_table_formula_eq_wrap)
        if block.type in {"image", "chart"} and block.content:
            try:
                block.content = process_image_or_chart(block.content)
            except Exception as e:
                print("Warning: Failed to process image/chart: ", e)
                print("Content: ", block.content)
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

    if simple_post_process:
        return _finalize_simple_blocks(blocks)

    for block in blocks:
        if block.type == "equation" and block.content:
            try:
                block.content = _process_equation(block.content, debug=debug)
            except Exception as e:
                print("Warning: Failed to process equation: ", e)
                print("Content: ", block.content)
                
        elif block.type == "text" and block.content:
            try:
                block.content = try_convert_display_to_inline(block.content, debug=debug)
                block.content = try_fix_macro_spacing_in_markdown(block.content, debug=debug)
                block.content = try_move_underscores_outside(block.content, debug=debug)
            except Exception as e:
                print("Warning: Failed to process text: ", e)
                print("Content: ", block.content)

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
