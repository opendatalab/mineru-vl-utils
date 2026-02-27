import os
import hashlib
import re
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageStat
from ..structs import ContentBlock

FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "arial.ttf",
]

def load_font(size: int):
    """
    Load a font with the specified size.
    Args:
        size (int): The font size.
    Returns:
        ImageFont.FreeTypeFont: The loaded font object.
    """
    for path in FONT_PATH_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()

def get_optimal_pil_font(
    text: str,
    box_w: int,
    box_h: int,
    fill_ratio: float = 0.9,
    min_size: int = 4,
    max_size: int = 256,
):
    """
    Find the optimal font size to fit text within a box.
    Args:
        text (str): The text to render.
        box_w (int): Width of the bounding box.
        box_h (int): Height of the bounding box.
        fill_ratio (float): The ratio of the box to fill.
        min_size (int): Minimum font size.
        max_size (int): Maximum font size.
    Returns:
        Tuple[ImageFont.FreeTypeFont, int, int]: Best font, text width, text height.
    """
    left, right = min_size, max_size
    best_size = left
    best_font = load_font(best_size)
    best_w, best_h = 0, 0

    for _ in range(30):
        if left > right:
            break
        mid = (left + right) // 2
        font = load_font(mid)
        try:
            bbox = font.getbbox(text)
        except AttributeError:
             # Fallback for older PIL
             w, h = font.getsize(text)
             bbox = (0, 0, w, h)

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if w <= box_w * fill_ratio and h <= box_h * fill_ratio:
            best_size = mid
            best_font = font
            best_w, best_h = w, h
            left = mid + 1
        else:
            right = mid - 1

    return best_font, best_w, best_h

def get_average_color(image: Image.Image, box) -> Tuple[int, int, int]:
    """
    Calculate the average color of a region in an image.
    Args:
        image (Image.Image): The source image.
        box (tuple): The region to crop (left, upper, right, lower).
    Returns:
        Tuple[int, int, int]: Average RGB color.
    """
    try:
        region = image.crop(box)
        stat = ImageStat.Stat(region)
        return tuple(map(int, stat.mean[:3]))
    except Exception:
        return (255, 255, 255)

def get_contrast_text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Determine whether black or white text contrasts better with the background.
    Args:
        bg_color (Tuple[int, int, int]): The background RGB color.
    Returns:
        Tuple[int, int, int]: (255, 255, 255) for white or (0, 0, 0) for black.
    """
    r, g, b = bg_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)

def _bbox_intersection_area(a, b) -> float:
    """
    Calculate the intersection area of two bounding boxes.
    Args:
        a (tuple): First bbox (x1, y1, x2, y2).
        b (tuple): Second bbox (x1, y1, x2, y2).
    Returns:
        float: Intersection area.
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)

def _bbox_area(a) -> float:
    """
    Calculate the area of a bounding box.
    Args:
        a (tuple): Bbox (x1, y1, x2, y2).
    Returns:
        float: Area.ƒ
    """
    return max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))

def _overlap_ratio(inner, outer) -> float:
    """
    Calculate the ratio of the inner box's area that overlaps with the outer box.
    Args:
        inner (tuple): The inner bbox (x1, y1, x2, y2).
        outer (tuple): The outer bbox (x1, y1, x2, y2).
    Returns:
        float: Overlap ratio (intersection area / inner area).
    """
    inter = _bbox_intersection_area(inner, outer)
    denom = _bbox_area(inner)
    return 0.0 if denom == 0 else inter / denom

def build_table_image_map(blocks: List[ContentBlock], threshold: float = 0.1) -> Dict[int, List[int]]:
    """
    Build a mapping from table blocks to image blocks contained within them.
    Args:
        blocks (List[ContentBlock]): List of all content blocks.
        threshold (float): Minimum overlap ratio to consider an image as part of a table.
    Returns:
        Dict[int, List[int]]: Map where keys are table block indices and values are lists of image block indices.
    """
    table_indices = [i for i, b in enumerate(blocks) if b.type == "table"]
    image_indices = [i for i, b in enumerate(blocks) if b.type == "image"]
    table_to_images = {ti: [] for ti in table_indices}

    for ti in table_indices:
        tbox = blocks[ti].bbox
        for ii in image_indices:
            ratio = _overlap_ratio(blocks[ii].bbox, tbox)
            if ratio >= threshold:
                table_to_images[ti].append(ii)

        # Sort images by position (top-left: y, then x)
        table_to_images[ti].sort(key=lambda ii: (blocks[ii].bbox[1], blocks[ii].bbox[0]))
    return table_to_images

def mask_table_image(
    page_image: Image.Image,
    table_block: ContentBlock,
    image_blocks: List[ContentBlock],
    table_image: Image.Image,
) -> Image.Image:
    """
    Mask images inside a table block with their UIDs.
    
    Args:
        page_image: The full page image.
        table_block: The block object representing the table.
        image_blocks: List of image blocks inside the table (must have uid set).
        table_image: The cropped image of the table.
        
    Returns:
        The masked table image with [UID] tokens drawn on embedded images.
    """
    if not image_blocks:
        return table_image

    width, height = page_image.size
    x1_t, y1_t, x2_t, y2_t = table_block.bbox
    abs_x1_t = x1_t * width
    abs_y1_t = y1_t * height

    # Create a copy of the block image to modify
    masked_table_image = table_image.copy()
    draw = ImageDraw.Draw(masked_table_image)
    table_w, table_h = masked_table_image.size

    font_cache = {}

    def get_font_for_box(box_w, box_h, token_text):
        bucket_h = int(box_h // 16)
        key = (bucket_h, len(token_text))
        if key in font_cache:
            return font_cache[key]
        font, text_w, text_h = get_optimal_pil_font(
            token_text,
            box_w,
            box_h,
            fill_ratio=0.7,
            min_size=4,
            max_size=max(100, int(box_h * 0.7)),
        )
        font_cache[key] = (font, text_w, text_h)
        return font, text_w, text_h

    for img_block in image_blocks:
        if not img_block.uid:
            continue

        # Use the image block's uid as token: [UID]
        token_text = f"[{img_block.uid}]"

        # Normalized coordinates of the image block
        ix1, iy1, ix2, iy2 = img_block.bbox

        # Absolute pixel coords on the page
        abs_ix1 = ix1 * width
        abs_iy1 = iy1 * height
        abs_ix2 = ix2 * width
        abs_iy2 = iy2 * height

        # Relative pixel coords on the table block image
        base_x = int(abs_x1_t)
        base_y = int(abs_y1_t)

        rel_x1 = int(max(0, abs_ix1 - base_x))
        rel_y1 = int(max(0, abs_iy1 - base_y))
        rel_x2 = int(min(table_w, abs_ix2 - base_x))
        rel_y2 = int(min(table_h, abs_iy2 - base_y))

        if rel_x2 <= rel_x1 or rel_y2 <= rel_y1:
            continue

        box_w = rel_x2 - rel_x1
        box_h = rel_y2 - rel_y1

        # Mask on block image
        pad = 2
        image_mask_bbox = (
            max(0, rel_x1 - pad),
            max(0, rel_y1 - pad),
            min(table_w, rel_x2 + pad),
            min(table_h, rel_y2 + pad)
        )
        avg_color = get_average_color(masked_table_image, image_mask_bbox)

        draw.rectangle([rel_x1, rel_y1, rel_x2, rel_y2], fill=avg_color, outline=None)

        font, text_w, text_h = get_font_for_box(box_w, box_h, token_text)
        center_x = rel_x1 + box_w / 2
        center_y = rel_y1 + box_h / 2
        text_pos = (center_x - text_w / 2, center_y - text_h / 2)

        text_color = get_contrast_text_color(avg_color)
        draw.text(text_pos, token_text, fill=text_color, font=font)

    return masked_table_image


def post_process_table_content(content: str | None) -> str | None:
    """
    Replace [UID] tokens with <img src="*" data-uid="UID"> tags.
    
    Args:
        content: The table content string that may contain [UID] tokens.
        
    Returns:
        The processed content with tokens replaced by img tags.
    """
    if not content:
        return content
    
    # Match [XXXX] format token, where X is uppercase letter or digit
    pattern = r'\[([A-Z0-9]{4})\]'
    replacement = r'<img src="*" data-uid="\1">'
    return re.sub(pattern, replacement, content)