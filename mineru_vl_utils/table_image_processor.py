import os
import hashlib
import random
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageStat
from .structs import ContentBlock

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
    Calculate average color from 8 points around the box (corners and side midpoints), with padding.
    Args:
        image (Image.Image): The source image.
        box (tuple): The region (left, upper, right, lower).
    Returns:
        Tuple[int, int, int]: Average RGB color.
    """
    try:
        left, upper, right, lower = map(int, box)
        width, height = image.size
        pad = 2

        mid_x = (left + right) // 2
        mid_y = (upper + lower) // 2

        points = [
            (left - pad, upper - pad),      # Top-left
            (mid_x, upper - pad),           # Top-mid
            (right + pad, upper - pad),     # Top-right
            (right + pad, mid_y),           # Right-mid
            (right + pad, lower + pad),     # Bottom-right
            (mid_x, lower + pad),           # Bottom-mid
            (left - pad, lower + pad),      # Bottom-left
            (left - pad, mid_y),            # Left-mid
        ]

        valid_pixels = []
        for px, py in points:

            px = max(0, min(px, width - 1))
            py = max(0, min(py, height - 1))

            pixel = image.getpixel((px, py))

            # to RGB.
            if isinstance(pixel, int):
                valid_pixels.append((pixel, pixel, pixel))
            elif len(pixel) >= 3:
                valid_pixels.append(pixel[:3])

        if not valid_pixels:
            return (255, 255, 255)

        # Calculate average
        r_sum = sum(p[0] for p in valid_pixels)
        g_sum = sum(p[1] for p in valid_pixels)
        b_sum = sum(p[2] for p in valid_pixels)
        count = len(valid_pixels)

        return (int(r_sum / count), int(g_sum / count), int(b_sum / count))

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

def mask_and_crop_table_image(
    page_image: Image.Image,
    table_block: ContentBlock,
    image_blocks: List[ContentBlock],
    table_image: Image.Image,
    token_format: str = "[[G{idx:02d}]]",
    output_root: str | None = None,
    rel_base: str | None = None,
    page_idx: int | None = None,
    table_idx: int | None = None,
) -> Tuple[Image.Image, Dict[str, str]]:
    """
    Process a table image by masking embedded images and optionally saving crops.
    Args:
        page_image (Image.Image): The full page image.
        table_block (ContentBlock): The block object representing the table.
        image_blocks (List[ContentBlock]): List of image blocks inside the table.
        table_image (Image.Image): The cropped image of the table.
        token_format (str): Format string for replacement tokens.
        output_root (str | None): Root directory to save cropped images.
        rel_base (str | None): Base path for relative URLs.
        page_idx (int | None): Page index for filename generation.
    Returns:
        Tuple[Image.Image, Dict[str, str]]: The masked table image and a dictionary mapping tokens to image paths/URLs.
    """
    """
    Masks images inside a table block and optionally saves crops.
    Returns the masked block image and a map of tokens to URLs.
    """
    width, height = page_image.size

    # Calculate absolute pixel coordinates of the table block on the page
    # to align with how the crop was originally created.
    x1_t, y1_t, x2_t, y2_t = table_block.bbox
    abs_x1_t = x1_t * width
    abs_y1_t = y1_t * height

    # Create a copy of the block image to modify
    masked_table_image = table_image.copy()
    draw = ImageDraw.Draw(masked_table_image)

    # Use the actual dimensions of the cropped table image
    table_w, table_h = masked_table_image.size

    token_map = {}

    images_dir = None
    if output_root:
        images_dir = output_root  
        os.makedirs(images_dir, exist_ok=True)
        if rel_base is None:
            rel_base = os.path.dirname(images_dir)

    font_cache = {}

    def get_font_for_box(box_w, box_h, token_text):
        bucket_h = int(box_h // 16)
        key = (bucket_h, len(token_text))
        if key in font_cache:
            font, text_w, text_h = font_cache[key]
            if text_w <= box_w and text_h <= box_h:
                return font, text_w, text_h
        font, text_w, text_h = get_optimal_pil_font(
            token_text,
            box_w,
            box_h,
            fill_ratio=0.7,
            min_size=4,
            max_size=max(100, int(box_h * 0.7)),
        )

        if text_w > box_w or text_h > box_h:
             pass

        font_cache[key] = (font, text_w, text_h)
        return font, text_w, text_h

    letters = 'ACDGHKTWXYZ'
    numbers = '2345678'
    
    valid_tokens = []
    for l1 in letters:
        for l2 in letters:
            for n1 in numbers:
                for n2 in numbers:
                    valid_tokens.append(f"{l1}{l2}{n1}{n2}")

    random.shuffle(valid_tokens)
    for i, img_block in enumerate(image_blocks):
        if i < len(valid_tokens):
            token_code = valid_tokens[i]
        else:
            # Fallback strategy if we run out of unique 4-char tokens (unlikely)
            token_code = f"IMG{i}"

        token_text = token_format.format(idx=token_code)

        # Normalized coordinates of the image block
        ix1, iy1, ix2, iy2 = img_block.bbox

        # Absolute pixel coords on the page
        abs_ix1 = ix1 * width
        abs_iy1 = iy1 * height
        abs_ix2 = ix2 * width
        abs_iy2 = iy2 * height

        # Relative pixel coords on the table block image
        # We use round() or int() consistently with how PIL handles crops
        # FIX: Use integer base coordinates to match crop behavior
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

        crop_img = page_image.crop((int(abs_ix1), int(abs_iy1), int(abs_ix2), int(abs_iy2)))

        coords = f"{int(abs_ix1)}_{int(abs_iy1)}_{int(abs_ix2)}_{int(abs_iy2)}"
        page_part = f"p{page_idx}" if page_idx is not None else "pNone"
        table_part = f"t{table_idx}" if table_idx is not None else "tNone"
        hash_input = f"table_image_{page_part}_{table_part}_{coords}"
        filename = f"{hashlib.sha256(hash_input.encode('utf-8')).hexdigest()}.jpg"

        save_path = os.path.join(images_dir, filename)
        crop_img.save(save_path, format="JPEG")

        if rel_base:
            url = os.path.relpath(save_path, rel_base)
            if not url.startswith((".", "/")):
                 url = "./" + url
        else:
            url = os.path.basename(save_path)

        token_map[token_text] = url
        # print(f"TOKENMAP:{token_map}")

        # 2. Mask on block image
        image_mask_bbox = (rel_x1, rel_y1, rel_x2, rel_y2)
        avg_color = get_average_color(masked_table_image, image_mask_bbox)

        draw.rectangle([rel_x1, rel_y1, rel_x2, rel_y2], fill=avg_color, outline=None)

        font, text_w, text_h = get_font_for_box(box_w, box_h, token_text)
        center_x = rel_x1 + box_w / 2
        center_y = rel_y1 + box_h / 2
        text_pos = (center_x - text_w / 2, center_y - text_h / 2)

        text_color = get_contrast_text_color(avg_color)
        if text_w <= rel_x2 - rel_x1 and text_h <= rel_y2 - rel_y1:
            draw.text(text_pos, token_text, fill=text_color, font=font)

    return masked_table_image, token_map
