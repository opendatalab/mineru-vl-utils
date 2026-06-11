from typing import Literal

from .vlm_client.base_client import ScoredOutput


class BlockType:
    TEXT = "text"  # 文本
    TITLE = "title"  # 段落标题
    TABLE = "table"  # 表格
    EQUATION = "equation"  # 公式(独立公式)
    FORMULA_NUMBER = "formula_number"  # 公式编号
    CODE = "code"  # 代码
    ALGORITHM = "algorithm"  # 算法/伪代码
    ASIDE_TEXT = "aside_text"  # 侧栏文本(装订线等)
    REF_TEXT = "ref_text"  # 参考文献(一条)
    INDEX = "index"  # 目录/索引文本块
    PHONETIC = "phonetic"  # 注音符号
    LIST_ITEM = "list_item"  # 列表项(无序/有序列表)
    # captions
    TABLE_CAPTION = "table_caption"  # 表格标题
    IMAGE_CAPTION = "image_caption"  # 图像标题
    CODE_CAPTION = "code_caption"  # 代码标题
    TABLE_FOOTNOTE = "table_footnote"  # 表格脚注
    IMAGE_FOOTNOTE = "image_footnote"  # 图像脚注
    # paratexts
    HEADER = "header"  # 页眉
    FOOTER = "footer"  # 页脚
    PAGE_NUMBER = "page_number"  # 页码
    PAGE_FOOTNOTE = "page_footnote"  # 脚注
    # images
    IMAGE = "image"  # 图像
    CHART = "chart"
    # containers
    LIST = "list"  # 列表块(无序/有序列表)
    IMAGE_BLOCK = "image_block"  # 图像块(多图)
    EQUATION_BLOCK = "equation_block"  # 公式块(多行公式)
    # unknown
    UNKNOWN = "unknown"  # 未知块


BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.TABLE,
    BlockType.EQUATION,
    BlockType.FORMULA_NUMBER,
    BlockType.CODE,
    BlockType.ALGORITHM,
    BlockType.ASIDE_TEXT,
    BlockType.REF_TEXT,
    BlockType.INDEX,
    BlockType.PHONETIC,
    BlockType.LIST_ITEM,
    # captions
    BlockType.TABLE_CAPTION,
    BlockType.IMAGE_CAPTION,
    BlockType.CODE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.IMAGE_FOOTNOTE,
    # paratexts
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.PAGE_FOOTNOTE,
    # images
    BlockType.IMAGE,
    BlockType.CHART,
    # containers
    BlockType.LIST,
    BlockType.IMAGE_BLOCK,
    BlockType.EQUATION_BLOCK,
    # unknown
    BlockType.UNKNOWN,
}

ANGLE_OPTIONS = {
    None,
    0,
    90,
    180,
    270,
}


class ExtractResult(list["ContentBlock"]):
    """
    list[ContentBlock] subclass returned by two_step_extract() and related methods.
    Backward-compatible: all existing list[ContentBlock] usage works unchanged.

    When scored=True is passed to the extraction method:
    - layout_scored: ScoredOutput for the layout detection step (whole-page score)
    - blocks[i].scored: optional ScoredOutput for each content block's extraction step
    """

    layout_scored: ScoredOutput | None

    def __init__(self, blocks=(), layout_scored: ScoredOutput | None = None):
        super().__init__(blocks)
        self.layout_scored = layout_scored


class ExtractStr(str):
    """
    str subclass returned by content_extract() and related methods when scored=True.
    Backward-compatible: all existing str usage works unchanged.

    When scored=True is passed to the extraction method:
    - scored: ScoredOutput for the content extraction step
    """

    scored: ScoredOutput | None

    def __new__(cls, value: str, scored: ScoredOutput | None = None):
        instance = super().__new__(cls, value)
        instance.scored = scored
        return instance


class ContentBlock(dict):
    def __init__(
        self,
        type: str,
        bbox: list[float],
        angle: Literal[None, 0, 90, 180, 270] = None,
        content: str | None = None,
        merge_prev: bool = False,
    ):
        """
        Initialize a layout block.
        Args:
            type (str): Type of the block (e.g., 'text', 'image', 'table').
            bbox (list[float]): Bounding box coordinates [xmin, ymin, xmax, ymax].
            angle (int or None): Rotation angle of the block. Must be one of {None, 0, 90, 180, 270}.
            content (str or None): The content of the block (if exists).
            merge_prev (bool): Whether the current text block should merge with the previous block.
        """
        super().__init__()

        assert type in BLOCK_TYPES, f"Unknown type: {type}"
        assert isinstance(bbox, list) and len(bbox) == 4, "Bounding box must be a list of four coordinates"
        assert all(isinstance(coord, (int, float)) for coord in bbox), "Bounding box coordinates must be numbers"
        assert all(0 <= coord <= 1 for coord in bbox), "Bounding box coordinates must be in the range [0, 1]"
        assert bbox[0] < bbox[2], "Bounding box x1 must be less than x2"
        assert bbox[1] < bbox[3], "Bounding box y1 must be less than y2"
        assert angle in ANGLE_OPTIONS, f"Invalid angle: {angle}. Must be one of {ANGLE_OPTIONS}"
        assert content is None or isinstance(content, str), "Content must be a string or None"
        assert merge_prev.__class__ is bool, f"Invalid merge_prev: {merge_prev}. Must be bool"
        if type != BlockType.TEXT:
            assert merge_prev is False, "merge_prev can only be set for text blocks"
        self["type"] = type
        self["bbox"] = bbox
        self["angle"] = angle
        self["content"] = content
        if type == BlockType.TEXT:
            self["merge_prev"] = merge_prev

    @property
    def type(self) -> str:
        return self["type"]

    @type.setter
    def type(self, value: str):
        assert value in BLOCK_TYPES, f"Unknown type: {value}"
        merge_prev = self.get("merge_prev", False)
        self["type"] = value
        if value == BlockType.TEXT:
            self["merge_prev"] = merge_prev if merge_prev.__class__ is bool else False
        else:
            self.pop("merge_prev", None)

    @property
    def bbox(self) -> list[float]:
        return self["bbox"]

    @bbox.setter
    def bbox(self, value: list[float]):
        assert isinstance(value, list) and len(value) == 4, "Bounding box must be a list of four coordinates"
        assert all(isinstance(coord, (int, float)) for coord in value), "Bounding box coordinates must be numbers"
        assert all(0 <= coord <= 1 for coord in value), "Bounding box coordinates must be in the range [0, 1]"
        assert value[0] < value[2], "Bounding box x1 must be less than x2"
        assert value[1] < value[3], "Bounding box y1 must be less than y2"
        self["bbox"] = value

    @property
    def angle(self) -> Literal[None, 0, 90, 180, 270]:
        return self["angle"]

    @angle.setter
    def angle(self, value: Literal[None, 0, 90, 180, 270]):
        assert value in ANGLE_OPTIONS, f"Invalid angle: {value}. Must be one of {ANGLE_OPTIONS}"
        self["angle"] = value

    @property
    def content(self) -> str | None:
        return self["content"]

    @content.setter
    def content(self, value: str | None):
        assert value is None or isinstance(value, str), "Content must be a string or None"
        self["content"] = value

    @property
    def merge_prev(self) -> bool:
        assert self.type == BlockType.TEXT, "merge_prev is only available for text blocks"
        return self["merge_prev"]

    @merge_prev.setter
    def merge_prev(self, value: bool):
        assert self.type == BlockType.TEXT, "merge_prev is only available for text blocks"
        assert value.__class__ is bool, f"Invalid merge_prev: {value}. Must be bool"
        self["merge_prev"] = value

    @property
    def scored(self) -> ScoredOutput | None:
        return self.get("scored")

    @scored.setter
    def scored(self, value: ScoredOutput | None):
        if value is None:
            self.pop("scored", None)
        else:
            self["scored"] = value
