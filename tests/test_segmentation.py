"""
test_segmentation.py — 碎片分割模块单元测试
"""

import numpy as np
import pytest

from app.segmentation import PieceInfo, segment_pieces


def _make_white_bg_with_squares(
    img_size: int = 400,
    piece_size: int = 60,
    positions: list[tuple[int, int]] = None,
) -> np.ndarray:
    """
    生成白色背景上若干深色方块的测试图像，模拟散乱碎片。
    positions: 方块左上角 (x, y) 列表
    """
    img = np.full((img_size, img_size, 3), 240, dtype=np.uint8)
    if positions is None:
        positions = [(30, 30), (150, 50), (80, 200), (250, 250)]
    for x, y in positions:
        x2 = min(x + piece_size, img_size)
        y2 = min(y + piece_size, img_size)
        img[y:y2, x:x2] = [50, 80, 120]  # 深蓝色方块
    return img


class TestSegmentPieces:
    def test_detects_expected_piece_count(self):
        positions = [(20, 20), (160, 20), (20, 160), (160, 160)]
        img = _make_white_bg_with_squares(positions=positions, piece_size=80)
        pieces = segment_pieces(img, min_area=500)
        # 应检测到 4 块（允许轻微误差）
        assert 2 <= len(pieces) <= 6, f"Expected ~4 pieces, got {len(pieces)}"

    def test_piece_ids_are_unique(self):
        img = _make_white_bg_with_squares()
        pieces = segment_pieces(img, min_area=200)
        ids = [p.id for p in pieces]
        assert len(ids) == len(set(ids)), "Piece IDs should be unique"

    def test_piece_has_required_fields(self):
        img = _make_white_bg_with_squares(positions=[(50, 50)], piece_size=100)
        pieces = segment_pieces(img, min_area=200)
        assert len(pieces) >= 1
        p = pieces[0]
        assert isinstance(p, PieceInfo)
        assert p.mask is not None
        assert len(p.bbox) == 4   # (x, y, w, h)
        assert p.contour is not None
        assert len(p.centroid) == 2
        assert p.area > 0

    def test_empty_image_returns_no_pieces(self):
        # 纯白图像，不包含任何碎片
        img = np.full((200, 200, 3), 250, dtype=np.uint8)
        pieces = segment_pieces(img, min_area=500)
        assert len(pieces) == 0

    def test_pieces_sorted_by_area_descending(self):
        # 大小不同的方块
        img = np.full((500, 500, 3), 240, dtype=np.uint8)
        img[20:120, 20:120] = [50, 50, 50]   # 100×100
        img[200:250, 200:250] = [50, 50, 50]  # 50×50
        pieces = segment_pieces(img, min_area=200)
        if len(pieces) >= 2:
            assert pieces[0].area >= pieces[1].area

    def test_min_area_filters_small_regions(self):
        img = np.full((300, 300, 3), 240, dtype=np.uint8)
        # 大碎片
        img[50:150, 50:150] = [40, 40, 40]
        # 极小噪点
        img[200:205, 200:205] = [40, 40, 40]

        pieces_strict = segment_pieces(img, min_area=500)
        pieces_loose = segment_pieces(img, min_area=1)

        assert len(pieces_strict) <= len(pieces_loose)

    def test_centroid_within_image_bounds(self):
        img = _make_white_bg_with_squares()
        pieces = segment_pieces(img, min_area=200)
        h, w = img.shape[:2]
        for p in pieces:
            cx, cy = p.centroid
            assert 0 <= cx < w, f"centroid x={cx} out of bounds"
            assert 0 <= cy < h, f"centroid y={cy} out of bounds"
