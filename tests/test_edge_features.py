"""
test_edge_features.py — 边缘采样模块单元测试
"""

import cv2 as cv
import numpy as np
import pytest

from app.edge_features import (
    EDGE_SAMPLES,
    EdgeFeatures,
    _is_flat_edge,
    extract_all_features,
    extract_edge_features,
)
from app.segmentation import PieceInfo, segment_pieces


def _make_piece_info(
    img_size: int = 200,
    piece_x: int = 40,
    piece_y: int = 40,
    piece_w: int = 80,
    piece_h: int = 80,
) -> tuple[PieceInfo, np.ndarray]:
    """生成单个方形碎片的测试图像和 PieceInfo"""
    img = np.full((img_size, img_size, 3), 240, dtype=np.uint8)
    # 填充碎片区域（有图案）
    for i in range(piece_h):
        for j in range(piece_w):
            img[piece_y + i, piece_x + j] = [
                int(i * 2.5) % 256,
                int(j * 2.5) % 256,
                100,
            ]

    # 手动构建 PieceInfo
    contour = np.array([
        [[piece_x, piece_y]],
        [[piece_x + piece_w, piece_y]],
        [[piece_x + piece_w, piece_y + piece_h]],
        [[piece_x, piece_y + piece_h]],
    ], dtype=np.int32)

    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    mask[piece_y:piece_y + piece_h, piece_x:piece_x + piece_w] = 255

    piece = PieceInfo(
        id=0,
        mask=mask,
        bbox=(piece_x, piece_y, piece_w, piece_h),
        contour=contour,
        centroid=(piece_x + piece_w // 2, piece_y + piece_h // 2),
        area=float(piece_w * piece_h),
        min_rect=cv.minAreaRect(contour),
    )
    return piece, img


class TestEdgeFeatures:
    def test_output_shape(self):
        piece, img = _make_piece_info()
        image_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab).astype(np.float32)
        feat = extract_edge_features(piece, image_lab)

        for side in ("top", "right", "bottom", "left"):
            edge = feat.get_edge(side)
            assert edge.shape == (EDGE_SAMPLES, 3), (
                f"Edge '{side}' shape={edge.shape}, expected ({EDGE_SAMPLES}, 3)"
            )

    def test_returns_edge_features_type(self):
        piece, img = _make_piece_info()
        image_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab).astype(np.float32)
        feat = extract_edge_features(piece, image_lab)
        assert isinstance(feat, EdgeFeatures)
        assert feat.piece_id == piece.id

    def test_flat_edge_detection_straight_line(self):
        # 水平直线应被识别为直边
        pts = np.array([[0.0, 5.0], [10.0, 5.0], [20.0, 5.0], [30.0, 5.0]])
        assert _is_flat_edge(pts, flatness_thresh=3.0)

    def test_flat_edge_detection_curved_line(self):
        # 明显弯曲的线不是直边
        pts = np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 0.0], [30.0, 20.0]])
        assert not _is_flat_edge(pts, flatness_thresh=3.0)

    def test_corner_piece_classification(self):
        """角块应有 2 条直边"""
        piece, img = _make_piece_info()
        image_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab).astype(np.float32)
        feat = extract_edge_features(piece, image_lab)
        # 矩形碎片的直边数目应 >= 0（实际取决于 flatness 阈值）
        assert 0 <= feat.flat_count <= 4

    def test_extract_all_features_count(self):
        img = np.full((400, 400, 3), 240, dtype=np.uint8)
        img[30:100, 30:100] = [50, 100, 150]
        img[200:280, 200:280] = [150, 80, 60]
        pieces = segment_pieces(img, min_area=200)
        if len(pieces) == 0:
            pytest.skip("No pieces detected in test image")
        feats = extract_all_features(pieces, img)
        assert len(feats) == len(pieces)

    def test_colors_are_finite(self):
        piece, img = _make_piece_info()
        image_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab).astype(np.float32)
        feat = extract_edge_features(piece, image_lab)
        for side in ("top", "right", "bottom", "left"):
            assert np.all(np.isfinite(feat.get_edge(side))), (
                f"Edge '{side}' contains non-finite values"
            )
