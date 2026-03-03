"""
test_layout_solver.py — 布局求解模块单元测试
"""

import numpy as np
import pytest

from app.edge_features import EDGE_SAMPLES, EdgeFeatures
from app.layout_solver import (
    Placement,
    infer_grid_size,
    solve_layout,
)
from app.pair_scorer import compute_pair_scores


def _make_feat(
    piece_id: int,
    top_flat: bool = False,
    right_flat: bool = False,
    bottom_flat: bool = False,
    left_flat: bool = False,
    color: float = 128.0,
) -> EdgeFeatures:
    def _edge(v: float = color) -> np.ndarray:
        return np.full((EDGE_SAMPLES, 3), v, dtype=np.float32)

    return EdgeFeatures(
        piece_id=piece_id,
        top=_edge(),
        right=_edge(),
        bottom=_edge(),
        left=_edge(),
        top_is_flat=top_flat,
        right_is_flat=right_flat,
        bottom_is_flat=bottom_flat,
        left_is_flat=left_flat,
    )


def _make_2x2_grid_features() -> list[EdgeFeatures]:
    """
    构造 2×2 拼图的边缘特征（理想情况）：
      [0] 角块 (0,0): top/left 直边
      [1] 角块 (0,1): top/right 直边
      [2] 角块 (1,0): bottom/left 直边
      [3] 角块 (1,1): bottom/right 直边
    """
    return [
        _make_feat(0, top_flat=True, left_flat=True),
        _make_feat(1, top_flat=True, right_flat=True),
        _make_feat(2, bottom_flat=True, left_flat=True),
        _make_feat(3, bottom_flat=True, right_flat=True),
    ]


class TestInferGridSize:
    def test_single_placement(self):
        placements = [Placement(piece_id=0, row=0, col=0, rotation=0)]
        assert infer_grid_size(placements) == (1, 1)

    def test_2x3_grid(self):
        placements = [
            Placement(piece_id=i, row=r, col=c, rotation=0)
            for i, (r, c) in enumerate([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)])
        ]
        rows, cols = infer_grid_size(placements)
        assert rows == 2
        assert cols == 3

    def test_empty_returns_1x1(self):
        assert infer_grid_size([]) == (1, 1)


class TestSolveLayout:
    def test_returns_placements_and_unmatched(self):
        feats = _make_2x2_grid_features()
        scores = compute_pair_scores(feats)
        placements, unmatched = solve_layout(feats, scores)
        assert isinstance(placements, list)
        assert isinstance(unmatched, list)

    def test_all_pieces_placed_or_unmatched(self):
        feats = _make_2x2_grid_features()
        scores = compute_pair_scores(feats)
        placements, unmatched = solve_layout(feats, scores)
        placed_ids = {p.piece_id for p in placements}
        all_ids = {f.piece_id for f in feats}
        assert placed_ids | set(unmatched) == all_ids

    def test_no_duplicate_positions(self):
        """同一个格子不能放两块碎片"""
        feats = _make_2x2_grid_features()
        scores = compute_pair_scores(feats)
        placements, _ = solve_layout(feats, scores)
        positions = [(p.row, p.col) for p in placements]
        assert len(positions) == len(set(positions)), "Duplicate grid positions found"

    def test_no_duplicate_piece_ids_in_placements(self):
        """每块碎片只能放一次"""
        feats = _make_2x2_grid_features()
        scores = compute_pair_scores(feats)
        placements, _ = solve_layout(feats, scores)
        ids = [p.piece_id for p in placements]
        assert len(ids) == len(set(ids)), "Duplicate piece IDs in placements"

    def test_placement_rotation_valid(self):
        """旋转值只能是 0/1/2/3"""
        feats = _make_2x2_grid_features()
        scores = compute_pair_scores(feats)
        placements, _ = solve_layout(feats, scores)
        for p in placements:
            assert p.rotation in (0, 1, 2, 3), f"Invalid rotation: {p.rotation}"

    def test_confidence_in_range(self):
        feats = _make_2x2_grid_features()
        scores = compute_pair_scores(feats)
        placements, _ = solve_layout(feats, scores)
        for p in placements:
            assert 0.0 <= p.confidence <= 1.0

    def test_empty_features_returns_empty(self):
        placements, unmatched = solve_layout([], [])
        assert placements == []
        assert unmatched == []
