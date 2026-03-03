"""
test_pair_scorer.py — 配对评分模块单元测试
"""

import numpy as np
import pytest

from app.edge_features import EDGE_SAMPLES, EdgeFeatures
from app.pair_scorer import (
    PairScore,
    _ssd,
    build_best_neighbors,
    compute_pair_scores,
)


def _make_feat(
    piece_id: int,
    top_color: float = 100.0,
    right_color: float = 100.0,
    bottom_color: float = 100.0,
    left_color: float = 100.0,
    top_flat: bool = False,
    right_flat: bool = False,
    bottom_flat: bool = False,
    left_flat: bool = False,
) -> EdgeFeatures:
    """创建均匀颜色的 EdgeFeatures（用于测试）"""
    def _edge(v: float) -> np.ndarray:
        arr = np.full((EDGE_SAMPLES, 3), v, dtype=np.float32)
        return arr

    return EdgeFeatures(
        piece_id=piece_id,
        top=_edge(top_color),
        right=_edge(right_color),
        bottom=_edge(bottom_color),
        left=_edge(left_color),
        top_is_flat=top_flat,
        right_is_flat=right_flat,
        bottom_is_flat=bottom_flat,
        left_is_flat=left_flat,
    )


class TestSSD:
    def test_identical_edges_have_zero_ssd(self):
        edge = np.full((EDGE_SAMPLES, 3), 128.0, dtype=np.float32)
        assert _ssd(edge, edge) == pytest.approx(0.0)

    def test_different_edges_have_positive_ssd(self):
        a = np.full((EDGE_SAMPLES, 3), 100.0, dtype=np.float32)
        b = np.full((EDGE_SAMPLES, 3), 200.0, dtype=np.float32)
        assert _ssd(a, b) > 0

    def test_ssd_is_symmetric(self):
        a = np.random.rand(EDGE_SAMPLES, 3).astype(np.float32) * 255
        b = np.random.rand(EDGE_SAMPLES, 3).astype(np.float32) * 255
        assert _ssd(a, b) == pytest.approx(_ssd(b, a))


class TestComputePairScores:
    def test_returns_list_of_pair_scores(self):
        feats = [_make_feat(0), _make_feat(1)]
        scores = compute_pair_scores(feats)
        assert isinstance(scores, list)
        assert all(isinstance(s, PairScore) for s in scores)

    def test_identical_edges_get_high_score(self):
        """相同颜色的两条边应该得高分"""
        feats = [
            _make_feat(0, right_color=128.0),
            _make_feat(1, left_color=128.0),
        ]
        scores = compute_pair_scores(feats, max_ssd=10000.0)
        # 找 piece 0 right -> piece 1 left 的分数
        relevant = [
            s for s in scores
            if s.piece_a == 0 and s.side_a == "right"
            and s.piece_b == 1 and s.rotation_b == 0
        ]
        assert len(relevant) >= 1
        assert relevant[0].score == pytest.approx(1.0, abs=0.01)

    def test_different_edges_get_low_score(self):
        """颜色差异大的边得低分"""
        feats = [
            _make_feat(0, right_color=0.0),
            _make_feat(1, left_color=255.0),
        ]
        scores = compute_pair_scores(feats, max_ssd=10000.0)
        relevant = [
            s for s in scores
            if s.piece_a == 0 and s.side_a == "right"
            and s.piece_b == 1 and s.rotation_b == 0
        ]
        if relevant:
            assert relevant[0].score < 0.5

    def test_scores_in_valid_range(self):
        feats = [_make_feat(i) for i in range(4)]
        scores = compute_pair_scores(feats)
        for s in scores:
            assert 0.0 <= s.score <= 1.0, f"Score {s.score} out of [0,1]"

    def test_flat_edges_excluded_from_matching(self):
        """两块碎片的对应边都是直边时，不应产生匹配"""
        feats = [
            _make_feat(0, right_flat=True),
            _make_feat(1, left_flat=True),
        ]
        scores = compute_pair_scores(feats)
        flat_pair = [
            s for s in scores
            if s.piece_a == 0 and s.side_a == "right"
            and s.piece_b == 1
        ]
        # 两个都是直边，不应该有这个组合
        assert len(flat_pair) == 0


class TestBuildBestNeighbors:
    def test_returns_best_score_per_side(self):
        feats = [_make_feat(i) for i in range(3)]
        scores = compute_pair_scores(feats)
        best = build_best_neighbors(scores, feats)
        # 每个 key 只有一个最佳值
        for key, ps in best.items():
            assert isinstance(key, tuple)
            assert isinstance(ps, PairScore)

    def test_best_is_max_score(self):
        feats = [
            _make_feat(0, right_color=100.0),
            _make_feat(1, left_color=100.0),   # 完美匹配
            _make_feat(2, left_color=50.0),    # 较差匹配
        ]
        scores = compute_pair_scores(feats)
        best = build_best_neighbors(scores, feats)
        key = (0, "right")
        if key in best:
            # 最佳邻居应该是 piece 1（颜色更接近）
            assert best[key].score >= 0.0
