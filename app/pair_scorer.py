"""
pair_scorer.py — 计算碎片对之间的边缘兼容性分数。

两块相邻碎片的匹配分数基于：
  - 颜色 SSD（Sum of Squared Differences）：颜色越接近分数越低（越好）
  - 归一化后取反，得到 [0, 1] 的相似度分数（越高越好）

相邻关系：piece_i 的 right 边与 piece_j 的 left 边相邻
         piece_i 的 bottom 边与 piece_j 的 top 边相邻
（考虑 piece_j 的四种旋转）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edge_features import EdgeFeatures

# 边对：(piece_a 的边, piece_b 的边) 表示 a 在 b 的某个方向
OPPOSITE = {"right": "left", "bottom": "top", "left": "right", "top": "bottom"}
ROTATIONS = [0, 1, 2, 3]   # 0=0°, 1=90°, 2=180°, 3=270°

# 旋转后的边名称映射：key=旋转次数（90°为单位），value=原始边→旋转后边
_ROTATION_MAP = [
    {"top": "top",    "right": "right",  "bottom": "bottom", "left": "left"},    # 0°
    {"top": "left",   "right": "top",    "bottom": "right",  "left": "bottom"},  # 90° CW
    {"top": "bottom", "right": "left",   "bottom": "top",    "left": "right"},   # 180°
    {"top": "right",  "right": "bottom", "bottom": "left",   "left": "top"},     # 270° CW
]


@dataclass
class PairScore:
    piece_a: int
    piece_b: int
    side_a: str          # piece_a 朝向 piece_b 的边（如 "right"）
    side_b: str          # piece_b 朝向 piece_a 的边（如 "left"）
    rotation_b: int      # piece_b 的旋转（0/1/2/3，以 90° 为单位）
    score: float         # 相似度分数 [0, 1]，越高越好


def _ssd(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    """计算两条边颜色序列的 SSD（均方差）"""
    diff = edge_a.astype(float) - edge_b.astype(float)
    return float(np.mean(diff ** 2))


def _rotated_edge(feat: EdgeFeatures, side: str, rotation: int) -> np.ndarray:
    """获取碎片旋转后某条边的颜色序列"""
    # 找到旋转后 side 对应的原始边名
    original_side = None
    rot_map = _ROTATION_MAP[rotation]
    for orig, rotated in rot_map.items():
        if rotated == side:
            original_side = orig
            break
    return feat.get_edge(original_side)


def _rotated_is_flat(feat: EdgeFeatures, side: str, rotation: int) -> bool:
    """获取碎片旋转后某条边是否为直边"""
    rot_map = _ROTATION_MAP[rotation]
    for orig, rotated in rot_map.items():
        if rotated == side:
            return feat.is_flat(orig)
    return False


def compute_pair_scores(
    features: list[EdgeFeatures],
    max_ssd: float = 10000.0,
) -> list[PairScore]:
    """
    计算所有碎片对的兼容性分数。

    只计算有意义的组合：
      - piece_a 的 right/bottom 边 vs piece_b 的对应对面边（含旋转）
      - 跳过两块碎片同侧均为直边的组合（外缘不会相邻）

    Args:
        features: 所有碎片的边缘特征列表
        max_ssd: SSD 归一化上限，超过此值相似度视为 0

    Returns:
        PairScore 列表（未排序）
    """
    n = len(features)
    scores: list[PairScore] = []

    feat_by_id = {f.piece_id: f for f in features}

    for i in range(n):
        fa = features[i]
        for j in range(n):
            if i == j:
                continue
            fb = features[j]
            # 只枚举 a 的 right/bottom，避免重复
            for side_a in ("right", "bottom"):
                side_b = OPPOSITE[side_a]
                for rot in ROTATIONS:
                    # 跳过：a 的这条边是直边，b 旋转后对应边也是直边（两外缘不相邻）
                    if fa.is_flat(side_a) and _rotated_is_flat(fb, side_b, rot):
                        continue
                    # 跳过：a 是直边但 b 不是（外缘只能与外缘配对，内边不能接外缘）
                    # 实际上直边可以匹配直边，但不能匹配非直边
                    if fa.is_flat(side_a) != _rotated_is_flat(fb, side_b, rot):
                        continue

                    edge_a = fa.get_edge(side_a)
                    edge_b = _rotated_edge(fb, side_b, rot)

                    ssd = _ssd(edge_a, edge_b)
                    # 归一化到 [0, 1]
                    similarity = max(0.0, 1.0 - ssd / max_ssd)

                    scores.append(PairScore(
                        piece_a=fa.piece_id,
                        piece_b=fb.piece_id,
                        side_a=side_a,
                        side_b=side_b,
                        rotation_b=rot,
                        score=similarity,
                    ))

    return scores


def build_best_neighbors(
    scores: list[PairScore],
    features: list[EdgeFeatures],
) -> dict[tuple[int, str], PairScore]:
    """
    为每个 (piece_id, side) 找到得分最高的邻居。

    Returns:
        key = (piece_id, side)，value = 最佳 PairScore
    """
    best: dict[tuple[int, str], PairScore] = {}
    for ps in scores:
        key = (ps.piece_a, ps.side_a)
        if key not in best or ps.score > best[key].score:
            best[key] = ps
    return best
