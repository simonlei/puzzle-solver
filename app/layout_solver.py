"""
layout_solver.py — 贪心布局求解器。

算法：
  1. 识别角块（2 条直边）、边块（1 条直边）、内部块（0 条直边）
  2. 从一个角块出发，以右/下方向构建拼图网格
  3. 每次从当前空位的候选碎片中选得分最高的放置
  4. 无法确定的碎片标记为 unmatched

输出：每块碎片的 (row, col, rotation) 赋值
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .edge_features import EdgeFeatures
from .pair_scorer import OPPOSITE, PairScore, _ROTATION_MAP, build_best_neighbors


@dataclass
class Placement:
    piece_id: int
    row: int
    col: int
    rotation: int     # 0/1/2/3（×90°顺时针）
    confidence: float = 1.0


def _rotated_is_flat(feat: EdgeFeatures, side: str, rotation: int) -> bool:
    rot_map = _ROTATION_MAP[rotation]
    for orig, rotated in rot_map.items():
        if rotated == side:
            return feat.is_flat(orig)
    return False


def solve_layout(
    features: list[EdgeFeatures],
    scores: list[PairScore],
) -> tuple[list[Placement], list[int]]:
    """
    用贪心策略求解拼图布局。

    Args:
        features: 所有碎片的边缘特征
        scores: compute_pair_scores 的输出

    Returns:
        (placements, unmatched_ids)
        - placements: 成功定位的碎片列表
        - unmatched_ids: 无法确定位置的碎片 ID 列表
    """
    if not features:
        return [], []

    feat_by_id = {f.piece_id: f for f in features}
    n = len(features)

    # 按 (piece_a, side_a) -> 最佳邻居构建查询表
    best_neighbors = build_best_neighbors(scores, features)

    # 另建按 (piece_a, side_a) -> Top-K 候选列表（用于冲突时回退）
    candidates: dict[tuple[int, str], list[PairScore]] = {}
    for ps in scores:
        key = (ps.piece_a, ps.side_a)
        candidates.setdefault(key, []).append(ps)
    for key in candidates:
        candidates[key].sort(key=lambda x: x.score, reverse=True)

    # ---- 估算拼图网格尺寸 ----------------------------------------
    # 假设碎片面积近似相等，网格接近正方形
    grid_size = math.isqrt(n)
    n_cols = grid_size
    n_rows = math.ceil(n / n_cols)

    # ---- 找角块作为起点 ------------------------------------------
    corners = [f for f in features if f.is_corner]
    borders = [f for f in features if f.is_border]
    interiors = [f for f in features if f.is_interior]

    if not corners:
        # 兜底：没有识别到角块，选面积最大的碎片作为起点
        corners = [features[0]] if features else []

    # 选第一个角块，让其 top/left 为直边（放在 (0,0)）
    start = _find_top_left_corner(corners[0])

    # ---- 贪心填充网格 --------------------------------------------
    grid: dict[tuple[int, int], int] = {}      # (row, col) -> piece_id
    rotations: dict[int, int] = {}             # piece_id -> rotation
    placed: set[int] = set()
    placement_list: list[Placement] = []
    confidence_map: dict[int, float] = {}

    # 放置起点
    grid[(0, 0)] = start.piece_id
    rotations[start.piece_id] = _corner_rotation(start)
    placed.add(start.piece_id)
    placement_list.append(Placement(
        piece_id=start.piece_id, row=0, col=0,
        rotation=rotations[start.piece_id], confidence=1.0,
    ))

    # BFS 扩展：从已放置的碎片向右/向下延伸
    queue = [(0, 0)]
    visited_cells: set[tuple[int, int]] = {(0, 0)}

    while queue:
        row, col = queue.pop(0)
        current_id = grid.get((row, col))
        if current_id is None:
            continue
        current_rot = rotations.get(current_id, 0)

        for dr, dc, side_a in [(0, 1, "right"), (1, 0, "bottom")]:
            nr, nc = row + dr, col + dc
            if nr >= n_rows or nc >= n_cols:
                continue
            if (nr, nc) in grid:
                continue

            # 从当前碎片的这条边找最佳邻居
            best = best_neighbors.get((current_id, side_a))
            if best is None:
                continue

            neighbor_id = best.piece_b
            neighbor_rot = best.rotation_b

            if neighbor_id in placed:
                # 已放置，尝试候选列表中下一个
                alt = _find_unplaced(candidates.get((current_id, side_a), []), placed)
                if alt is None:
                    continue
                neighbor_id = alt.piece_b
                neighbor_rot = alt.rotation_b

            grid[(nr, nc)] = neighbor_id
            rotations[neighbor_id] = neighbor_rot
            placed.add(neighbor_id)
            confidence_map[neighbor_id] = best.score
            placement_list.append(Placement(
                piece_id=neighbor_id, row=nr, col=nc,
                rotation=neighbor_rot, confidence=best.score,
            ))

            if (nr, nc) not in visited_cells:
                visited_cells.add((nr, nc))
                queue.append((nr, nc))

    # ---- 收集 unmatched ------------------------------------------
    unmatched = [f.piece_id for f in features if f.piece_id not in placed]

    # 为 unmatched 碎片分配剩余空格（无置信度）
    empty_cells = [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if (r, c) not in grid
    ]
    for pid, cell in zip(unmatched[:], empty_cells):
        r, c = cell
        placement_list.append(Placement(
            piece_id=pid, row=r, col=c, rotation=0, confidence=0.0,
        ))
        placed.add(pid)

    still_unmatched = [pid for pid in unmatched if pid not in placed]

    return placement_list, still_unmatched


def _find_unplaced(
    candidate_list: list[PairScore], placed: set[int]
) -> Optional[PairScore]:
    for ps in candidate_list:
        if ps.piece_b not in placed:
            return ps
    return None


def _find_top_left_corner(feat: EdgeFeatures) -> EdgeFeatures:
    """
    调整角块的旋转，使得 top 和 left 均为直边（适合放在 (0,0)）。
    直接返回原对象（旋转由 placement 记录）。
    """
    return feat


def _corner_rotation(feat: EdgeFeatures) -> int:
    """
    对角块，找到旋转量使得 top 和 left 均为直边。
    返回旋转次数（0/1/2/3）。
    """
    for rot in range(4):
        top_flat = _rotated_is_flat(feat, "top", rot)
        left_flat = _rotated_is_flat(feat, "left", rot)
        if top_flat and left_flat:
            return rot
    return 0  # 兜底


def infer_grid_size(placements: list[Placement]) -> tuple[int, int]:
    """从放置结果推断网格尺寸"""
    if not placements:
        return 1, 1
    max_row = max(p.row for p in placements) + 1
    max_col = max(p.col for p in placements) + 1
    return max_row, max_col
