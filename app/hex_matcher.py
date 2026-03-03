"""
hex_matcher.py — 六边形碎片错位检测与归位匹配。

核心逻辑：
  对每个 misplaced 格子（红边），通过比较其6条边的图案
  与网格中所有空缺位置的相邻 placed 格子边缘的连续性，
  找出最可能的目标位置。

六边形 pointy-top 的 6 条边方向（角度以右为 0°，逆时针）：
  0: 右       ( 0°)
  1: 右上     ( 60°)
  2: 左上     (120°)
  3: 左       (180°)
  4: 左下     (240°)
  5: 右下     (300°)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .hex_grid import HexCell, HEX_DIRECTIONS, OPPOSITE_DIR, build_coord_map, get_neighbors

EDGE_SAMPLES = 48   # 每条边采样点数


# 六边形 pointy-top 的 6 条边，用边中点方向表示
# 相邻两顶角之间的中点方向（从格子中心出发）
_EDGE_ANGLES_DEG = [0, 60, 120, 180, 240, 300]

# 旋转映射：rotation k 表示顺时针旋转 k×60°
# rotation_k[dir_idx] = 旋转后的方向
def _rotated_dir(dir_idx: int, k: int) -> int:
    return (dir_idx + k) % 6


@dataclass
class HexPlacement:
    misplaced_cell: HexCell    # 需要归位的格子
    target_q: int              # 目标位置轴坐标
    target_r: int
    rotation: int              # 旋转量（0~5，单位 60°）
    score: float               # 匹配分数 [0,1]，越高越好
    evidence: list[str] = None # 调试信息


def extract_hex_edge_colors(
    cell: HexCell,
    image_lab: np.ndarray,
) -> list[np.ndarray]:
    """
    提取六边形格子 6 条边的颜色序列。

    每条边从两个相邻顶点之间的像素采样，
    返回 6 个 shape=(EDGE_SAMPLES, 3) 的 Lab 颜色数组。
    """
    cx, cy = cell.cx, cell.cy
    R = cell.radius / (np.sqrt(3) / 2)  # 外接圆半径（从内切圆换算）
    h, w = image_lab.shape[:2]

    # pointy-top 六边形的 6 个顶点角度（从右上顶点开始，逆时针）
    vertex_angles = [np.radians(30 + 60 * i) for i in range(6)]
    vertices = [
        (cx + R * np.cos(a), cy - R * np.sin(a))
        for a in vertex_angles
    ]

    edges = []
    for i in range(6):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % 6]
        # 在边的内侧略微偏移（避免采到相邻格子的像素）
        inset = 0.85
        mid_x = (v1[0] + v2[0]) / 2
        mid_y = (v1[1] + v2[1]) / 2
        # 向格子中心偏移 15%
        p1 = (v1[0] * inset + cx * (1 - inset), v1[1] * inset + cy * (1 - inset))
        p2 = (v2[0] * inset + cx * (1 - inset), v2[1] * inset + cy * (1 - inset))

        # 沿边均匀采样
        ts = np.linspace(0, 1, EDGE_SAMPLES)
        sample_xs = np.clip((p1[0] * (1 - ts) + p2[0] * ts).astype(int), 0, w - 1)
        sample_ys = np.clip((p1[1] * (1 - ts) + p2[1] * ts).astype(int), 0, h - 1)
        colors = image_lab[sample_ys, sample_xs].astype(np.float32)
        edges.append(colors)

    return edges


def _edge_ssd(a: np.ndarray, b: np.ndarray) -> float:
    """两条边颜色序列的均方差（越小越相似）"""
    diff = a.astype(float) - b.astype(float)
    return float(np.mean(diff ** 2))


def find_target_position(
    misplaced: HexCell,
    all_cells: list[HexCell],
    image_bgr: np.ndarray,
    top_k: int = 3,
) -> list[HexPlacement]:
    """
    为一个 misplaced 格子找到最可能的目标位置。

    策略：
    1. 找出所有 placed 格子及其周围的空缺位置（候选目标）
    2. 对每个候选位置，计算 misplaced 格子放过去后
       与已放好邻居的边缘连续性得分
    3. 返回 top_k 个最佳候选

    Args:
        misplaced: 需要归位的格子
        all_cells: 所有格子（包含 placed 和 misplaced）
        image_bgr: 原始 BGR 图像
        top_k: 返回最佳候选数量

    Returns:
        按得分降序排列的 HexPlacement 列表
    """
    image_lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2Lab).astype(np.float32)
    coord_map = build_coord_map(all_cells)

    placed_cells = [c for c in all_cells if c.status == 'placed']
    placed_coords = {(c.q, c.r) for c in placed_cells}

    # ---- 预先提取所有 placed 格子的边缘颜色 ----------------------
    placed_edges: dict[tuple[int, int], list[np.ndarray]] = {}
    for cell in placed_cells:
        placed_edges[(cell.q, cell.r)] = extract_hex_edge_colors(cell, image_lab)

    # ---- 提取 misplaced 格子的边缘颜色 --------------------------
    mis_edges = extract_hex_edge_colors(misplaced, image_lab)

    # ---- 枚举候选目标位置 ----------------------------------------
    # 候选位置 = 所有 placed 格子的空邻居（未被任何格子占据的位置）
    occupied = {(c.q, c.r) for c in all_cells}
    candidate_positions: set[tuple[int, int]] = set()
    for cell in placed_cells:
        for dq, dr in HEX_DIRECTIONS:
            pos = (cell.q + dq, cell.r + dr)
            if pos not in occupied:
                candidate_positions.add(pos)

    # 如果没有空缺位置，说明整个拼图已满，misplaced 格子是当前位置就是错的
    # 这时候把所有其他 misplaced 格子的位置也加进来作为候选
    if not candidate_positions:
        for cell in all_cells:
            if cell.status == 'misplaced' and (cell.q, cell.r) != (misplaced.q, misplaced.r):
                candidate_positions.add((cell.q, cell.r))

    if not candidate_positions:
        return []

    # ---- 对每个候选位置 × 6种旋转，计算匹配分数 -----------------
    results: list[HexPlacement] = []

    for (tq, tr) in candidate_positions:
        for rot in range(6):
            score, evidence = _score_placement(
                mis_edges, tq, tr, rot,
                placed_edges, coord_map
            )
            results.append(HexPlacement(
                misplaced_cell=misplaced,
                target_q=tq,
                target_r=tr,
                rotation=rot,
                score=score,
                evidence=evidence,
            ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]


def _score_placement(
    mis_edges: list[np.ndarray],
    tq: int, tr: int,
    rotation: int,
    placed_edges: dict[tuple[int, int], list[np.ndarray]],
    coord_map: dict,
) -> tuple[float, list[str]]:
    """
    计算将 misplaced 格子（旋转 rotation×60°）放到 (tq,tr) 后的匹配分数。

    对每个方向，如果该方向有 placed 邻居，比较：
      - misplaced 格子旋转后朝该方向的边
      - placed 邻居朝 misplaced 方向的边（对面边）
    """
    ssds = []
    evidence = []

    for dir_idx, (dq, dr) in enumerate(HEX_DIRECTIONS):
        nq, nr = tq + dq, tr + dr
        if (nq, nr) not in placed_edges:
            continue

        # misplaced 格子旋转后，dir_idx 方向对应原始的哪条边
        orig_edge_idx = _rotated_dir(dir_idx, -rotation) % 6
        mis_edge = mis_edges[orig_edge_idx]

        # 邻居朝 misplaced 方向的边（对面方向）
        opp_dir = OPPOSITE_DIR[dir_idx]
        neighbor_edge = placed_edges[(nq, nr)][opp_dir]

        ssd = _edge_ssd(mis_edge, neighbor_edge)
        ssds.append(ssd)
        evidence.append(f"dir{dir_idx}:ssd={ssd:.1f}")

    if not ssds:
        return 0.0, []

    avg_ssd = float(np.mean(ssds))
    # 归一化到 [0,1]，SSD=0 → score=1，SSD=5000 → score≈0
    score = float(np.exp(-avg_ssd / 2000.0))
    return score, evidence


def solve_all_misplaced(
    cells: list[HexCell],
    image_bgr: np.ndarray,
) -> list[HexPlacement]:
    """
    对所有 misplaced 格子求解目标位置，使用匈牙利算法保证一对一分配。

    构建 misplaced × candidate_positions 的分数矩阵，
    用线性分配（最大权重二分匹配）找全局最优方案。
    """
    from scipy.optimize import linear_sum_assignment

    image_lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2Lab).astype(np.float32)
    coord_map = build_coord_map(cells)

    misplaced_cells = [c for c in cells if c.status == 'misplaced']
    placed_cells = [c for c in cells if c.status == 'placed']

    if not misplaced_cells:
        return []

    # 预计算所有 placed 格子的边缘颜色
    placed_edges: dict[tuple[int, int], list[np.ndarray]] = {
        (c.q, c.r): extract_hex_edge_colors(c, image_lab)
        for c in placed_cells
    }

    # 候选目标位置 = 所有 placed 格子的空邻居
    occupied = {(c.q, c.r) for c in cells}
    candidate_positions: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in placed_cells:
        for dq, dr in HEX_DIRECTIONS:
            pos = (cell.q + dq, cell.r + dr)
            if pos not in occupied and pos not in seen:
                candidate_positions.append(pos)
                seen.add(pos)

    if not candidate_positions:
        # 网格已满，候选为其他 misplaced 格子的位置
        for cell in misplaced_cells:
            if (cell.q, cell.r) not in seen:
                candidate_positions.append((cell.q, cell.r))
                seen.add((cell.q, cell.r))

    if not candidate_positions:
        return []

    n_mis = len(misplaced_cells)
    n_cand = len(candidate_positions)

    # 预提取所有 misplaced 格子的边缘颜色
    mis_edges_list = [extract_hex_edge_colors(c, image_lab) for c in misplaced_cells]

    # 构建分数矩阵 [n_mis × n_cand]
    # 对每个 misplaced × candidate × rotation 取最大分数
    score_matrix = np.zeros((n_mis, n_cand), dtype=np.float32)
    best_rot_matrix = np.zeros((n_mis, n_cand), dtype=np.int32)

    for i, mis_edges in enumerate(mis_edges_list):
        for j, (tq, tr) in enumerate(candidate_positions):
            best_score = 0.0
            best_rot = 0
            for rot in range(6):
                score, _ = _score_placement(
                    mis_edges, tq, tr, rot, placed_edges, coord_map
                )
                if score > best_score:
                    best_score = score
                    best_rot = rot
            score_matrix[i, j] = best_score
            best_rot_matrix[i, j] = best_rot

    # 用匈牙利算法找最大权重匹配
    # linear_sum_assignment 最小化，所以用负分数
    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    results: list[HexPlacement] = []
    for i, j in zip(row_ind, col_ind):
        tq, tr = candidate_positions[j]
        rot = int(best_rot_matrix[i, j])
        score = float(score_matrix[i, j])
        results.append(HexPlacement(
            misplaced_cell=misplaced_cells[i],
            target_q=tq,
            target_r=tr,
            rotation=rot,
            score=score,
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results
