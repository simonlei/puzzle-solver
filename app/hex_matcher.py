"""
hex_matcher.py — 六边形碎片错位检测与归位匹配。

核心逻辑：
  对每个 misplaced 格子（红边），通过比较其6条边的图案
  与网格中所有空缺位置的相邻 placed 格子边缘的连续性，
  找出最可能的目标位置。碎片方向固定，不考虑旋转。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .hex_grid import HexCell, HEX_DIRECTIONS, OPPOSITE_DIR

EDGE_SAMPLES = 48   # 每条边采样点数


@dataclass
class HexPlacement:
    misplaced_cell: HexCell    # 需要归位的格子
    target_q: int              # 目标位置轴坐标
    target_r: int
    score: float               # 匹配分数 [0,1]，越高越好


def extract_hex_edge_colors(
    cell: HexCell,
    image_lab: np.ndarray,
) -> list[np.ndarray]:
    """
    提取六边形格子 6 条边的颜色序列。
    返回 6 个 shape=(EDGE_SAMPLES, 3) 的 Lab 颜色数组。
    """
    cx, cy = cell.cx, cell.cy
    R = cell.radius / (np.sqrt(3) / 2)
    h, w = image_lab.shape[:2]

    vertex_angles = [np.radians(30 + 60 * i) for i in range(6)]
    vertices = [
        (cx + R * np.cos(a), cy - R * np.sin(a))
        for a in vertex_angles
    ]

    edges = []
    for i in range(6):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % 6]
        inset = 0.85
        p1 = (v1[0] * inset + cx * (1 - inset), v1[1] * inset + cy * (1 - inset))
        p2 = (v2[0] * inset + cx * (1 - inset), v2[1] * inset + cy * (1 - inset))

        ts = np.linspace(0, 1, EDGE_SAMPLES)
        sample_xs = np.clip((p1[0] * (1 - ts) + p2[0] * ts).astype(int), 0, w - 1)
        sample_ys = np.clip((p1[1] * (1 - ts) + p2[1] * ts).astype(int), 0, h - 1)
        edges.append(image_lab[sample_ys, sample_xs].astype(np.float32))

    return edges


def _edge_ssd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(float) - b.astype(float)
    return float(np.mean(diff ** 2))


def _score_placement(
    mis_edges: list[np.ndarray],
    tq: int, tr: int,
    placed_edges: dict[tuple[int, int], list[np.ndarray]],
) -> float:
    """计算将 misplaced 格子放到 (tq,tr) 后与已放好邻居的边缘匹配分数。"""
    ssds = []
    for dir_idx, (dq, dr) in enumerate(HEX_DIRECTIONS):
        nq, nr = tq + dq, tr + dr
        if (nq, nr) not in placed_edges:
            continue
        opp_dir = OPPOSITE_DIR[dir_idx]
        ssds.append(_edge_ssd(mis_edges[dir_idx], placed_edges[(nq, nr)][opp_dir]))

    if not ssds:
        return 0.0
    return float(np.exp(-np.mean(ssds) / 2000.0))


def solve_all_misplaced(
    cells: list[HexCell],
    image_bgr: np.ndarray,
) -> list[HexPlacement]:
    """
    对所有 misplaced 格子求解目标位置，使用匈牙利算法保证一对一分配。
    碎片方向固定，不考虑旋转。
    """
    from scipy.optimize import linear_sum_assignment

    image_lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2Lab).astype(np.float32)

    misplaced_cells = [c for c in cells if c.status == 'misplaced']
    placed_cells = [c for c in cells if c.status == 'placed']

    if not misplaced_cells:
        return []

    placed_edges: dict[tuple[int, int], list[np.ndarray]] = {
        (c.q, c.r): extract_hex_edge_colors(c, image_lab)
        for c in placed_cells
    }

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
        return []

    mis_edges_list = [extract_hex_edge_colors(c, image_lab) for c in misplaced_cells]

    n_mis = len(misplaced_cells)
    n_cand = len(candidate_positions)
    score_matrix = np.zeros((n_mis, n_cand), dtype=np.float32)
    for i, mis_edges in enumerate(mis_edges_list):
        for j, (tq, tr) in enumerate(candidate_positions):
            score_matrix[i, j] = _score_placement(mis_edges, tq, tr, placed_edges)

    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    results: list[HexPlacement] = []
    for i, j in zip(row_ind, col_ind):
        tq, tr = candidate_positions[j]
        results.append(HexPlacement(
            misplaced_cell=misplaced_cells[i],
            target_q=tq,
            target_r=tr,
            score=float(score_matrix[i, j]),
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results
