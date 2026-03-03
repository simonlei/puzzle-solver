"""
hex_grid.py — 从六边形拼图截图中检测网格结构。

输出每个六边形格子的：
- 像素中心坐标
- 轴坐标 (q, r)（六边形网格标准坐标系）
- 状态：'placed'（白边，已放对）或 'misplaced'（红边，需要归位）

六边形轴坐标系（pointy-top，尖顶朝上）：
  - q 轴：向右
  - r 轴：向右下
  相邻方向：(+1,0),(−1,0),(0,+1),(0,−1),(+1,−1),(−1,+1)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
from scipy.ndimage import maximum_filter


@dataclass
class HexCell:
    cx: int                  # 像素中心 x
    cy: int                  # 像素中心 y
    q: int                   # 轴坐标 q
    r: int                   # 轴坐标 r
    status: str              # 'placed' | 'misplaced' | 'unknown'
    radius: float = 0.0      # 内切圆半径（像素）
    edge_colors: list = field(default_factory=list)


# 六边形 pointy-top 的 6 个邻居方向偏移（轴坐标）
HEX_DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
OPPOSITE_DIR = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}


def detect_hex_grid(image_bgr: np.ndarray) -> list[HexCell]:
    """
    从拼图截图中检测所有六边形格子。

    流程：
    1. 提取红色和白色网格线
    2. 距离变换 + 局部极大值找格子中心
    3. 用 NMS 去除重复中心点
    4. 聚类分行，建立六边形轴坐标系
    5. 判断每个格子的边框颜色（红/白）
    """
    h, w = image_bgr.shape[:2]
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)

    # ---- 1. 提取网格线 ------------------------------------------
    red1 = cv.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
    red2 = cv.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))
    red_mask = cv.bitwise_or(red1, red2)
    # placed 格子的边框是灰/银色（低饱和度，亮度适中）
    gray_mask = cv.inRange(hsv, np.array([0, 0, 140]), np.array([180, 30, 255]))
    # 保留 white_mask 兼容性（white 是 gray 的子集）
    white_mask = gray_mask
    all_lines = cv.bitwise_or(red_mask, gray_mask)

    # 去掉金色/橙色外框
    gold_mask = cv.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
    gold_fat = cv.dilate(gold_mask, np.ones((3, 3), np.uint8), iterations=15)

    lines_fat = cv.dilate(all_lines, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=2)
    interior = cv.bitwise_and(cv.bitwise_not(lines_fat), cv.bitwise_not(gold_fat))

    # ---- 2. 距离变换估算六边形尺寸 ------------------------------
    dist = cv.distanceTransform(interior, cv.DIST_L2, 5)
    hex_radius = float(np.percentile(dist[dist > 5], 95))  # 用95分位数，更稳定
    hex_R = hex_radius / (np.sqrt(3) / 2)   # 外接圆半径
    row_h = np.sqrt(3) * hex_R              # pointy-top 行间距

    # ---- 3. 局部极大值找格子中心 ---------------------------------
    # 窗口必须略大于六边形直径，确保每个格子只有一个极大值
    window = max(30, int(hex_R * 1.6))
    local_max_mask = (dist == maximum_filter(dist, size=window)) & (dist > hex_radius * 0.55)

    ys, xs = np.where(local_max_mask)

    # 过滤靠近图像边缘和金色外框的点
    margin = int(hex_radius * 0.8)
    gold_border = gold_fat > 0
    valid_mask = (
        (xs > margin) & (xs < w - margin) &
        (ys > margin) & (ys < h - margin) &
        ~gold_border[ys, xs]
    )
    xs, ys = xs[valid_mask], ys[valid_mask]

    if len(xs) == 0:
        return []

    # NMS：若两个中心距离 < hex_R，保留 dist 值更大的
    centers_xy = np.column_stack([xs, ys])
    dist_vals = dist[ys, xs]
    keep = _nms_centers(centers_xy, dist_vals, min_dist=hex_R * 0.8)
    centers_xy = centers_xy[keep]
    xs, ys = centers_xy[:, 0], centers_xy[:, 1]

    # ---- 4. 建立轴坐标系 ----------------------------------------
    cells = _assign_axial_coords(xs, ys, row_h, hex_R, hex_radius)

    # ---- 5. 判断边框颜色 ----------------------------------------
    _classify_cells(cells, red_mask, white_mask, hex_radius)

    return cells


def _nms_centers(
    centers: np.ndarray,
    scores: np.ndarray,
    min_dist: float,
) -> np.ndarray:
    """
    非极大值抑制：去除距离过近的重复中心点，保留得分最高的。
    返回保留点的索引数组。
    """
    order = np.argsort(-scores)  # 按得分降序
    keep = []
    suppressed = np.zeros(len(centers), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        # 抑制距离过近的其他点
        dists = np.linalg.norm(centers - centers[idx], axis=1)
        suppressed |= (dists < min_dist) & (dists > 0)

    return np.array(keep)


def _assign_axial_coords(
    xs: np.ndarray,
    ys: np.ndarray,
    row_h: float,
    hex_R: float,
    hex_radius: float,
) -> list[HexCell]:
    """
    将像素坐标转换为六边形轴坐标。

    算法：
    1. 按 y 坐标聚类分行（容差 = row_h * 0.4）
    2. 每行内按 x 排序分配列索引
    3. offset 坐标 → axial 坐标
    """
    ys_arr = ys.astype(float)
    xs_arr = xs.astype(float)

    # 用贪心聚类分行
    sorted_idx = np.argsort(ys_arr)
    row_labels = np.full(len(ys_arr), -1, dtype=int)
    row_y_centers = []
    tolerance = row_h * 0.45

    for idx in sorted_idx:
        cy = ys_arr[idx]
        assigned = False
        for row_id, ry in enumerate(row_y_centers):
            if abs(cy - ry) < tolerance:
                row_labels[idx] = row_id
                # 更新行中心（滑动平均）
                row_y_centers[row_id] = (ry * 0.8 + cy * 0.2)
                assigned = True
                break
        if not assigned:
            row_labels[idx] = len(row_y_centers)
            row_y_centers.append(cy)

    # 按行构建格子
    rows: dict[int, list] = defaultdict(list)
    for i in range(len(xs_arr)):
        rows[row_labels[i]].append((int(xs_arr[i]), int(ys_arr[i])))

    # 在每行内按 x 排序，分配列索引
    # pointy-top 六边形：偶数行和奇数行交错偏移半个格宽
    # col_w = 1.5 * hex_R（相邻列中心 x 距离）
    col_w = 1.5 * hex_R

    cells: list[HexCell] = []
    sorted_rows = sorted(rows.keys())

    # 找最左侧参考 x（第一个格子的 x 坐标）
    all_xs = [cx for row_pts in rows.values() for cx, cy in row_pts]
    x_ref = min(all_xs)

    for row_idx, row_id in enumerate(sorted_rows):
        pts = sorted(rows[row_id], key=lambda p: p[0])
        row_y = int(np.mean([p[1] for p in pts]))

        for cx, cy in pts:
            # 将 x 坐标量化到最近的列位置
            # pointy-top offset：奇数行右偏 hex_R*0.75
            offset_x = hex_R * 0.75 if (row_idx % 2 == 1) else 0.0
            col_float = (cx - x_ref - offset_x) / col_w
            col_idx = int(round(col_float))

            # offset → axial 坐标（pointy-top odd-r offset）
            q = col_idx - (row_idx - (row_idx & 1)) // 2
            r = row_idx

            cells.append(HexCell(
                cx=cx, cy=cy,
                q=q, r=r,
                status='unknown',
                radius=hex_radius,
            ))

    return cells


def _classify_cells(
    cells: list[HexCell],
    red_mask: np.ndarray,
    white_mask: np.ndarray,
    hex_radius: float,
) -> None:
    """
    对每个格子，沿 6 个边中点方向径向扫描，判断是红边还是白边。
    原地修改 cell.status。

    采样策略：沿 6 个边法线方向，从外接圆半径的 85% 到 150% 进行径向扫描，
    统计红色和白色像素数量，多者为准。
    """
    h, w = red_mask.shape[:2]

    # pointy-top 六边形：6条边的外法线方向（从中心指向边中点）
    edge_angles_rad = [np.radians(a) for a in [0, 60, 120, 180, 240, 300]]

    for cell in cells:
        cx, cy = cell.cx, cell.cy
        cell_R = cell.radius / (np.sqrt(3) / 2)  # 外接圆半径

        # 按方向统计：每个方向单独统计红/白像素
        dir_reds = []
        dir_whites = []

        for angle_rad in edge_angles_rad:
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            r_cnt = w_cnt = 0
            for dist in np.arange(cell_R * 0.85, cell_R * 1.5, 1.0):
                sx = int(cx + dist * cos_a)
                sy = int(cy + dist * sin_a)
                if 0 <= sx < w and 0 <= sy < h:
                    if red_mask[sy, sx] > 0:
                        r_cnt += 1
                    elif white_mask[sy, sx] > 0:
                        w_cnt += 1
            dir_reds.append(r_cnt)
            dir_whites.append(w_cnt)

        # 红色格子：只要有任意方向出现明显红色（>=3像素），即判为 misplaced
        # 白色格子：多个方向都有白色且无明显红色，判为 placed
        max_red = max(dir_reds)
        total_red = sum(dir_reds)
        total_white = sum(dir_whites)

        if max_red >= 3:
            cell.status = 'misplaced'
        elif total_white > 0:
            cell.status = 'placed'
        else:
            cell.status = 'unknown'


def build_coord_map(cells: list[HexCell]) -> dict[tuple[int, int], HexCell]:
    """构建 (q, r) → HexCell 查询字典"""
    return {(c.q, c.r): c for c in cells}


def get_neighbors(cell: HexCell, coord_map: dict) -> list[tuple[int, HexCell]]:
    """返回格子的所有存在的邻居，格式 [(direction_idx, neighbor_cell), ...]"""
    neighbors = []
    for dir_idx, (dq, dr) in enumerate(HEX_DIRECTIONS):
        nq, nr = cell.q + dq, cell.r + dr
        if (nq, nr) in coord_map:
            neighbors.append((dir_idx, coord_map[(nq, nr)]))
    return neighbors
