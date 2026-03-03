"""
edge_features.py — 提取每块碎片四条边的颜色特征序列。

核心思路：
  - 用 minAreaRect 确定碎片主轴方向（对齐旋转后的坐标系）
  - 将轮廓点按相对质心的角度分为 top/right/bottom/left 四组
  - 各组插值到固定长度 N，在 Lab 色彩空间采样颜色
  - 同时检测直边（判断是否为角块/边块）
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .segmentation import PieceInfo

EDGE_SAMPLES = 64   # 每条边的采样点数


@dataclass
class EdgeFeatures:
    piece_id: int
    # 四条边的颜色序列，shape=(EDGE_SAMPLES, 3)，Lab 色彩空间
    top: np.ndarray
    right: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    # 是否为直边（True 表示该边是拼图外边缘，不需要匹配）
    top_is_flat: bool = False
    right_is_flat: bool = False
    bottom_is_flat: bool = False
    left_is_flat: bool = False

    def get_edge(self, side: str) -> np.ndarray:
        return getattr(self, side)

    def is_flat(self, side: str) -> bool:
        return getattr(self, f"{side}_is_flat")

    @property
    def flat_count(self) -> int:
        return sum([self.top_is_flat, self.right_is_flat,
                    self.bottom_is_flat, self.left_is_flat])

    @property
    def is_corner(self) -> bool:
        return self.flat_count == 2

    @property
    def is_border(self) -> bool:
        return self.flat_count == 1

    @property
    def is_interior(self) -> bool:
        return self.flat_count == 0


def extract_edge_features(piece: PieceInfo, image_lab: np.ndarray) -> EdgeFeatures:
    """
    提取一块碎片的四边颜色特征。

    Args:
        piece: PieceInfo，含 contour、min_rect、centroid
        image_lab: 整张图像在 Lab 色彩空间的版本

    Returns:
        EdgeFeatures，含四条边的颜色序列和直边标志
    """
    cx, cy = piece.centroid
    rect_center, rect_size, rect_angle = piece.min_rect

    # 将轮廓点转为相对于质心的极坐标角度
    pts = piece.contour.reshape(-1, 2).astype(float)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)  # [-π, π]

    # 按主轴方向旋转角度，使 top/right/bottom/left 对齐碎片自身坐标系
    axis_angle = np.deg2rad(rect_angle)  # minAreaRect 角度转弧度
    rotated_angles = angles - axis_angle

    # 分配四个象限（每象限对应一条边）
    # right: [-π/4, π/4), top: [π/4, 3π/4), left: [3π/4, 5π/4), bottom: [5π/4, 7π/4)
    def get_edge_points(a_min: float, a_max: float) -> np.ndarray:
        """取落在角度范围内的轮廓点"""
        ra = (rotated_angles - a_min) % (2 * np.pi)
        span = (a_max - a_min) % (2 * np.pi)
        idx = np.where(ra < span)[0]
        if len(idx) == 0:
            return pts[:1]  # 兜底，避免空数组
        # 按原始角度排序，保持方向一致
        idx = idx[np.argsort(angles[idx])]
        return pts[idx]

    quarter = np.pi / 2
    right_pts  = get_edge_points(-quarter,      quarter)
    top_pts    = get_edge_points(quarter,        3 * quarter)
    left_pts   = get_edge_points(3 * quarter,    5 * quarter)
    bottom_pts = get_edge_points(-3 * quarter,  -quarter)

    def sample_colors(edge_pts: np.ndarray) -> np.ndarray:
        """沿边上的点采样 Lab 颜色，插值到固定长度"""
        # 按沿边弧长参数化，均匀重采样到 EDGE_SAMPLES 个点
        if len(edge_pts) < 2:
            return np.zeros((EDGE_SAMPLES, 3), dtype=np.float32)

        diffs = np.diff(edge_pts, axis=0)
        dists = np.concatenate([[0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
        total = dists[-1]
        if total == 0:
            return np.zeros((EDGE_SAMPLES, 3), dtype=np.float32)

        # 在 [0, total] 上均匀取 EDGE_SAMPLES 个位置
        sample_dists = np.linspace(0, total, EDGE_SAMPLES)
        # 插值 x, y 坐标
        xs = np.interp(sample_dists, dists, edge_pts[:, 0])
        ys = np.interp(sample_dists, dists, edge_pts[:, 1])

        # 裁剪到图像范围
        img_h, img_w = image_lab.shape[:2]
        xs = np.clip(xs, 0, img_w - 1).astype(int)
        ys = np.clip(ys, 0, img_h - 1).astype(int)

        colors = image_lab[ys, xs].astype(np.float32)
        return colors

    top_colors    = sample_colors(top_pts)
    right_colors  = sample_colors(right_pts)
    bottom_colors = sample_colors(bottom_pts)
    left_colors   = sample_colors(left_pts)

    # 检测直边：若该边的 y（或 x）方差很小，认为是直边（外缘）
    top_flat    = _is_flat_edge(top_pts)
    right_flat  = _is_flat_edge(right_pts)
    bottom_flat = _is_flat_edge(bottom_pts)
    left_flat   = _is_flat_edge(left_pts)

    return EdgeFeatures(
        piece_id=piece.id,
        top=top_colors,
        right=right_colors,
        bottom=bottom_colors,
        left=left_colors,
        top_is_flat=top_flat,
        right_is_flat=right_flat,
        bottom_is_flat=bottom_flat,
        left_is_flat=left_flat,
    )


def _is_flat_edge(pts: np.ndarray, flatness_thresh: float = 5.0) -> bool:
    """
    判断一组点是否近似为直线（外边缘）。
    用点到两端连线的最大偏差来衡量。
    """
    if len(pts) < 3:
        return True
    p0, p1 = pts[0], pts[-1]
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return True
    # 各点到直线的距离
    normal = np.array([-line_vec[1], line_vec[0]]) / line_len
    deviations = np.abs((pts - p0) @ normal)
    return float(deviations.max()) < flatness_thresh


def extract_all_features(
    pieces: list[PieceInfo], image_bgr: np.ndarray
) -> list[EdgeFeatures]:
    """对所有碎片批量提取边缘特征"""
    image_lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2Lab).astype(np.float32)
    return [extract_edge_features(p, image_lab) for p in pieces]
