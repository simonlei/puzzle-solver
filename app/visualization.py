"""
visualization.py — 生成拼图求解结果的可视化输出。

输出：两图并排的单张 JPEG：
  左侧 - 标注图：在原始散乱图上为每块碎片标上编号和彩色轮廓
  右侧 - 网格图：等大小格子网格，每格标注对应碎片编号
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .segmentation import PieceInfo
from .layout_solver import Placement, infer_grid_size

# 每块碎片的唯一颜色（BGR）
_RNG = np.random.default_rng(seed=42)


def _piece_colors(n: int) -> list[tuple[int, int, int]]:
    """为 n 块碎片生成视觉上易区分的 BGR 颜色"""
    colors = []
    for i in range(n):
        hue = int(i * 180 / max(n, 1)) % 180
        hsv_color = np.array([[[hue, 220, 200]]], dtype=np.uint8)
        bgr = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def draw_annotated_image(
    image_bgr: np.ndarray,
    pieces: list[PieceInfo],
) -> np.ndarray:
    """
    在原始散乱图上绘制每块碎片的编号和彩色轮廓。

    Args:
        image_bgr: 原始 BGR 图像
        pieces: 碎片列表（含轮廓和质心）

    Returns:
        标注后的 BGR 图像
    """
    output = image_bgr.copy()
    colors = _piece_colors(len(pieces))

    for piece in pieces:
        color = colors[piece.id % len(colors)]
        cx, cy = piece.centroid

        # 绘制轮廓
        cv.drawContours(output, [piece.contour], -1, color, 2, cv.LINE_AA)

        # 绘制编号标签
        label = str(piece.id)
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), baseline = cv.getTextSize(label, font, font_scale, thickness)

        lx = cx - tw // 2
        ly = cy + th // 2

        # 背景色块
        cv.rectangle(
            output,
            (lx - 3, ly - th - 3),
            (lx + tw + 3, ly + baseline + 1),
            color, cv.FILLED,
        )
        # 白色文字
        cv.putText(output, label, (lx, ly), font, font_scale,
                   (255, 255, 255), thickness, cv.LINE_AA)

    return output


def draw_grid_image(
    placements: list[Placement],
    unmatched_ids: list[int],
    cell_size: int = 60,
) -> np.ndarray:
    """
    生成网格图：每个格子标注对应的碎片编号。

    Args:
        placements: 布局求解结果
        unmatched_ids: 未定位的碎片 ID
        cell_size: 每个格子的像素大小

    Returns:
        网格图 BGR 图像
    """
    if not placements:
        blank = np.full((cell_size, cell_size, 3), 240, dtype=np.uint8)
        return blank

    n_rows, n_cols = infer_grid_size(placements)
    colors = _piece_colors(max(p.piece_id for p in placements) + 1)

    padding = 40   # 顶部留给标题
    img_h = n_rows * cell_size + padding
    img_w = n_cols * cell_size

    canvas = np.full((img_h, img_w, 3), 245, dtype=np.uint8)

    # 绘制格子和编号
    for p in placements:
        y0 = p.row * cell_size + padding
        x0 = p.col * cell_size
        y1 = y0 + cell_size
        x1 = x0 + cell_size

        color = colors[p.piece_id % len(colors)]

        # 低置信度用灰色底
        if p.confidence < 0.3:
            bg = (220, 220, 220)
        else:
            # 用碎片颜色的浅化版作为背景
            bg = tuple(min(255, int(c * 0.3 + 200)) for c in color)

        cv.rectangle(canvas, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), bg, cv.FILLED)
        cv.rectangle(canvas, (x0, y0), (x1, y1), (180, 180, 180), 1)

        # 编号文字
        label = str(p.piece_id)
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 if cell_size >= 50 else 0.35
        thickness = 1
        (tw, th), _ = cv.getTextSize(label, font, font_scale, thickness)
        tx = x0 + (cell_size - tw) // 2
        ty = y0 + (cell_size + th) // 2

        text_color = (60, 60, 60) if p.confidence >= 0.3 else (160, 160, 160)
        cv.putText(canvas, label, (tx, ty), font, font_scale,
                   text_color, thickness, cv.LINE_AA)

        # 低置信度加问号标记
        if p.confidence < 0.3:
            cv.putText(canvas, "?", (x1 - 14, y0 + 14),
                       cv.FONT_HERSHEY_SIMPLEX, 0.35, (180, 100, 100), 1, cv.LINE_AA)

    # 绘制标题（用 Pillow 支持更好的字体）
    canvas_pil = Image.fromarray(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas_pil)
    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
    except (IOError, OSError):
        font_title = ImageFont.load_default()

    title = f"Grid {n_rows}×{n_cols}  |  {len(placements)} pieces"
    if unmatched_ids:
        title += f"  |  {len(unmatched_ids)} unmatched"
    draw.text((6, 6), title, fill=(60, 60, 60), font=font_title)

    return cv.cvtColor(np.array(canvas_pil), cv.COLOR_RGB2BGR)


def compose_result_image(
    annotated: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    将标注图和网格图并排合并为单张图。
    高度对齐（不足的一侧补灰边）。
    """
    h1, w1 = annotated.shape[:2]
    h2, w2 = grid.shape[:2]
    h = max(h1, h2)

    def pad_height(img: np.ndarray, target_h: int) -> np.ndarray:
        dh = target_h - img.shape[0]
        if dh <= 0:
            return img
        pad = np.full((dh, img.shape[1], 3), 240, dtype=np.uint8)
        return np.vstack([img, pad])

    left = pad_height(annotated, h)
    right = pad_height(grid, h)

    # 垂直分隔线
    divider = np.full((h, 12, 3), 200, dtype=np.uint8)

    return np.hstack([left, divider, right])


def generate_output(
    image_bgr: np.ndarray,
    pieces: list[PieceInfo],
    placements: list[Placement],
    unmatched_ids: list[int],
    cell_size: int = 60,
) -> np.ndarray:
    """
    一步生成最终输出图（标注图 + 网格图并排）。
    """
    annotated = draw_annotated_image(image_bgr, pieces)

    # 自适应格子大小（图太宽时缩小）
    _, n_cols = infer_grid_size(placements)
    max_grid_width = max(image_bgr.shape[1], 400)
    adapted_cell = min(cell_size, max_grid_width // max(n_cols, 1))
    adapted_cell = max(adapted_cell, 30)

    grid = draw_grid_image(placements, unmatched_ids, cell_size=adapted_cell)

    return compose_result_image(annotated, grid)
