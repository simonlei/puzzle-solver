"""
hex_visualization.py — 六边形拼图求解结果可视化。

在原始截图上标注：
- misplaced 格子：彩色六边形轮廓 + 编号
- 目标位置：绿色六边形轮廓 + 编号
- 从 misplaced 格子到目标位置的箭头
"""
from __future__ import annotations

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .hex_grid import HexCell
from .hex_matcher import HexPlacement


def _estimate_pixel_pos(
    tq: int, tr: int,
    all_cells: list[HexCell],
) -> tuple[int, int] | None:
    """
    根据已知格子的轴坐标和像素坐标，用最近邻线性插值估算目标位置的像素坐标。

    使用已知格子中最近的几个点做加权平均位移。
    """
    if not all_cells:
        return None

    # 收集已知格子的坐标映射
    known = [(c.q, c.r, c.cx, c.cy) for c in all_cells]

    # 用所有已知格子的平均偏移量建立线性模型
    # px = a * q + b * r + c_x
    # py = d * q + e * r + c_y
    qs = np.array([k[0] for k in known], dtype=float)
    rs = np.array([k[1] for k in known], dtype=float)
    pxs = np.array([k[2] for k in known], dtype=float)
    pys = np.array([k[3] for k in known], dtype=float)

    # 最小二乘拟合
    A = np.column_stack([qs, rs, np.ones(len(known))])
    try:
        coeffs_x, _, _, _ = np.linalg.lstsq(A, pxs, rcond=None)
        coeffs_y, _, _, _ = np.linalg.lstsq(A, pys, rcond=None)
        px = int(coeffs_x[0] * tq + coeffs_x[1] * tr + coeffs_x[2])
        py = int(coeffs_y[0] * tq + coeffs_y[1] * tr + coeffs_y[2])
        return (px, py)
    except Exception:
        return None


def draw_hex_solution(
    image_bgr: np.ndarray,
    all_cells: list[HexCell],
    placements: list[HexPlacement],
) -> np.ndarray:
    """
    在原图上标注求解结果。

    标注内容：
    - 每个 misplaced 格子：彩色六边形轮廓 + 编号标签
    - 每个目标位置：对应颜色六边形轮廓（虚线）+ 编号
    - 从 misplaced 格子到目标位置的箭头
    - 旋转量标注（如 "↻120°"）
    """
    output = image_bgr.copy()

    # 若没有需要归位的碎片
    if not placements:
        _draw_text_center(output, "All pieces placed correctly!", color=(0, 200, 0))
        return output

    # 为每个 placement 分配颜色（HSV 均匀分布）
    n = len(placements)
    colors = []
    for i in range(n):
        hue = int(i * 180 / max(n, 1)) % 180
        hsv_c = np.array([[[hue, 220, 220]]], dtype=np.uint8)
        bgr = cv.cvtColor(hsv_c, cv.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))

    # 推算目标位置像素坐标
    target_pixels: dict[int, tuple[int, int] | None] = {}
    for idx, pl in enumerate(placements):
        # 先检查目标位置是否已有已知格子
        target_cell = next(
            (c for c in all_cells if c.q == pl.target_q and c.r == pl.target_r),
            None
        )
        if target_cell:
            target_pixels[idx] = (target_cell.cx, target_cell.cy)
        else:
            target_pixels[idx] = _estimate_pixel_pos(pl.target_q, pl.target_r, all_cells)

    # 画目标位置轮廓（先画，在箭头之下）
    if all_cells:
        ref_R = all_cells[0].radius / (np.sqrt(3) / 2)
    else:
        ref_R = 40.0

    for idx, (pl, color) in enumerate(zip(placements, colors)):
        tpx = target_pixels.get(idx)
        if tpx is None:
            continue
        tx, ty = tpx
        h, w = output.shape[:2]
        if not (0 <= tx < w and 0 <= ty < h):
            continue
        pts = _hex_vertices(tx, ty, ref_R)
        pts_arr = np.array(pts, dtype=np.int32)
        # 画虚线轮廓（用短线段模拟）
        for j in range(6):
            p1 = pts[j]
            p2 = pts[(j + 1) % 6]
            # 只画前半段（模拟虚线）
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv.line(output, p1, mid, color, 2, cv.LINE_AA)
        # 编号
        label_r = max(12, int(ref_R * 0.3))
        cv.circle(output, (tx, ty), label_r, color, 2)
        cv.putText(output, str(idx + 1), (tx - 6 * len(str(idx + 1)) // 2, ty + 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

    # 画 misplaced 格子标注和箭头
    for idx, (pl, color) in enumerate(zip(placements, colors)):
        _draw_misplaced(output, pl, idx + 1, color, target_pixels.get(idx))

    # 图例
    _draw_legend(output, placements)

    return output


def _hex_vertices(cx: float, cy: float, R: float) -> list[tuple[int, int]]:
    """计算 pointy-top 六边形的 6 个顶点"""
    pts = []
    for i in range(6):
        angle = np.radians(30 + 60 * i)
        pts.append((int(cx + R * np.cos(angle)), int(cy - R * np.sin(angle))))
    return pts


def _draw_target(
    canvas: np.ndarray,
    pl: HexPlacement,
    label: int,
    color: tuple,
) -> None:
    pass  # 目标位置绘制已移到 draw_hex_solution 中


def _draw_misplaced(
    canvas: np.ndarray,
    pl: HexPlacement,
    label: int,
    color: tuple,
    target_px: tuple[int, int] | None = None,
) -> None:
    """标注 misplaced 格子：编号、箭头指向目标、旋转量"""
    cell = pl.misplaced_cell
    cx, cy = cell.cx, cell.cy
    R = cell.radius / (np.sqrt(3) / 2)
    r = cell.radius  # 内切圆半径

    # 画六边形轮廓（粗线）
    pts = _hex_vertices(cx, cy, R)
    pts_arr = np.array(pts, dtype=np.int32)
    cv.polylines(canvas, [pts_arr], True, color, 3, cv.LINE_AA)

    # 半透明填充
    overlay = canvas.copy()
    cv.fillPoly(overlay, [pts_arr], color)
    cv.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)

    # 编号标签（圆形背景）
    label_r = max(16, int(r * 0.35))
    cv.circle(canvas, (cx, cy), label_r + 2, (255, 255, 255), -1)
    cv.circle(canvas, (cx, cy), label_r, color, -1)
    cv.putText(canvas, str(label), (cx - 7 * len(str(label)) // 2, cy + 6),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

    # 画箭头到目标位置
    if target_px is not None:
        tx, ty = target_px
        h, w = canvas.shape[:2]
        if 0 <= tx < w and 0 <= ty < h:
            # 箭头从 misplaced 格子边缘出发
            dx, dy = tx - cx, ty - cy
            dist = max(1, np.sqrt(dx * dx + dy * dy))
            # 起点在格子外接圆边缘
            start_x = int(cx + (R + 4) * dx / dist)
            start_y = int(cy + (R + 4) * dy / dist)
            cv.arrowedLine(canvas, (start_x, start_y), (tx, ty),
                          color, 2, cv.LINE_AA, tipLength=0.15)

    # 旋转量标注已移除（碎片方向固定）


def _draw_legend(canvas: np.ndarray, placements: list[HexPlacement]) -> None:
    """在左上角绘制图例"""
    h, w = canvas.shape[:2]
    n = len(placements)

    lines = [
        f"Misplaced: {n} pieces",
    ]
    for idx, pl in enumerate(placements):
        lines.append(
            f"  #{idx+1}: ({pl.misplaced_cell.q},{pl.misplaced_cell.r})"
            f" -> ({pl.target_q},{pl.target_r})"
            f"  conf={pl.score:.0%}"
        )

    # 用 Pillow 渲染（支持更好的字体）
    pil = Image.fromarray(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # 半透明背景
    box_h = (len(lines) + 1) * 18 + 10
    box_w = 320
    draw.rectangle([(8, 8), (8 + box_w, 8 + box_h)], fill=(0, 0, 0, 160))

    for i, line in enumerate(lines):
        color = (255, 220, 100) if i == 0 else (200, 200, 200)
        draw.text((14, 14 + i * 18), line, fill=color, font=font)

    result = cv.cvtColor(np.array(pil), cv.COLOR_RGB2BGR)
    canvas[:] = result


def _draw_text_center(canvas: np.ndarray, text: str, color: tuple) -> None:
    h, w = canvas.shape[:2]
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    (tw, th), _ = cv.getTextSize(text, font, scale, 2)
    cv.putText(canvas, text, (w // 2 - tw // 2, h // 2), font, scale, color, 2, cv.LINE_AA)
