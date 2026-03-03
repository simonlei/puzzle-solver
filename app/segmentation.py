"""
segmentation.py — 从拼图图片中分割出各个碎片。

输入：BGR 格式的 numpy 图像
输出：PieceInfo 列表，每项包含碎片的 mask、bbox、轮廓、质心等信息
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np


@dataclass
class PieceInfo:
    id: int
    mask: np.ndarray          # H×W uint8，碎片区域为 255
    bbox: tuple               # (x, y, w, h)
    contour: np.ndarray       # OpenCV 轮廓点数组
    centroid: tuple           # (cx, cy) 质心坐标
    area: float               # 像素面积
    min_rect: tuple = field(default=None)  # cv.minAreaRect 结果，含旋转角度


def segment_pieces(
    image_bgr: np.ndarray,
    min_area: int = 800,
    bg_saturation_thresh: int = 40,
    bg_value_thresh: int = 180,
) -> list[PieceInfo]:
    """
    从散乱拼图照片中分割出所有碎片。

    假设：单色背景（白色/浅色），碎片不严重重叠。

    Args:
        image_bgr: 输入图像（BGR）
        min_area: 过滤面积小于此值的噪点区域（像素数）
        bg_saturation_thresh: 背景 HSV 饱和度上限（用于提取前景）
        bg_value_thresh: 背景 HSV 明度下限（白色背景时较高）

    Returns:
        PieceInfo 列表，按面积从大到小排序
    """
    h, w = image_bgr.shape[:2]

    # ---- 1. 前景提取 ------------------------------------------------
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)

    # 白/浅色背景：低饱和度 + 高明度
    bg_mask = cv.inRange(
        hsv,
        np.array([0, 0, bg_value_thresh]),
        np.array([180, bg_saturation_thresh, 255]),
    )
    fg_mask = cv.bitwise_not(bg_mask)

    # Otsu 兜底（应对不均匀光照）
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    _, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    fg_mask = cv.bitwise_or(fg_mask, otsu)

    # ---- 2. 形态学去噪 -----------------------------------------------
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel, iterations=1)

    # ---- 3. Watershed 分离粘连碎片 -----------------------------------
    dist = cv.distanceTransform(fg_mask, cv.DIST_L2, 5)
    dist_norm = cv.normalize(dist, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # 确定前景种子：距离变换峰值区域
    _, sure_fg = cv.threshold(dist_norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    sure_fg = sure_fg.astype(np.uint8)

    sure_bg = cv.dilate(fg_mask, kernel, iterations=3)
    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1          # 背景标为 1
    markers[unknown == 255] = 0    # 待确定区域标为 0

    markers = cv.watershed(image_bgr, markers)

    # ---- 4. 提取各碎片信息 -------------------------------------------
    pieces: list[PieceInfo] = []
    unique_labels = np.unique(markers)

    for label in unique_labels:
        if label <= 1:   # 1 = 背景，-1 = 边界线
            continue

        piece_mask = np.zeros((h, w), dtype=np.uint8)
        piece_mask[markers == label] = 255

        contours, _ = cv.findContours(
            piece_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv.contourArea)
        area = cv.contourArea(contour)
        if area < min_area:
            continue

        M = cv.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        pieces.append(
            PieceInfo(
                id=len(pieces),          # 临时 ID，后续按排序重编
                mask=piece_mask,
                bbox=cv.boundingRect(contour),
                contour=contour,
                centroid=(cx, cy),
                area=area,
                min_rect=cv.minAreaRect(contour),
            )
        )

    # 按面积从大到小排序，重编 ID
    pieces.sort(key=lambda p: p.area, reverse=True)
    for idx, p in enumerate(pieces):
        p.id = idx

    return pieces
