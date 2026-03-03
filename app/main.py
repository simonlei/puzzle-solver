"""
main.py — Puzzle Solver FastAPI 服务入口。

端点：
  POST /solve        接受散乱拼图图片，返回并排标注图（JPEG）
  POST /solve/json   返回结构化布局数据（JSON）
  GET  /health       服务健康检查
"""

from __future__ import annotations

import io
import time
from typing import Annotated

import cv2 as cv
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from .edge_features import extract_all_features
from .layout_solver import infer_grid_size, solve_layout
from .pair_scorer import compute_pair_scores
from .segmentation import segment_pieces
from .visualization import generate_output

app = FastAPI(
    title="Puzzle Solver",
    version="1.0.0",
    description=(
        "接受一张散乱拼图碎片照片，输出标注图（碎片编号）"
        "和网格图（每个位置对应的碎片编号）。"
    ),
)

# 图片尺寸上限（像素），超过则等比缩放
MAX_DIMENSION = 4000
# 碎片数量下限，不足时报错
MIN_PIECES = 2


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析图片，请上传有效的 JPEG/PNG 文件。")
    return img


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MAX_DIMENSION:
        return img
    scale = MAX_DIMENSION / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)


def _run_pipeline(image_bgr: np.ndarray) -> dict:
    """运行完整的拼图求解流水线，返回结果字典。"""
    t0 = time.monotonic()

    # 1. 碎片分割
    pieces = segment_pieces(image_bgr)
    if len(pieces) < MIN_PIECES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"只检测到 {len(pieces)} 块碎片，无法求解。"
                "请确保：① 使用单色背景（白色/蓝色）；② 碎片不严重重叠；③ 图片清晰。"
            ),
        )

    # 2. 边缘特征提取
    features = extract_all_features(pieces, image_bgr)

    # 3. 配对评分
    scores = compute_pair_scores(features)

    # 4. 布局求解
    placements, unmatched = solve_layout(features, scores)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    return {
        "image_bgr": image_bgr,
        "pieces": pieces,
        "placements": placements,
        "unmatched": unmatched,
        "elapsed_ms": elapsed_ms,
    }


@app.post(
    "/solve",
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "并排输出：左侧标注图 + 右侧网格图（JPEG）",
        },
        400: {"description": "无效图片"},
        422: {"description": "碎片检测失败"},
    },
    summary="求解拼图并返回可视化图片",
)
async def solve(
    pieces_image: Annotated[UploadFile, File(description="散乱拼图碎片照片（JPEG/PNG）")],
) -> Response:
    data = await pieces_image.read()
    image_bgr = _decode_image(data)
    image_bgr = _resize_if_needed(image_bgr)

    result = _run_pipeline(image_bgr)

    output_img = generate_output(
        result["image_bgr"],
        result["pieces"],
        result["placements"],
        result["unmatched"],
    )

    _, buf = cv.imencode(".jpg", output_img, [cv.IMWRITE_JPEG_QUALITY, 90])

    n_rows, n_cols = infer_grid_size(result["placements"])
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Pieces-Count": str(len(result["pieces"])),
            "X-Grid-Size": f"{n_rows}x{n_cols}",
            "X-Unmatched-Count": str(len(result["unmatched"])),
            "X-Solve-Time-Ms": str(result["elapsed_ms"]),
        },
    )


@app.post(
    "/solve/json",
    summary="求解拼图并返回结构化 JSON 数据",
)
async def solve_json(
    pieces_image: Annotated[UploadFile, File(description="散乱拼图碎片照片（JPEG/PNG）")],
) -> dict:
    data = await pieces_image.read()
    image_bgr = _decode_image(data)
    image_bgr = _resize_if_needed(image_bgr)

    result = _run_pipeline(image_bgr)

    n_rows, n_cols = infer_grid_size(result["placements"])

    return {
        "pieces_count": len(result["pieces"]),
        "grid": {"rows": n_rows, "cols": n_cols},
        "placements": [
            {
                "piece_id": p.piece_id,
                "row": p.row,
                "col": p.col,
                "rotation": p.rotation * 90,
                "confidence": round(p.confidence, 3),
            }
            for p in result["placements"]
        ],
        "unmatched": result["unmatched"],
        "solve_time_ms": result["elapsed_ms"],
    }


@app.get("/health", summary="服务健康检查")
async def health() -> dict:
    return {
        "status": "ok",
        "opencv_version": cv.__version__,
    }
