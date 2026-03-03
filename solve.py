#!/usr/bin/env python3
"""
六边形拼图求解脚本。

用法：
    python3 solve.py <输入图片路径> [输出图片路径]

示例：
    python3 solve.py p.jpg
    python3 solve.py p.jpg result.jpg
"""

import sys
import cv2 as cv

from app.hex_grid import detect_hex_grid
from app.hex_matcher import solve_all_misplaced
from app.hex_visualization import draw_hex_solution


def main():
    if len(sys.argv) < 2:
        print("用法: python3 solve.py <输入图片> [输出图片]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "result.jpg"

    img = cv.imread(input_path)
    if img is None:
        print(f"错误：无法读取图片 {input_path}")
        sys.exit(1)

    print(f"读取图片：{input_path} ({img.shape[1]}x{img.shape[0]})")

    print("检测六边形网格...")
    cells = detect_hex_grid(img)
    placed = [c for c in cells if c.status == "placed"]
    misplaced = [c for c in cells if c.status == "misplaced"]
    print(f"  共 {len(cells)} 个格子，已放置 {len(placed)}，错位 {len(misplaced)}")

    if not misplaced:
        print("所有格子已正确放置！")
        sys.exit(0)

    print("计算目标位置（匈牙利算法）...")
    placements = solve_all_misplaced(cells, img)
    print(f"  求解完成，共 {len(placements)} 个格子需要归位")

    print("\n归位方案：")
    for pl in placements:
        print(f"  ({pl.misplaced_cell.q},{pl.misplaced_cell.r}) -> "
              f"({pl.target_q},{pl.target_r})  置信度 {pl.score:.0%}")

    result = draw_hex_solution(img, cells, placements)
    cv.imwrite(output_path, result)
    print(f"\n结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
