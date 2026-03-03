# CLAUDE.md

---
title: "feat: Puzzle Piece Position Solver Service"
type: feat
status: completed
date: 2026-03-03
---

# feat: Puzzle Piece Position Solver Service

## Overview

构建一个 HTTP 服务，接受一张散乱拼图碎片的照片（100+ 块，碎片边缘带有图案），通过图案/边缘匹配算法推断每块碎片的目标位置，输出两张图：

1. **标注图**：在原始散乱图上为每块碎片标上编号
2. **网格图**：等大小的网格，每个格子标注对应的碎片编号，表示"这个位置应该放哪块碎片"

无参考完整图，系统需纯粹依靠碎片之间的图案连续性和形状凸凹互补来求解。

## Problem Statement

用户手中有一盒散开的拼图（100+ 块），拍摄一张全体碎片照，希望得到一份"哪块碎片去哪里"的指引，无需手动比对每一块。

核心难点：
- **无参考图**：没有完整成品图，必须通过碎片间相互关系求解
- **规模**：100+ 块，算法需在合理时间内完成
- **图案匹配**：碎片边缘有图案，是匹配的主要依据（优于纯形状凸凹）
- **旋转**：碎片可能以任意角度散落

## Proposed Solution

### 技术架构

```
输入图（散乱碎片照）
        │
        ▼
┌─────────────────────┐
│ 1. 碎片分割          │  OpenCV 轮廓检测 + Watershed
│    (Segmentation)   │  → 每块碎片的 mask、bbox、轮廓
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. 边缘采样          │  沿每块碎片四条边均匀采样像素颜色序列
│    (Edge Sampling)  │  → 每条边的颜色"指纹"
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. 配对评分          │  对所有碎片对的相邻边打分（颜色差异 + 形状互补）
│    (Pair Scoring)   │  → N×N 兼容性矩阵
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. 全局布局求解       │  贪心拼接 + 约束传播（先拼边角，再填内部）
│    (Layout Solver)  │  → 每块碎片的 (row, col, rotation) 赋值
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 5. 可视化输出        │  生成标注图 + 网格图
│    (Visualization)  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 6. FastAPI 服务      │  POST /solve → multipart image response
└─────────────────────┘
```

### 关键技术决策

| 环节 | 选型 | 理由 |
|------|------|------|
| 碎片分割 | OpenCV Watershed + 形态学 | 无 GPU 依赖，适合白/单色背景 |
| 边缘特征 | 像素颜色序列（Lab 色彩空间）| 颜色对光照变化更鲁棒；比纯形状包含更多信息 |
| 配对评分 | SSD（Sum of Squared Differences）| 简单高效；可并行 |
| 布局求解 | 贪心 + 回溯 | 先固定角块（4块），再沿边延伸，逐步填内部 |
| 旋转处理 | 每块碎片测试 0°/90°/180°/270° | 标准拼图只有 4 种方向 |
| HTTP 框架 | FastAPI + python-multipart | 标准图像 API 模式 |
| 图像处理 | opencv-contrib-python + Pillow | OpenCV 做计算，Pillow 做渲染 |

## Technical Approach

### 项目结构

```
puzzle-solver/
├── app/
│   ├── main.py              # FastAPI 入口，/solve 端点
│   ├── segmentation.py      # 碎片分割：OpenCV 轮廓 + Watershed
│   ├── edge_features.py     # 边缘采样：提取四边颜色序列
│   ├── pair_scorer.py       # 配对评分：计算碎片-碎片兼容性矩阵
│   ├── layout_solver.py     # 布局求解：贪心拼接算法
│   └── visualization.py     # 可视化：生成标注图 + 网格图
├── tests/
│   ├── test_segmentation.py
│   ├── test_edge_features.py
│   ├── test_pair_scorer.py
│   └── test_layout_solver.py
├── requirements.txt
└── Dockerfile
```

### Phase 1：碎片分割（`segmentation.py`）

```python
# 输入：BGR 图像
# 输出：List[PieceInfo]
#   PieceInfo.id: int
#   PieceInfo.mask: np.ndarray (H×W bool)
#   PieceInfo.bbox: tuple (x, y, w, h)
#   PieceInfo.contour: np.ndarray
#   PieceInfo.centroid: tuple (cx, cy)

# 流程：
# 1. HSV 颜色过滤（分离背景）→ 前景 mask
# 2. 形态学开闭运算（去噪）
# 3. Watershed 分离粘连碎片
# 4. 过滤面积 < 阈值的噪点区域
# 5. 返回每块碎片的元信息
```

**前置条件（用户需保证）**：
- 碎片放置于单色背景（白色/蓝色桌面）
- 碎片不严重重叠（轻微接触可接受）
- 拍摄角度尽量俯视（减少透视畸变）

### Phase 2：边缘采样（`edge_features.py`）

```python
# 针对每块碎片，沿其凸包/边框的四条逻辑边各均匀采样 N 个像素点
# 使用 Lab 色彩空间（对光照更鲁棒）
# 每条边输出：np.ndarray shape=(N, 3)，dtype=float32

# 边缘识别策略：
# - 用 minAreaRect 确定碎片的主轴方向
# - 将轮廓上的点按角度分为 top/right/bottom/left 四组
# - 各组插值到固定长度 N=64
```

### Phase 3：配对评分（`pair_scorer.py`）

```python
# 对每对 (piece_i, edge_a) <-> (piece_j, edge_b) 计算兼容分数
# 分数 = 颜色 SSD（越小越兼容）× 形状惩罚（凸凹不互补时加惩罚）
#
# 优化：只计算 "碎片i右边 vs 碎片j左边" 等有意义的组合
# 旋转：每块碎片测试 4 种旋转，选最优
#
# 输出：compatibility_matrix[i][j][rot] = score
# 时间复杂度：O(N² × 4)，N=100 时约 40,000 次比较，可接受
```

### Phase 4：布局求解（`layout_solver.py`）

```python
# 贪心策略（参考学术界 "greedy best-first" 拼图求解）：
# 1. 识别角块（4 条边中 2 条为直边的碎片）
# 2. 识别边块（1 条直边）
# 3. 固定一个角块在 (0,0)，以右/下方向扩展
# 4. 用 Minimum Spanning Tree 思路：每次选评分最高的未放置碎片
# 5. 当多块候选得分相近时，使用全局约束剪枝
#
# 输出：List[Placement]
#   Placement.piece_id: int
#   Placement.row: int
#   Placement.col: int
#   Placement.rotation: int  # 0/90/180/270
```

### Phase 5：可视化（`visualization.py`）

**标注图**（左）：
- 在原始散乱图上，用随机颜色为每块碎片绘制轮廓
- 在碎片质心处叠加白色编号标签（背景色块保证可读性）

**网格图**（右）：
- 根据求解结果的行列数生成等大格子网格
- 每个格子内写入对应的碎片编号
- 格子用浅色填充，边框清晰
- 标题注明拼图尺寸（如 "10×12 Grid"）

**拼接输出**：两图并排，中间分隔线，整体返回为单张 JPEG。

### Phase 6：API（`main.py`）

```python
POST /solve
  Content-Type: multipart/form-data
  Body: pieces_image (file)

  Response: image/jpeg
    Body: 并排标注图（标注图 | 网格图）
    Headers:
      X-Pieces-Count: 107
      X-Grid-Size: 9x12
      X-Solve-Time-Ms: 4230

POST /solve/json   # 可选端点，返回结构化数据
  Response: application/json
  {
    "pieces_count": 107,
    "grid": {"rows": 9, "cols": 12},
    "placements": [
      {"piece_id": 1, "row": 0, "col": 0, "rotation": 0},
      ...
    ],
    "unmatched": [42, 78]   // 无法确定位置的碎片 ID
  }

GET /health
  Response: {"status": "ok", "opencv_version": "4.x.x"}
```

## System-Wide Impact

### 处理时间估算（N=100 块）

| 阶段 | 估算时间 |
|------|---------|
| 碎片分割 | 1-3s（取决于图片分辨率） |
| 边缘采样 | < 0.5s |
| 配对评分 | 1-5s（40,000 次比较） |
| 布局求解 | 1-10s（取决于约束复杂度） |
| 可视化 | < 1s |
| **总计** | **3-20s** |

对于 HTTP 同步请求，20s 在可接受边界。如后续需要支持 500+ 块，需引入异步任务队列（Celery + Redis）。

### 已知局限性

- **严重重叠碎片**：Watershed 无法正确分割，返回 HTTP 422 并提示用户重新摆放
- **复杂背景**：颜色过滤失效，建议用户使用白色或蓝色背景
- **非标准拼图**：算法假设拼图碎片有四条逻辑边；圆形/不规则形状拼图不适用
- **低置信度碎片**：最终输出中标记为 `unmatched`，不强行分配位置

## Acceptance Criteria

- [x] `POST /solve` 接受单张图片，返回并排标注图（JPEG）
- [x] 标注图左侧：原图上每块碎片标有唯一编号
- [x] 网格图右侧：每个格子标有对应碎片编号，网格尺寸与拼图行列数一致
- [x] 支持 100 块以上的拼图（不超过 30 秒响应时间）
- [x] 碎片分割失败时返回 HTTP 422 + 明确错误信息
- [x] `/solve/json` 端点返回结构化坐标数据
- [x] 所有核心模块有单元测试（segmentation、edge_features、pair_scorer、layout_solver）
- [x] `GET /health` 返回服务状态

## Dependencies & Risks

### 依赖

```
fastapi==0.115.x
uvicorn[standard]==0.32.x
python-multipart==0.0.12
opencv-contrib-python==4.10.x   # 必须 contrib 版，含 SIFT
numpy>=2.0
Pillow>=10.4
scikit-image>=0.24
scipy>=1.14
```

### 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 碎片分割不准确（背景复杂） | 高 | 高 | 在文档中明确输入约束，422 报错给出提示 |
| 布局求解错误率高（100+ 块） | 中 | 中 | 输出置信度分数，低置信度碎片标记为 unmatched |
| 响应超时（大图 + 多碎片） | 中 | 中 | 先限制图片尺寸上限（如 4000×4000px）；后续可加异步 |
| 碎片图案重复度高（如全天空） | 高 | 高 | 已知局限，在文档中说明；纯色区域碎片会被标记为 unmatched |

## Sources & References

- OpenCV Watershed 分割：https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
- 拼图求解学术参考：Gallagher (2012) "Jigsaw Puzzles with Large Piece Collections"
- SIFT 特征匹配：`opencv-contrib-python` 文档
- FastAPI 文件上传：https://fastapi.tiangolo.com/tutorial/request-files/
