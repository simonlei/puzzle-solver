# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# 安装依赖
pip3 install -r requirements.txt

# 启动服务（开发模式）
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 运行所有测试
python3 -m pytest tests/ -v

# 运行单个测试文件
python3 -m pytest tests/test_segmentation.py -v

# 运行单个测试用例
python3 -m pytest tests/test_layout_solver.py::TestSolveLayout::test_no_duplicate_positions -v

# 调用 API（返回标注图）
curl -X POST "http://localhost:8000/solve" \
  -F "pieces_image=@your_puzzle.jpg" \
  --output result.jpg

# Docker 构建和运行
docker build -t puzzle-solver .
docker run -p 8000:8000 puzzle-solver
```

## Architecture

六阶段流水线，每个阶段对应一个模块：

```
散乱碎片图 → segmentation → edge_features → pair_scorer → layout_solver → visualization → FastAPI 响应
```

### 模块职责

- **`app/segmentation.py`**：OpenCV Watershed 分割，输出 `PieceInfo` 列表（mask、bbox、轮廓、质心）
- **`app/edge_features.py`**：沿每块碎片的四条逻辑边采样颜色序列（Lab 色彩空间，64 点/边），检测直边（角块/边块识别）
- **`app/pair_scorer.py`**：计算所有碎片对的边缘兼容性分数（颜色 SSD），支持 4 种旋转；定义 `OPPOSITE` 和 `_ROTATION_MAP` 常量（被 `layout_solver` 导入）
- **`app/layout_solver.py`**：贪心 BFS 布局求解，从角块出发向右/下扩展，输出每块碎片的 `(row, col, rotation)`；未能定位的碎片标记为 `unmatched`
- **`app/visualization.py`**：生成并排输出图——左侧标注图（原图+编号轮廓），右侧网格图（格子+编号）
- **`app/main.py`**：FastAPI 入口，三个端点：`POST /solve`（返回图片）、`POST /solve/json`（返回 JSON）、`GET /health`

### 关键数据类型

- `PieceInfo`：碎片的物理信息（来自图像分割）
- `EdgeFeatures`：碎片四边的颜色特征 + 直边标志（`is_corner`、`is_border`、`is_interior`）
- `PairScore`：两块碎片某对边的兼容性分数
- `Placement`：碎片在网格中的最终位置 `(piece_id, row, col, rotation, confidence)`

### 输入约束

碎片分割依赖背景对比度，用户需保证：
- 单色背景（白色或蓝色桌面效果最佳）
- 碎片不严重重叠（轻微接触可接受）
- 拍摄角度尽量俯视

图片最大尺寸限制为 4000×4000px（超过则自动等比缩放）。
