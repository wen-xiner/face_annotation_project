# 人脸标注工具 (Face Annotation Tool)

> 基于 MediaPipe + InsightFace 的自动化人脸标注工具，支持人脸检测、关键点提取、质量评估、人工审核、格式转换的全流程数据标注管线。

## 功能特性

- 🔍 **人脸检测** — 基于 MediaPipe BlazeFace 的高效人脸检测，支持置信度阈值配置
- 📍 **关键点提取** — MediaPipe FaceLandmarker 478 稠密点 → 5 个简化关键点（双眼、鼻尖、双嘴角）
- 🏷️ **自动标注** — 检测 + 关键点一步完成，输出标准化 JSON 标注文件
- ✅ **质量评估** — 模糊度、亮度、姿态三维度图片质量检测，自动拒绝低质量标注
- 📊 **质量报告** — 批量评估标注质量，生成统计分布报告
- 👁️ **人工审核** — 交互式命令行审核工具，支持修改 bbox/关键点，记录修改历史
- 🎨 **可视化** — 在图片上绘制检测框和关键点，保存标注效果图
- 🔄 **格式转换** — 支持导出 COCO JSON、YOLO txt、Pascal VOC XML 等主流标注格式
- 🌐 **Web 界面** — FastAPI + HTML 前端，支持上传、拍照、标注、下载、人物库管理

## 环境要求

- **Python**: ≥ 3.9
- **操作系统**: Windows / Linux / macOS
- **模型文件**: 需下载 MediaPipe 模型（blaze_face_short_range.tflite, face_landmarker.task）
- **可选**: InsightFace（用于人脸识别与人物库功能）

## 安装步骤

### 1. 克隆项目

```bash
git clone <repo-url>
cd face_annotation_project
```

### 2. 安装核心依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型文件

将以下模型文件放入 `models/` 目录：

- `blaze_face_short_range.tflite` — [MediaPipe Face Detector](https://developers.google.com/mediapipe/solutions/vision/face_detector)
- `face_landmarker.task` — [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)

### 4. （可选）安装 InsightFace 人脸识别模块

```bash
pip install insightface onnxruntime
```

首次运行时会自动下载 `buffalo_l` 模型包。

### 5. （可选）安装 Web 界面依赖

```bash
pip install -r web_app/requirements.txt
```

## 快速开始

### 命令行 — 批量标注

```bash
# 对 dataset/raw_images/ 下所有图片执行批量标注
python -m src.batch_processor

# 仅处理指定人物
python -m src.batch_processor --person 温惠锦

# 启用质量自动过滤
python -c "
from src.annotation_generator import AnnotationGenerator
gen = AnnotationGenerator(auto_reject=True, quality_threshold=0.4)
# 低质量图片（模糊/过暗/倾斜过大）将被自动拒绝
"
```

### 命令行 — 质量评估

```bash
# 对已有标注执行质量评估，生成报告
python -m src.quality_report output/annotations/auto_labeled/

# 同时保存 JSON 详细报告
python -m src.quality_report output/annotations/auto_labeled/ --json output/quality_report.json
```

### 命令行 — 标注校验

```bash
# 校验单个标注文件
python -m src.validator output/annotations/auto_labeled/xxx_annotation.json

# 批量校验整个目录
python -m src.validator output/annotations/auto_labeled/

# 严格模式（警告也视为错误）
python -m src.validator output/annotations/auto_labeled/ --strict
```

### 命令行 — 人工审核

```bash
# 列出待审核文件
python -m src.manual_reviewer list

# 交互式审核单个文件
python -m src.manual_reviewer review output/annotations/auto_labeled/xxx_annotation.json

# 批量批准所有自动标注（信任模式）
python -m src.manual_reviewer batch
```

### 命令行 — 格式转换

```bash
# 导出 COCO 格式
python -m src.format_converter output/annotations/auto_labeled/ --format coco --output output/exports/coco.json

# 导出 YOLO 格式
python -m src.format_converter output/annotations/auto_labeled/ --format yolo --output output/exports/yolo/

# 导出 Pascal VOC 格式
python -m src.format_converter output/annotations/auto_labeled/ --format voc --output output/exports/voc/
```

### 命令行 — 数据集整理

```bash
# 智能分组（基于人脸特征聚类）
python -m src.dataset_organizer

# 按文件名分组
python -m src.dataset_organizer --by-name

# 预览模式（不实际移动文件）
python -m src.dataset_organizer --dry-run
```

### Web 界面

```bash
# 启动 Web 服务
python web_app/app.py

# 浏览器访问 http://localhost:8000
```

Web 界面功能：
1. 拖拽上传图片或使用摄像头拍照
2. 点击"标注"按钮执行自动检测+关键点提取
3. 查看可视化标注效果
4. 标注结果自动与人物库比对（人脸识别）
5. 为新人物命名注册
6. 下载标注 JSON 或可视化图片
7. 批量导出 ZIP

## 数据格式说明

### 标注 JSON 结构

每个标注文件以 `{图片hash}_annotation.json` 命名，包含以下字段：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `image_path` | string | ✅ | 被标注图片的相对路径（正斜杠，相对于项目根目录） |
| `bbox` | object | ✅ | 人脸边界框 `{"x": int, "y": int, "width": int, "height": int}` |
| `landmarks` | object | ✅ | 5 个简化关键点坐标（详见下方） |
| `annotator` | string | ✅ | 标注者：`"auto"` / `"auto_partial"` / `"manual"` |
| `date` | string | ✅ | 标注日期，格式 `YYYY-MM-DD` |
| `confidence` | number | 否 | 人脸检测置信度 (0-1) |
| `quality` | object | 否 | 图片质量评估结果 |
| `reviewer` | string | 否 | 审核者标识（仅人工修正后） |
| `reviewed_from` | string | 否 | 原始标注来源（仅人工修正后） |
| `modifications` | array | 否 | 修改记录列表（仅人工修正后） |

### landmarks 关键点定义

| 关键点名称 | 说明 | 观察者视角 |
|-----------|------|-----------|
| `left_eye` | 左眼中心 | 图像右侧的眼睛 |
| `right_eye` | 右眼中心 | 图像左侧的眼睛 |
| `nose_tip` | 鼻尖 | — |
| `left_mouth_corner` | 左嘴角 | 图像右侧的嘴角 |
| `right_mouth_corner` | 右嘴角 | 图像左侧的嘴角 |

> ⚠️ **重要**：左/右采用**观察者视角**（看图人的左右），非被拍摄者自身的左右。

每个关键点的值为 `[x, y]` 整数像素坐标，坐标系原点为图片左上角。
当关键点提取失败时，该点值为 `[-1, -1]`（哨兵值），此时 `annotator` 标记为 `"auto_partial"`。

### 标注示例

```json
{
  "image_path": "web_app/uploads/4a948fbcca8a.jpg",
  "bbox": {
    "x": 325,
    "y": 460,
    "width": 656,
    "height": 656
  },
  "landmarks": {
    "left_eye": [493, 628],
    "right_eye": [771, 627],
    "nose_tip": [635, 807],
    "left_mouth_corner": [528, 922],
    "right_mouth_corner": [748, 921]
  },
  "annotator": "auto",
  "date": "2026-06-24",
  "confidence": 0.9406
}
```

完整的数据格式规范见 `annotation.schema.json`（JSON Schema Draft 2020-12）。

## 输出目录结构

```
output/
├── annotations/
│   ├── auto_labeled/              # 自动标注 JSON 文件
│   │   ├── 4a948fbcca8a_annotation.json
│   │   ├── a109993f2e72_annotation.json
│   │   └── ...
│   └── manually_reviewed/         # 人工审核后的标注 JSON 文件
│       └── ...
├── persons/                       # 人物库（人脸识别用）
│   ├── index.json                 # 人物索引（姓名、图片数、注册日期）
│   ├── 温惠锦/
│   │   └── reference.npy          # ArcFace 512 维人脸特征向量
│   └── ...
└── visualizations/                # 标注可视化图片
    └── ...
```

## 模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| `FaceDetector` | `src/face_detector.py` | MediaPipe BlazeFace 人脸检测，支持单张/多人脸 |
| `LandmarkExtractor` | `src/landmark_extractor.py` | MediaPipe FaceLandmarker 478点 → 5个简化关键点 |
| `AnnotationGenerator` | `src/annotation_generator.py` | 整合检测+提取+质量评估，生成标注 JSON |
| `BatchProcessor` | `src/batch_processor.py` | 遍历图片目录，批量执行标注 |
| `ManualReviewer` | `src/manual_reviewer.py` | 交互式人工审核，修改记录追踪 |
| `QualityFilter` | `src/quality_filter.py` | 模糊/亮度/姿态三维质量检测 |
| `AnnotationValidator` | `src/validator.py` | 标注 JSON 结构校验 + 质量评估 |
| `QualityReport` | `src/quality_report.py` | 批量质量评估报告生成 |
| `FormatConverter` | `src/format_converter.py` | COCO / YOLO / Pascal VOC 格式导出 |
| `Visualizer` | `src/visualizer.py` | bbox + 关键点可视化绘制 |
| `FaceMatcher` | `src/face_signature.py` | ArcFace 人脸特征提取、比对、聚类 |
| `DatasetOrganizer` | `src/dataset_organizer.py` | 数据集整理：智能分组、去重、路径更新 |
| `utils` | `src/utils.py` | 共享常量、JSON 读写、路径处理 |

## 项目依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| opencv-python | ≥4.8.0, <5.0.0 | 图像加载、色彩转换、质量检测 |
| mediapipe | ≥0.10.0, <0.11.0 | 人脸检测 + 关键点提取 |
| numpy | ≥1.24.0, <2.0.0 | 数值计算 |
| tqdm | ≥4.65.0 | 进度条显示 |
| insightface | (可选) | 人脸识别（ArcFace 512维embedding） |
| onnxruntime | (可选) | insightface 推理后端 |
| fastapi | ≥0.110.0 (Web) | Web 框架 |
| uvicorn | ≥0.29.0 (Web) | ASGI 服务器 |

## 规范文档

- [数据采集规范](docs/data_collection_spec.md) — 数据来源、采集环境、设备要求
- [标注规范](docs/annotation_spec.md) — 关键点精确定义、标注流程、常见问题
- [JSON Schema](annotation.schema.json) — 标注数据格式的形式化定义

## 许可证

本项目仅用于教育和研究目的。使用本工具处理的人脸数据应遵守相关隐私法规。

## 项目截图

![标注可视化](项目截图/annotation_visualization.png)
