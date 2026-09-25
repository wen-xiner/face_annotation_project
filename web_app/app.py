"""
文件名: app.py
功能: 人脸标注工具 Web 界面 — FastAPI 主应用
      - 图片上传 / 摄像头拍照
      - 自动标注 (bbox + landmarks)
      - 人脸识别 (ArcFace 比对 + 自动命名归类)
      - 可视化预览、JSON 下载、数据校验
"""

import os
import sys

# 抑制 TensorFlow oneDNN 日志
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import logging
import shutil
import tempfile
import uuid
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation_generator import AnnotationGenerator
from src.face_detector import FaceDetector
from src.landmark_extractor import LandmarkExtractor
from src.annotation_corrector import AnnotationCorrector
from src.quality_filter import QualityFilter
from src.format_converter import FormatConverter
from src.image_preprocessor import create_face_cleaning_pipeline, ImagePreprocessor
from src.quality_report import QualityReport
from src.utils import SUPPORTED_IMAGE_EXTENSIONS, build_annotation_dict, load_image
from src.validator import AnnotationValidator, AnnotationQualityAssessor
from src.visualizer import Visualizer

# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# FastAPI 初始化
# ============================================================
app = FastAPI(
    title="Face Annotator",
    description="人脸标注工具 — Precision Lab",
    version="2.0.0",
)

WEB_APP_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(WEB_APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEB_APP_DIR / "templates"))

# 目录
UPLOAD_DIR = WEB_APP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = PROJECT_ROOT / "output" / "annotations" / "auto_labeled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIZ_DIR = PROJECT_ROOT / "output" / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# 人物库目录
PERSONS_DIR = PROJECT_ROOT / "output" / "persons"
PERSONS_DIR.mkdir(parents=True, exist_ok=True)
PERSON_INDEX_FILE = PERSONS_DIR / "index.json"

# 图片存储根目录（按人物名分文件夹）
DATASET_DIR = PROJECT_ROOT / "dataset" / "raw_images"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# 全局引擎
detector: Optional[FaceDetector] = None
extractor: Optional[LandmarkExtractor] = None
visualizer: Optional[Visualizer] = None
validator: Optional[AnnotationValidator] = None
quality_filter: Optional[QualityFilter] = None
format_converter: Optional[FormatConverter] = None
quality_assessor: Optional[AnnotationQualityAssessor] = None
corrector: Optional[AnnotationCorrector] = None
cleaning_pipeline = None  # PreprocessPipeline 单例
face_app = None  # insightface FaceAnalysis 单例

# 任务状态
task_store: Dict[str, dict] = {}


# ============================================================
# 工具函数
# ============================================================

def get_detector() -> FaceDetector:
    global detector
    if detector is None:
        detector = FaceDetector()
    return detector


def get_extractor() -> LandmarkExtractor:
    global extractor
    if extractor is None:
        extractor = LandmarkExtractor()
    return extractor


def get_visualizer() -> Visualizer:
    global visualizer
    if visualizer is None:
        visualizer = Visualizer()
    return visualizer


def get_validator() -> AnnotationValidator:
    global validator
    if validator is None:
        validator = AnnotationValidator()
    return validator


def get_quality_filter() -> QualityFilter:
    global quality_filter
    if quality_filter is None:
        quality_filter = QualityFilter()
    return quality_filter


def get_format_converter() -> FormatConverter:
    global format_converter
    if format_converter is None:
        format_converter = FormatConverter()
    return format_converter


def get_quality_assessor() -> AnnotationQualityAssessor:
    global quality_assessor
    if quality_assessor is None:
        quality_assessor = AnnotationQualityAssessor()
    return quality_assessor


def get_corrector() -> AnnotationCorrector:
    global corrector
    if corrector is None:
        corrector = AnnotationCorrector()
    return corrector


def get_cleaning_pipeline():
    global cleaning_pipeline
    if cleaning_pipeline is None:
        cleaning_pipeline = create_face_cleaning_pipeline()
    return cleaning_pipeline


def get_face_app():
    """获取 insightface FaceAnalysis 单例。"""
    global face_app
    if face_app is None:
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=0)
    return face_app


def extract_face_embedding(image: np.ndarray, bbox: Optional[Dict[str, int]] = None) -> Optional[np.ndarray]:
    """
    提取人脸 512 维 ArcFace embedding。

    如果提供 bbox（MediaPipe 检测结果），先裁剪人脸区域再送入 insightface，
    避免对大图做 multi-scale 检测，速度提升 10-50 倍。

    Args:
        image: BGR 图片数组
        bbox: MediaPipe 检测的人脸边界框 {"x","y","width","height"}

    Returns:
        Optional[np.ndarray]: 512 维归一化 embedding
    """
    try:
        app = get_face_app()

        if bbox is not None:
            # 用 MediaPipe bbox 裁剪人脸，跳过 insightface 的重复检测
            h, w = image.shape[:2]
            x, y, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            margin = 0.3
            pad_x = int(bw * margin)
            pad_y = int(bh * margin)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            if x2 > x1 and y2 > y1:
                image = image[y1:y2, x1:x2]

        faces = app.get(image)
        if not faces:
            return None
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = best.embedding
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb
    except Exception as e:
        logger.warning(f"Embedding extraction failed: {e}")
        return None


def load_person_index() -> Dict[str, dict]:
    """加载人物库索引。"""
    if PERSON_INDEX_FILE.exists():
        with open(PERSON_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_person_index(index: Dict[str, dict]):
    """保存人物库索引。"""
    with open(PERSON_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def recognize_person(embedding: np.ndarray, threshold: float = 0.45) -> Optional[str]:
    """
    将 embedding 与人物库中所有人比对，返回匹配的人物名。

    Args:
        embedding: 512 维归一化 ArcFace embedding
        threshold: 余弦相似度阈值

    Returns:
        Optional[str]: 匹配的人物名，无匹配返回 None
    """
    index = load_person_index()
    if not index:
        return None

    best_name = None
    best_score = 0.0

    for name, info in index.items():
        ref_path = PERSONS_DIR / name / "reference.npy"
        if not ref_path.exists():
            continue
        ref_emb = np.load(ref_path)
        score = float(np.dot(embedding, ref_emb))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        logger.info(f"Recognized: {best_name} (score={best_score:.3f})")
        return best_name

    return None


def register_person(name: str, embedding: np.ndarray, image_path: Optional[Path] = None):
    """
    注册新人物或更新已有人的 reference embedding。
    如果提供 image_path，自动将图片复制到 dataset/raw_images/{name}/ 下。
    """
    person_dir = PERSONS_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    ref_path = person_dir / "reference.npy"

    if ref_path.exists():
        # 已有 → 取新旧平均
        old_emb = np.load(ref_path)
        new_emb = (old_emb + embedding) / 2.0
    else:
        new_emb = embedding

    np.save(ref_path, new_emb)

    # 确保图片存储文件夹存在
    img_folder = DATASET_DIR / name
    img_folder.mkdir(parents=True, exist_ok=True)

    # 复制图片到人物文件夹
    if image_path is not None and image_path.exists():
        dest = img_folder / image_path.name
        if not dest.exists():
            shutil.copy2(str(image_path), str(dest))
            logger.info(f"Image saved: {dest}")

    # 更新索引
    index = load_person_index()
    index[name] = {
        "folder": str(img_folder),
        "created_at": index.get(name, {}).get("created_at", date.today().isoformat()),
        "image_count": index.get(name, {}).get("image_count", 0) + 1,
    }
    save_person_index(index)

    logger.info(f"Registered: {name} (total images: {index[name]['image_count']})")


# ============================================================
# 页面路由
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ============================================================
# 图片上传
# ============================================================

@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """上传一张或多张图片。"""
    results = []
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in allowed:
            results.append({
                "file_id": None, "filename": file.filename,
                "status": "rejected", "error": f"不支持格式: {suffix}",
            })
            continue

        file_id = uuid.uuid4().hex[:12]
        safe_name = f"{file_id}{suffix}"
        file_path = UPLOAD_DIR / safe_name

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        task_store[file_id] = {
            "file_id": file_id,
            "original_name": file.filename,
            "file_path": str(file_path),
            "status": "uploaded",
            "person_name": None,
            "annotation": None,
            "annotation_path": None,
            "viz_path": None,
            "is_new_face": False,
            "error": None,
        }

        logger.info(f"Uploaded: {file.filename} -> {file_id}")
        results.append({
            "file_id": file_id, "filename": file.filename,
            "status": "uploaded", "size_bytes": len(content),
        })

    return {"files": results, "total": len(results)}


# ============================================================
# 摄像头拍照上传
# ============================================================

@app.post("/api/capture")
async def capture_image(file: UploadFile = File(...)):
    """接收摄像头拍照的图片（Base64 编码的 JPEG）。"""
    file_id = uuid.uuid4().hex[:12]
    safe_name = f"{file_id}.jpg"
    file_path = UPLOAD_DIR / safe_name

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    task_store[file_id] = {
        "file_id": file_id,
        "original_name": f"camera_{file_id[:6]}.jpg",
        "file_path": str(file_path),
        "status": "uploaded",
        "person_name": None,
        "annotation": None,
        "annotation_path": None,
        "viz_path": None,
        "is_new_face": False,
        "error": None,
    }

    logger.info(f"Camera capture: {file_id}")
    return {
        "file_id": file_id,
        "filename": f"camera_{file_id[:6]}.jpg",
        "status": "uploaded",
        "size_bytes": len(content),
    }


# ============================================================
# 视频帧实时检测（轻量，只检测 + 识别，不做标注）
# ============================================================

@app.post("/api/detect-frame")
async def detect_frame(file: UploadFile = File(...)):
    """
    接收视频帧（JPEG），快速返回人脸检测 + 识别结果。
    只做检测 + embedding + 比对，不保存标注文件。

    返回:
        faces: [{bbox: {x,y,width,height}, name: str|null, confidence: float, is_new: bool}]
    """
    content = await file.read()
    img_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"faces": []}

    try:
        det = get_detector()
        all_detections = det.detect_all(image)

        if not all_detections:
            return {"faces": []}

        app = get_face_app()
        rec_model = app.models.get('recognition')  # 只用识别模型，跳过检测
        faces_result = []

        for d in all_detections[:3]:
            bbox = {"x": d["x"], "y": d["y"], "width": d["width"], "height": d["height"]}

            name = None
            is_new = False
            recog_conf = 0.0

            h, w = image.shape[:2]
            margin = 0.2
            x1 = max(0, int(d["x"] - d["width"] * margin))
            y1 = max(0, int(d["y"] - d["height"] * margin))
            x2 = min(w, int(d["x"] + d["width"] * (1 + margin)))
            y2 = min(h, int(d["y"] + d["height"] * (1 + margin)))

            if x2 > x1 and y2 > y1 and rec_model is not None:
                face_crop = image[y1:y2, x1:x2]
                # resize 到 112x112 直接送入识别模型（跳过 insightface 检测+关键点+性别年龄）
                face_aligned = cv2.resize(face_crop, (112, 112))
                emb = rec_model.get_feat(face_aligned)
                if emb is not None and len(emb) > 0:
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                    matched = recognize_person(emb, threshold=0.45)
                    if matched:
                        name = matched
                        ref_path = PERSONS_DIR / matched / "reference.npy"
                        if ref_path.exists():
                            recog_conf = float(np.dot(emb, np.load(ref_path)))
                    else:
                        is_new = True

            faces_result.append({
                "bbox": bbox,
                "name": name,
                "confidence": round(recog_conf, 3),
                "is_new": is_new,
            })

        return {"faces": faces_result}

    except Exception as e:
        logger.warning(f"Frame detection failed: {e}")
        return {"faces": []}


# ============================================================
# 标注 + 识别
# ============================================================

@app.post("/api/annotate/{file_id}")
async def annotate_image(file_id: str):
    """
    对已上传图片执行：人脸检测 → 关键点提取 → 人脸识别 → 自动归类。
    """
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")

    task = task_store[file_id]
    file_path = Path(task["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件已被清理")

    try:
        task["status"] = "processing"
        image = load_image(file_path)

        # 步骤 1：人脸检测
        det = get_detector()
        all_detections = det.detect_all(image)
        if not all_detections:
            task["status"] = "failed"
            task["error"] = "未检测到人脸"
            return {
                "file_id": file_id, "status": "failed",
                "error": "未检测到人脸",
                "bbox": None, "landmarks": None,
                "person_name": None, "is_new_face": False,
            }

        best = all_detections[0]
        bbox = {"x": best["x"], "y": best["y"], "width": best["width"], "height": best["height"]}
        confidence = best.get("confidence", None)

        # 步骤 2：关键点提取
        ext = get_extractor()
        landmarks = ext.extract(image, bbox=bbox)

        if landmarks is not None:
            annotator = "auto"
        else:
            annotator = "auto_partial"
            landmarks = {
                "left_eye": [-1, -1], "right_eye": [-1, -1],
                "nose_tip": [-1, -1],
                "left_mouth_corner": [-1, -1], "right_mouth_corner": [-1, -1],
            }

        # 步骤 3：构建标注
        annotation = build_annotation_dict(
            image_path=file_path, bbox=bbox, landmarks=landmarks,
            annotator=annotator, confidence=confidence,
        )

        # 步骤 3.5：质量评估
        try:
            qf = get_quality_filter()
            quality = qf.assess(image, landmarks if landmarks else None)
            annotation["quality"] = quality
            logger.info(
                f"Quality: {file_path.name} -> score={quality['quality_score']:.2f} "
                f"label={quality['quality_label']}"
            )
        except Exception as e:
            logger.warning(f"质量评估失败: {file_path.name} - {e}")

        # 保存 JSON
        output_path = OUTPUT_DIR / f"{file_id}_annotation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

        # 可视化
        vis = get_visualizer()
        viz_image = vis.draw_annotation(file_path, output_path)
        viz_path = VIZ_DIR / f"{file_id}_annotated.jpg"
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(viz_path), viz_image)

        # 步骤 4：人脸识别（传入 bbox 裁剪，加速 10-50x）
        embedding = extract_face_embedding(image, bbox=bbox)
        person_name = None
        is_new_face = False

        if embedding is not None:
            matched = recognize_person(embedding)
            if matched:
                person_name = matched
                register_person(matched, embedding, image_path=file_path)
            else:
                is_new_face = True

        task.update({
            "status": "completed",
            "annotation": annotation,
            "annotation_path": str(output_path),
            "viz_path": str(viz_path),
            "person_name": person_name,
            "is_new_face": is_new_face,
            "error": None,
        })

        logger.info(f"Annotated: {file_path.name} → person={person_name or 'NEW'}")

        return {
            "file_id": file_id,
            "status": "completed",
            "annotation": annotation,
            "image_url": f"/api/image/{file_id}",
            "viz_url": f"/api/viz/{file_id}",
            "person_name": person_name,
            "is_new_face": is_new_face,
        }

    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        logger.error(f"Annotation failed: {e}")
        raise HTTPException(status_code=500, detail=f"标注失败: {e}")


# ============================================================
# 人物命名
# ============================================================

@app.post("/api/person/name")
async def name_person(file_id: str = Form(...), name: str = Form(...)):
    """
    为新识别的人脸命名，注册到人物库。
    之后该人物的照片会自动归类到此名称。
    """
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")

    task = task_store[file_id]
    file_path = Path(task["file_path"])

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件已被清理")

    # 提取 embedding 并注册
    image = load_image(file_path)
    embedding = extract_face_embedding(image)

    if embedding is None:
        raise HTTPException(status_code=400, detail="无法提取人脸特征")

    register_person(name, embedding, image_path=file_path)

    task["person_name"] = name
    task["is_new_face"] = False

    logger.info(f"Named: {file_id} -> {name}")

    return {
        "file_id": file_id,
        "person_name": name,
        "status": "named",
    }


@app.get("/api/persons")
async def list_persons():
    """列出所有已注册的人物。"""
    index = load_person_index()
    persons = []
    for name, info in index.items():
        persons.append({
            "name": name,
            "image_count": info.get("image_count", 0),
            "created_at": info.get("created_at", ""),
        })
    return {"persons": sorted(persons, key=lambda p: p["name"])}


@app.delete("/api/person/{name}")
async def delete_person(name: str):
    """删除一个人物及其所有关联数据。"""
    index = load_person_index()
    if name not in index:
        raise HTTPException(status_code=404, detail="人物不存在")

    # 删除人物目录
    person_dir = PERSONS_DIR / name
    if person_dir.exists():
        shutil.rmtree(person_dir)

    # 删除索引
    del index[name]
    save_person_index(index)

    logger.info(f"Deleted person: {name}")
    return {"status": "deleted", "name": name}


# ============================================================
# 数据获取
# ============================================================

@app.get("/api/image/{file_id}")
async def get_image(file_id: str):
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")
    file_path = Path(task_store[file_id]["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件已被清理")
    return FileResponse(file_path, media_type="image/jpeg")


@app.get("/api/viz/{file_id}")
async def get_viz_image(file_id: str):
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")
    task = task_store[file_id]
    viz_path = task.get("viz_path")
    if not viz_path or not Path(viz_path).exists():
        raise HTTPException(status_code=404, detail="标注结果不存在")
    return FileResponse(viz_path, media_type="image/jpeg")


@app.get("/api/annotation/{file_id}")
async def get_annotation(file_id: str):
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")
    task = task_store[file_id]
    if task["status"] != "completed" or not task.get("annotation"):
        raise HTTPException(status_code=404, detail="标注尚未完成")
    return task["annotation"]


@app.get("/api/tasks")
async def get_tasks():
    tasks = []
    for file_id, task in task_store.items():
        tasks.append({
            "file_id": task["file_id"],
            "original_name": task["original_name"],
            "status": task["status"],
            "person_name": task.get("person_name"),
            "is_new_face": task.get("is_new_face", False),
            "error": task.get("error"),
        })
    return {"tasks": tasks, "total": len(tasks)}


# ============================================================
# 下载
# ============================================================

@app.get("/api/download/{file_id}")
async def download_annotation(file_id: str):
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")
    task = task_store[file_id]
    ann_path = task.get("annotation_path")
    if not ann_path or not Path(ann_path).exists():
        raise HTTPException(status_code=404, detail="标注文件不存在")
    download_name = f"{Path(task['original_name']).stem}_annotation.json"
    return FileResponse(ann_path, media_type="application/json", filename=download_name)


@app.get("/api/download-all")
async def download_all():
    completed = [t for t in task_store.values()
                 if t["status"] == "completed" and t.get("annotation_path")]
    if not completed:
        raise HTTPException(status_code=404, detail="没有可下载的文件")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for task in completed:
            ap = Path(task["annotation_path"])
            if ap.exists():
                zf.write(ap, f"{Path(task['original_name']).stem}_annotation.json")
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=annotations_{date.today().isoformat()}.zip"},
    )


@app.get("/api/download-viz/{file_id}")
async def download_viz(file_id: str):
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")
    task = task_store[file_id]
    viz_path = task.get("viz_path")
    if not viz_path or not Path(viz_path).exists():
        raise HTTPException(status_code=404, detail="可视化图片不存在")
    download_name = f"{Path(task['original_name']).stem}_annotated.jpg"
    return FileResponse(viz_path, media_type="image/jpeg", filename=download_name)


# ============================================================
# 校验
# ============================================================

@app.post("/api/validate/{file_id}")
async def validate_annotation(file_id: str):
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")
    task = task_store[file_id]
    ann_path = task.get("annotation_path")
    if not ann_path or not Path(ann_path).exists():
        raise HTTPException(status_code=404, detail="标注文件不存在")
    val = get_validator()
    return val.validate_file(ann_path)


# ============================================================
# 质量评估
# ============================================================

@app.post("/api/quality/{file_id}")
async def assess_quality(file_id: str):
    """
    对已标注的文件进行质量评估（图片模糊度/亮度/姿态 + 标注合理性）。
    """
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")
    task = task_store[file_id]
    ann_path = task.get("annotation_path")
    if not ann_path or not Path(ann_path).exists():
        raise HTTPException(status_code=404, detail="标注文件不存在")

    qa = get_quality_assessor()
    result = qa.assess_file(ann_path)
    return result


@app.get("/api/quality-report")
async def quality_report_all():
    """
    对所有已完成的标注进行质量评估并生成报告。
    """
    completed = [
        t for t in task_store.values()
        if t["status"] == "completed" and t.get("annotation_path")
    ]
    if not completed:
        raise HTTPException(status_code=404, detail="没有已标注的文件")

    reporter = QualityReport()
    # 收集所有标注文件路径
    ann_paths = [Path(t["annotation_path"]) for t in completed if Path(t["annotation_path"]).exists()]

    if not ann_paths:
        raise HTTPException(status_code=404, detail="标注文件不存在")

    # 逐个评估
    qa = get_quality_assessor()
    results = []
    for ap in ann_paths:
        try:
            r = qa.assess_file(ap)
            results.append(r)
        except Exception as e:
            results.append({"file": str(ap), "status": "error", "error": str(e)})

    summary = QualityReport()._build_summary(results)
    return {
        "total": len(results),
        "summary": summary,
        "details": results,
    }


# ============================================================
# 格式转换导出
# ============================================================

@app.get("/api/export/{file_id}")
async def export_single(file_id: str, format: str = "coco"):
    """
    将单个标注文件导出为 COCO / YOLO / Pascal VOC 格式。

    Query params:
        format: "coco" | "yolo" | "voc"
    """
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")
    task = task_store[file_id]
    ann_path = task.get("annotation_path")
    if not ann_path or not Path(ann_path).exists():
        raise HTTPException(status_code=404, detail="标注文件不存在")

    format_lower = format.lower()
    if format_lower not in ("coco", "yolo", "voc", "csv"):
        raise HTTPException(status_code=400, detail=f"不支持的格式: {format}，可选: coco, yolo, voc, csv")

    fc = get_format_converter()

    with tempfile.TemporaryDirectory() as tmpdir:
        if format_lower == "coco":
            tmp_path = Path(tmpdir) / "export.json"
            fc.to_coco([Path(ann_path)], tmp_path)
            return FileResponse(
                str(tmp_path),
                media_type="application/json",
                filename=f"{task['original_name'].split('.')[0]}_coco.json",
            )
        elif format_lower == "yolo":
            files = fc.to_yolo([Path(ann_path)], tmpdir)
            if not files:
                raise HTTPException(status_code=500, detail="转换失败")
            return FileResponse(
                files[0],
                filename=files[0].name,
                media_type="text/plain",
            )
        elif format_lower == "voc":
            files = fc.to_voc([Path(ann_path)], tmpdir)
            if not files:
                raise HTTPException(status_code=500, detail="转换失败")
            return FileResponse(
                files[0],
                filename=files[0].name,
                media_type="application/xml",
            )
        else:  # csv
            csv_path = Path(tmpdir) / "export.csv"
            fc.to_csv([Path(ann_path)], csv_path)
            return FileResponse(
                str(csv_path),
                media_type="text/csv",
                filename=f"{task['original_name'].split('.')[0]}.csv",
            )


@app.get("/api/export-all")
async def export_all(format: str = "coco"):
    """
    批量导出所有已完成的标注为 COCO / YOLO / Pascal VOC 格式。

    Query params:
        format: "coco" | "yolo" | "voc"
    """
    format_lower = format.lower()
    if format_lower not in ("coco", "yolo", "voc", "csv"):
        raise HTTPException(status_code=400, detail=f"不支持的格式: {format}，可选: coco, yolo, voc, csv")

    completed = [
        t for t in task_store.values()
        if t["status"] == "completed" and t.get("annotation_path")
    ]
    if not completed:
        raise HTTPException(status_code=404, detail="没有可导出的文件")

    ann_paths = [Path(t["annotation_path"]) for t in completed if Path(t["annotation_path"]).exists()]
    if not ann_paths:
        raise HTTPException(status_code=404, detail="标注文件不存在")

    fc = get_format_converter()

    import tempfile
    if format_lower == "coco":
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            fc.to_coco(ann_paths, tmp.name)
            tmp_path = tmp.name
        return FileResponse(
            tmp_path,
            media_type="application/json",
            filename=f"annotations_coco_{date.today().isoformat()}.json",
        )
    elif format_lower == "csv":
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            fc.to_csv(ann_paths, tmp.name)
            tmp_path = tmp.name
        return FileResponse(
            tmp_path,
            media_type="text/csv",
            filename=f"annotations_{date.today().isoformat()}.csv",
        )
    else:
        # YOLO/VOC: 打包为 ZIP
        with tempfile.TemporaryDirectory() as tmpdir:
            if format_lower == "yolo":
                fc.to_yolo(ann_paths, tmpdir)
            else:
                fc.to_voc(ann_paths, tmpdir)

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in Path(tmpdir).iterdir():
                    if f.is_file():
                        zf.write(f, f.name)
            zip_buffer.seek(0)
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename=annotations_{format_lower}_{date.today().isoformat()}.zip"
                },
            )


# ============================================================
# 图片预处理
# ============================================================

@app.post("/api/preprocess/{file_id}")
async def preprocess_image(file_id: str):
    """
    对已上传图片执行预处理清洗（双边滤波 + CLAHE 亮度增强）。
    """
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")
    task = task_store[file_id]
    file_path = Path(task["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件已被清理")

    try:
        image = load_image(file_path)
        pipe = get_cleaning_pipeline()
        cleaned = pipe.apply(image)

        # 覆盖保存
        cv2.imwrite(str(file_path), cleaned)
        logger.info(f"Preprocessed: {file_path.name}")

        return {
            "file_id": file_id,
            "status": "cleaned",
            "steps": pipe.steps_description,
        }
    except Exception as e:
        logger.error(f"Preprocess failed: {e}")
        raise HTTPException(status_code=500, detail=f"预处理失败: {e}")


# ============================================================
# 标注自动纠错
# ============================================================

@app.post("/api/correct/{file_id}")
async def correct_annotation(file_id: str):
    """
    对已标注文件的关键点执行自动纠错（眼对称/鼻尖位置/嘴角对齐/bbox约束）。
    """
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")
    task = task_store[file_id]
    ann_path = task.get("annotation_path")
    if not ann_path or not Path(ann_path).exists():
        raise HTTPException(status_code=404, detail="标注文件不存在")

    try:
        corr = get_corrector()
        result = corr.correct_file(ann_path, output_path=ann_path, save=True)

        # 更新内存中的标注
        task["annotation"] = result["corrected"]

        return {
            "file_id": file_id,
            "corrections": result["corrections"],
            "modifications": result["corrected"].get("modifications", []),
        }
    except Exception as e:
        logger.error(f"Correct failed: {e}")
        raise HTTPException(status_code=500, detail=f"纠错失败: {e}")


# ============================================================
# 外部数据导入
# ============================================================

@app.post("/api/import/url")
async def import_from_url(url: str = Form(...)):
    """
    从 URL 下载图片并导入到上传目录。
    """
    import urllib.request
    from urllib.parse import urlparse

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FaceAnnotator/3.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        if len(data) < 1024:
            raise HTTPException(status_code=400, detail="文件过小或无效")

        parsed = urlparse(url)
        filename = Path(parsed.path).name or f"download_{uuid.uuid4().hex[:8]}.jpg"
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            filename += ".jpg"

        file_id = uuid.uuid4().hex[:12]
        ext = Path(filename).suffix.lower()
        saved_name = file_id + ext
        save_path = UPLOAD_DIR / saved_name
        save_path.write_bytes(data)

        task_store[file_id] = {
            "file_id": file_id,
            "file_path": str(save_path),
            "original_name": filename,
            "status": "uploaded",
            "person_name": None,
            "is_new_face": False,
            "error": None,
        }

        logger.info(f"URL import: {url[:80]} -> {saved_name}")
        return {
            "file_id": file_id,
            "filename": filename,
            "status": "uploaded",
            "source_url": url,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL import failed: {e}")
        raise HTTPException(status_code=500, detail=f"URL导入失败: {e}")


# ============================================================
# 文件管理
# ============================================================

@app.delete("/api/file/{file_id}")
async def delete_file(file_id: str):
    if file_id not in task_store:
        raise HTTPException(status_code=404, detail="文件不存在")
    task = task_store[file_id]
    for key in ("file_path", "annotation_path", "viz_path"):
        p = task.get(key)
        if p and Path(p).exists():
            Path(p).unlink()
    del task_store[file_id]
    return {"file_id": file_id, "status": "deleted"}


@app.post("/api/clear")
async def clear_all():
    count = len(task_store)
    for task in list(task_store.values()):
        for key in ("file_path", "annotation_path", "viz_path"):
            p = task.get(key)
            if p and Path(p).exists():
                Path(p).unlink()
    task_store.clear()
    logger.info(f"Cleared {count} tasks")
    return {"status": "cleared", "deleted_count": count}


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "detector_loaded": detector is not None,
        "extractor_loaded": extractor is not None,
        "face_app_loaded": face_app is not None,
        "tasks_count": len(task_store),
        "persons_count": len(load_person_index()),
    }


@app.post("/api/shutdown")
async def shutdown():
    """关闭服务器。"""
    logger.info("Shutdown requested via API")
    import threading
    # 延迟 0.5s 关闭，确保 HTTP 响应已发送给前端
    threading.Thread(target=lambda: (__import__('time').sleep(0.5), __import__('os')._exit(0))).start()
    return {"status": "shutting_down"}


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 54, flush=True)
    print("  FaceAnnotator v2.0 — Precision Lab", flush=True)
    print("  人脸标注 + 识别 + 自动命名归类", flush=True)
    print("=" * 54, flush=True)
    print("  加载模型（首次约 15 秒），请稍候...", flush=True)
    print("  启动后访问: http://localhost:8000", flush=True)
    print("=" * 54 + "\n", flush=True)

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")
    except (OSError, SystemExit) as e:
        if isinstance(e, SystemExit) or any(kw in str(e).lower() for kw in ("10048", "bind", "address")):
            print("\n  [错误] 端口 8000 已被占用", flush=True)
            print("    查找: netstat -ano | findstr :8000", flush=True)
        else:
            print(f"\n  [错误] 启动失败: {e}", flush=True)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  服务已停止", flush=True)
    except Exception as e:
        print(f"\n  [错误] 启动异常: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
