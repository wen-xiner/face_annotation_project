"""
文件名: utils.py
功能: 共享工具函数 —— 路径处理、图片加载、标注数据格式定义
"""

import json
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

# ============================================================
# 常量定义
# ============================================================

# 标注 JSON 必需顶层字段
ANNOTATION_REQUIRED_KEYS = {"image_path", "bbox", "landmarks", "annotator", "date"}

# bbox 字典必需子字段
BBOX_REQUIRED_KEYS = {"x", "y", "width", "height"}

# landmarks 字典必需的 5 个简化关键点
LANDMARK_REQUIRED_KEYS = {
    "left_eye",
    "right_eye",
    "nose_tip",
    "left_mouth_corner",
    "right_mouth_corner",
}

# 支持的图片格式
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


# ============================================================
# 路径处理
# ============================================================


def normalize_path(filepath: Union[str, Path]) -> str:
    """
    将任意路径转换为项目相对路径，统一使用正斜杠。

    Args:
        filepath: 输入路径（绝对路径或相对路径）

    Returns:
        str: 规范化后的相对路径字符串（正斜杠分隔）
    """
    path = Path(filepath)
    # 尝试转为相对于项目根目录的路径
    try:
        # 获取当前文件所在的项目根目录
        project_root = Path(__file__).resolve().parent.parent
        rel_path = path.resolve().relative_to(project_root)
    except ValueError:
        # 如果不在项目根目录下，使用原始路径
        rel_path = path

    # 转换为 POSIX 风格路径（正斜杠）
    return PurePosixPath(rel_path).as_posix()


def get_project_root() -> Path:
    """
    获取项目根目录的绝对路径。

    Returns:
        Path: 项目根目录路径
    """
    return Path(__file__).resolve().parent.parent


# ============================================================
# 图片处理
# ============================================================


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    使用 OpenCV 加载图片，失败时抛出异常。

    Args:
        image_path: 图片文件路径

    Returns:
        np.ndarray: BGR 格式的图片数组 (H, W, 3)

    Raises:
        FileNotFoundError: 图片文件不存在
        ValueError: 无法读取图片（文件损坏或格式不支持）
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法读取图片文件（可能损坏或格式不支持）: {path}")

    return image


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    使用 OpenCV 保存图片，自动创建父目录。

    Args:
        image: BGR 格式的图片数组
        output_path: 输出文件路径
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"无法保存图片到: {path}")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    将 BGR 图片转换为 RGB 格式。

    Args:
        image: BGR 格式图片数组

    Returns:
        np.ndarray: RGB 格式图片数组
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_image_size(image: np.ndarray) -> Dict[str, int]:
    """
    获取图片尺寸。

    Args:
        image: 图片数组 (H, W, C)

    Returns:
        Dict[str, int]: {"height": int, "width": int}
    """
    h, w = image.shape[:2]
    return {"height": h, "width": w}


# ============================================================
# 标注数据处理
# ============================================================


def build_annotation_dict(
    image_path: Union[str, Path],
    bbox: Dict[str, int],
    landmarks: Dict[str, List[int]],
    annotator: str = "auto",
    annotate_date: Optional[str] = None,
    **extra_fields,
) -> Dict[str, Any]:
    """
    构建标准标注字典，确保包含所有必需字段。

    Args:
        image_path: 原始图片的相对路径
        bbox: 边界框字典 {"x": int, "y": int, "width": int, "height": int}
        landmarks: 关键点字典 {"left_eye": [x,y], "right_eye": [x,y], ...}
        annotator: 标注者标识（"auto", "auto_partial", "manual"）
        annotate_date: 标注日期（YYYY-MM-DD），默认使用当天日期
        **extra_fields: 额外字段（如 confidence, mediapipe_version 等）

    Returns:
        Dict: 符合规范的标注字典
    """
    if annotate_date is None:
        annotate_date = date.today().isoformat()

    annotation = {
        "image_path": normalize_path(image_path),
        "bbox": {
            "x": int(bbox["x"]),
            "y": int(bbox["y"]),
            "width": int(bbox["width"]),
            "height": int(bbox["height"]),
        },
        "landmarks": {
            "left_eye": [int(landmarks["left_eye"][0]), int(landmarks["left_eye"][1])],
            "right_eye": [int(landmarks["right_eye"][0]), int(landmarks["right_eye"][1])],
            "nose_tip": [int(landmarks["nose_tip"][0]), int(landmarks["nose_tip"][1])],
            "left_mouth_corner": [
                int(landmarks["left_mouth_corner"][0]),
                int(landmarks["left_mouth_corner"][1]),
            ],
            "right_mouth_corner": [
                int(landmarks["right_mouth_corner"][0]),
                int(landmarks["right_mouth_corner"][1]),
            ],
        },
        "annotator": annotator,
        "date": annotate_date,
    }

    # 合并额外字段
    annotation.update(extra_fields)

    return annotation


def save_annotation_json(
    annotation: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    将标注字典保存为 JSON 文件。

    Args:
        annotation: 标注字典
        output_path: 输出 JSON 文件路径
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)


def load_annotation_json(annotation_path: Union[str, Path]) -> Dict[str, Any]:
    """
    从 JSON 文件加载标注数据。

    Args:
        annotation_path: 标注 JSON 文件路径

    Returns:
        Dict: 标注字典

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 格式无效
    """
    path = Path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(f"标注文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
