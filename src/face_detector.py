"""
文件名: face_detector.py
功能: 基于 MediaPipe Face Detection (Tasks API) 的人脸检测器，
      检测图片中的人脸并返回边界框（bbox）坐标
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import mediapipe as mp
import numpy as np
from mediapipe import Image as MpImage
from mediapipe import ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector as MpFaceDetector
from mediapipe.tasks.python.vision import FaceDetectorOptions
from mediapipe.tasks.python.vision import RunningMode

from .utils import bgr_to_rgb, get_project_root, load_image


class FaceDetector:
    """
    基于 MediaPipe Face Detection (Tasks API) 的人脸检测器。

    使用 BlazeFace 短距离模型，检测图片中人脸并返回像素边界框。

    Attributes:
        min_detection_confidence: 最小检测置信度阈值（0.0-1.0）
        model_path: 模型文件路径
    """

    # 默认模型文件路径（相对于项目根目录）
    DEFAULT_MODEL_PATH = "models/blaze_face_short_range.tflite"

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ):
        """
        初始化 MediaPipe Face Detection (Tasks API) 模型。

        Args:
            min_detection_confidence: 检测置信度低于此值的结果将被过滤
            model_path: 模型文件路径，默认使用 models/blaze_face_short_range.tflite
        """
        self.min_detection_confidence = min_detection_confidence

        # 解析模型路径
        if model_path is None:
            model_path = str(get_project_root() / self.DEFAULT_MODEL_PATH)
        self.model_path = model_path

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"人脸检测模型文件不存在: {model_path}\n"
                f"请先下载模型到 models/ 目录"
            )

        # 构建选项并创建检测器
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = MpFaceDetector.create_from_options(options)

    def detect(
        self, image: Union[str, Path, np.ndarray]
    ) -> Optional[Dict[str, int]]:
        """
        检测输入图片中的主要人脸（置信度最高的那张），返回边界框。

        Args:
            image: 图片路径（str/Path）或 BGR 格式的 numpy 数组 (H, W, 3)

        Returns:
            Optional[Dict[str, int]]: {"x": int, "y": int, "width": int, "height": int}
            若未检测到人脸，返回 None。
        """
        results = self.detect_all(image)
        if not results:
            return None

        # 返回置信度最高的人脸（已按置信度降序排列）
        best = results[0]
        return {
            "x": best["x"],
            "y": best["y"],
            "width": best["width"],
            "height": best["height"],
        }

    def detect_all(
        self, image: Union[str, Path, np.ndarray]
    ) -> List[Dict[str, int]]:
        """
        检测输入图片中的所有人脸，按置信度降序排列。

        Args:
            image: 图片路径或 BGR 格式的 numpy 数组

        Returns:
            List[Dict[str, int]]: 每张人脸的边界框及置信度，
            [{"x": int, "y": int, "width": int, "height": int, "confidence": float}, ...]
        """
        # 加载图片（若传入路径）
        if isinstance(image, (str, Path)):
            image = load_image(image)

        h, w = image.shape[:2]

        # 转换为 RGB，创建 MediaPipe Image
        rgb_image = bgr_to_rgb(image)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb_image)

        # 执行检测
        result = self._detector.detect(mp_image)

        if not result.detections:
            return []

        detections = []
        for detection in result.detections:
            bbox = detection.bounding_box

            # Tasks API 返回的坐标是像素坐标
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            width = int(bbox.width)
            height = int(bbox.height)

            # 边界裁剪，确保不超出图片范围
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            # 获取置信度
            confidence = 0.0
            if detection.categories:
                confidence = detection.categories[0].score

            detections.append(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "confidence": round(float(confidence), 4),
                }
            )

        # 按置信度降序排列
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        return detections

    def close(self):
        """释放 MediaPipe 检测器资源。"""
        if self._detector:
            self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
