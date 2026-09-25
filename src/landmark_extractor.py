"""
文件名: landmark_extractor.py
功能: 基于 MediaPipe FaceLandmarker (Tasks API) 的人脸关键点提取器，
      从 478 个稠密网格点中提取 5 个简化关键点
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from mediapipe import Image as MpImage
from mediapipe import ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker as MpFaceLandmarker
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode

from .utils import bgr_to_rgb, get_project_root, load_image

# ============================================================
# MediaPipe FaceLandmarker 478 点 → 5 个简化关键点的索引映射
# 注：索引 0-467 为原始 Face Mesh 468 点，468-477 为虹膜点
# ============================================================

# 左眼区域关键点索引（以观察者视角为准，即图片右侧的眼睛）
LEFT_EYE_INDICES = (33, 133, 157, 158, 159, 160, 161, 246)

# 右眼区域关键点索引（以观察者视角为准，即图片左侧的眼睛）
RIGHT_EYE_INDICES = (362, 263, 384, 385, 386, 387, 388, 466)

# 鼻尖关键点索引
NOSE_TIP_INDEX = 1

# 左嘴角关键点索引（观察者视角）
LEFT_MOUTH_CORNER_INDEX = 61

# 右嘴角关键点索引（观察者视角）
RIGHT_MOUTH_CORNER_INDEX = 291


class LandmarkExtractor:
    """
    基于 MediaPipe FaceLandmarker (Tasks API) 的人脸关键点提取器。

    从 478 个稠密面部网格点中，通过索引映射提取 5 个简化关键点：
    - left_eye（左眼中心）
    - right_eye（右眼中心）
    - nose_tip（鼻尖）
    - left_mouth_corner（左嘴角）
    - right_mouth_corner（右嘴角）

    支持通过传入 bbox 进行 ROI 裁剪，提高小脸或远距离人脸的精度。

    Attributes:
        num_faces: 最多检测的人脸数
        min_face_detection_confidence: 最小人脸检测置信度
        min_face_presence_confidence: 最小人脸存在置信度
        min_tracking_confidence: 最小跟踪置信度
    """

    # 默认模型文件路径（相对于项目根目录）
    DEFAULT_MODEL_PATH = "models/face_landmarker.task"

    def __init__(
        self,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ):
        """
        初始化 MediaPipe FaceLandmarker (Tasks API) 模型。

        Args:
            num_faces: 最多检测的人脸数量
            min_face_detection_confidence: 人脸检测置信度阈值
            min_face_presence_confidence: 人脸存在置信度阈值
            min_tracking_confidence: 跟踪置信度阈值（视频模式使用）
            model_path: 模型文件路径，默认使用 models/face_landmarker.task
        """
        self.num_faces = num_faces
        self.min_face_detection_confidence = min_face_detection_confidence
        self.min_face_presence_confidence = min_face_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # 解析模型路径
        if model_path is None:
            model_path = str(get_project_root() / self.DEFAULT_MODEL_PATH)
        self.model_path = model_path

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"人脸关键点模型文件不存在: {model_path}\n"
                f"请先下载模型到 models/ 目录"
            )

        # 构建选项并创建 landmarker
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = MpFaceLandmarker.create_from_options(options)

    def extract(
        self,
        image: Union[str, Path, np.ndarray],
        bbox: Optional[Dict[str, int]] = None,
    ) -> Optional[Dict[str, List[int]]]:
        """
        提取面部简化关键点。

        Args:
            image: 图片路径或 BGR 格式的 numpy 数组
            bbox: 可选的人脸边界框，传入后将先裁剪再提取以提高精度

        Returns:
            Optional[Dict[str, List[int]]]:
            {
                "left_eye": [x, y],              # 左眼中心坐标
                "right_eye": [x, y],             # 右眼中心坐标
                "nose_tip": [x, y],              # 鼻尖坐标
                "left_mouth_corner": [x, y],     # 左嘴角坐标
                "right_mouth_corner": [x, y],    # 右嘴角坐标
            }
            若未检测到人脸网格，返回 None。
        """
        # 加载图片（若传入路径）
        if isinstance(image, (str, Path)):
            image = load_image(image)

        h, w = image.shape[:2]

        # ROI 偏移量（若使用裁剪则需要映射回原图坐标）
        offset_x, offset_y = 0, 0
        roi_image = image

        if bbox is not None:
            # 带边距的 ROI 裁剪
            pad_x = int(bbox["width"] * 0.2)
            pad_y = int(bbox["height"] * 0.2)
            x1 = max(0, bbox["x"] - pad_x)
            y1 = max(0, bbox["y"] - pad_y)
            x2 = min(w, bbox["x"] + bbox["width"] + pad_x)
            y2 = min(h, bbox["y"] + bbox["height"] + pad_y)

            roi_image = image[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        # 转换为 RGB，创建 MediaPipe Image
        rgb_roi = bgr_to_rgb(roi_image)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb_roi)

        # 执行 FaceLandmarker 检测
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        # 取第一个人脸的 landmarks
        raw_landmarks = result.face_landmarks[0]
        roi_h, roi_w = roi_image.shape[:2]

        # 映射为简化关键点
        simplified = self._map_to_simplified(
            raw_landmarks, roi_w, roi_h, offset_x, offset_y
        )

        # 边界裁剪
        simplified = self._clamp_landmarks(simplified, w, h)

        return simplified

    def _map_to_simplified(
        self,
        raw_landmarks: List,
        img_w: int,
        img_h: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> Dict[str, List[int]]:
        """
        将 478 个稠密 landmarks 映射为 5 个简化关键点。

        Args:
            raw_landmarks: FaceLandmarker 返回的 landmarks 列表 (478个)
            img_w: ROI 图片宽度
            img_h: ROI 图片高度
            offset_x: ROI 在原图中的 x 偏移
            offset_y: ROI 在原图中的 y 偏移

        Returns:
            Dict[str, List[int]]: 简化关键点（绝对坐标）
        """
        # 辅助函数：将归一化坐标转为绝对像素坐标
        def to_pixel(idx: int) -> Tuple[int, int]:
            lm = raw_landmarks[idx]
            px = int(lm.x * img_w) + offset_x
            py = int(lm.y * img_h) + offset_y
            return px, py

        # 辅助函数：计算一组索引的平均坐标
        def average_of(indices: Tuple[int, ...]) -> List[int]:
            xs, ys = 0, 0
            for idx in indices:
                px, py = to_pixel(idx)
                xs += px
                ys += py
            n = len(indices)
            return [xs // n, ys // n]

        return {
            "left_eye": average_of(LEFT_EYE_INDICES),
            "right_eye": average_of(RIGHT_EYE_INDICES),
            "nose_tip": list(to_pixel(NOSE_TIP_INDEX)),
            "left_mouth_corner": list(to_pixel(LEFT_MOUTH_CORNER_INDEX)),
            "right_mouth_corner": list(to_pixel(RIGHT_MOUTH_CORNER_INDEX)),
        }

    @staticmethod
    def _clamp_landmarks(
        landmarks: Dict[str, List[int]],
        img_w: int,
        img_h: int,
    ) -> Dict[str, List[int]]:
        """
        将关键点坐标裁剪到图片范围内。

        Args:
            landmarks: 关键点字典
            img_w: 图片宽度
            img_h: 图片高度

        Returns:
            Dict[str, List[int]]: 裁剪后的关键点
        """
        clamped = {}
        for name, (lx, ly) in landmarks.items():
            clamped[name] = [
                max(0, min(lx, img_w - 1)),
                max(0, min(ly, img_h - 1)),
            ]
        return clamped

    def extract_raw_landmarks(
        self, image: Union[str, Path, np.ndarray]
    ) -> Optional[List]:
        """
        获取完整的 478 个稠密 landmarks（供高级用途使用）。

        Args:
            image: 图片路径或 BGR 格式的 numpy 数组

        Returns:
            Optional[List]: 478 个 NormalizedLandmark 对象的列表，或 None
        """
        if isinstance(image, (str, Path)):
            image = load_image(image)

        rgb_image = bgr_to_rgb(image)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb_image)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        return list(result.face_landmarks[0])

    def close(self):
        """释放 MediaPipe FaceLandmarker 资源。"""
        if self._landmarker:
            self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
