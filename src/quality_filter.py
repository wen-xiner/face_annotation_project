"""
文件名: quality_filter.py
功能: 图片质量过滤器，评估人脸标注图片的适用性

提供三个维度的质量检测：
  1. 模糊检测 — 基于 Laplacian 方差，评估图片清晰度
  2. 亮度检测 — 基于灰度均值，检测过暗/过曝
  3. 姿态检测 — 基于双眼连线与水平线的夹角，评估人脸倾斜程度

综合评分将三个维度加权组合，输出 0-1 的质量分数和质量等级。
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class QualityFilter:
    """
    图片质量过滤器，用于评估人脸标注图片的适用性。

    默认阈值基于经验值设定，所有参数均可通过构造函数自定义。

    Attributes:
        blur_threshold: Laplacian 方差阈值，低于此值视为模糊
        min_brightness: 最小亮度（灰度均值），低于此值视为过暗
        max_brightness: 最大亮度（灰度均值），高于此值视为过曝
        max_tilt_angle: 最大允许的人脸倾斜角度（度）
        weights: 三项指标的加权权重字典
    """

    # 质量等级阈值
    QUALITY_GOOD = 0.7       # 优秀
    QUALITY_ACCEPTABLE = 0.4  # 可接受

    def __init__(
        self,
        blur_threshold: float = 100.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        max_tilt_angle: float = 30.0,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        初始化质量过滤器。

        Args:
            blur_threshold: Laplacian 方差阈值，默认 100.0。
                            经验值：清晰图片通常 > 150，轻度模糊 50-100，严重模糊 < 50
            min_brightness: 最小亮度（0-255），默认 40.0。低于此值视为过暗
            max_brightness: 最大亮度（0-255），默认 220.0。高于此值视为过曝
            max_tilt_angle: 最大允许的人脸倾斜角（度），默认 30.0
            weights: 三项指标的权重，默认 {"blur": 0.4, "brightness": 0.3, "pose": 0.3}
        """
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_tilt_angle = max_tilt_angle
        self.weights = weights or {"blur": 0.4, "brightness": 0.3, "pose": 0.3}

        # 验证权重总和
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            # 自动归一化
            self.weights = {k: v / total for k, v in self.weights.items()}

    def assess(
        self,
        image: np.ndarray,
        landmarks: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        综合评估图片质量，返回各项分数和综合评分。

        Args:
            image: BGR 格式的 numpy 图片数组（OpenCV 读取格式）
            landmarks: 可选的人脸关键点字典，用于姿态检测。
                       若为 None 则跳过姿态检测，仅评估模糊和亮度。

        Returns:
            Dict[str, Any]:
            {
                "quality_score": float,       # 0-1 综合评分，越高越好
                "quality_label": str,          # "good" / "acceptable" / "poor"
                "blur_score": float,           # 模糊评分 (0-1)
                "brightness_score": float,     # 亮度评分 (0-1)
                "pose_score": float,           # 姿态评分 (0-1)，无关键点时为 1.0
                "details": {
                    "blur": {
                        "laplacian_variance": float,  # Laplacian 方差原始值
                        "is_blurry": bool,
                    },
                    "brightness": {
                        "mean_value": float,           # 灰度均值
                        "is_too_dark": bool,
                        "is_too_bright": bool,
                    },
                    "pose": {
                        "tilt_angle": float,           # 倾斜角度（度）
                        "is_tilted": bool,
                    } | None,
                },
            }
        """
        # 单项检测
        blur_result = self.check_blur(image)
        brightness_result = self.check_brightness(image)
        pose_result = self.check_pose(landmarks) if landmarks else None

        # 单项评分
        blur_score = self._score_blur(blur_result["laplacian_variance"])
        brightness_score = self._score_brightness(brightness_result["mean_value"])
        pose_score = self._score_pose(
            pose_result["tilt_angle"] if pose_result else 0.0
        )

        # 加权综合评分
        quality_score = (
            self.weights["blur"] * blur_score
            + self.weights["brightness"] * brightness_score
            + self.weights["pose"] * pose_score
        )

        # 质量等级
        quality_score = round(min(1.0, max(0.0, quality_score)), 4)
        if quality_score >= self.QUALITY_GOOD:
            quality_label = "good"
        elif quality_score >= self.QUALITY_ACCEPTABLE:
            quality_label = "acceptable"
        else:
            quality_label = "poor"

        # 构建详情
        details = {
            "blur": {
                "laplacian_variance": round(blur_result["laplacian_variance"], 2),
                "is_blurry": blur_result["is_blurry"],
            },
            "brightness": {
                "mean_value": round(brightness_result["mean_value"], 2),
                "is_too_dark": brightness_result["is_too_dark"],
                "is_too_bright": brightness_result["is_too_bright"],
            },
            "pose": {
                "tilt_angle": round(pose_result["tilt_angle"], 2),
                "is_tilted": pose_result["is_tilted"],
            } if pose_result else None,
        }

        return {
            "quality_score": quality_score,
            "quality_label": quality_label,
            "blur_score": round(blur_score, 4),
            "brightness_score": round(brightness_score, 4),
            "pose_score": round(pose_score, 4),
            "details": details,
        }

    # ================================================================
    # 单项检测方法
    # ================================================================

    def check_blur(self, image: np.ndarray) -> Dict[str, Any]:
        """
        检测图片模糊程度（基于 Laplacian 方差）。

        原理：对图像应用 Laplacian 算子（二阶导数），计算像素值的方差。
        清晰图片边缘明显，方差高；模糊图片边缘模糊，方差低。

        Args:
            image: BGR 格式的 numpy 图片数组

        Returns:
            Dict: {"laplacian_variance": float, "is_blurry": bool}
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return {
            "laplacian_variance": float(variance),
            "is_blurry": bool(variance < self.blur_threshold),
        }

    def check_brightness(self, image: np.ndarray) -> Dict[str, Any]:
        """
        检测图片亮度（基于灰度均值）。

        Args:
            image: BGR 格式的 numpy 图片数组

        Returns:
            Dict: {
                "mean_value": float,      # 灰度均值 (0-255)
                "is_too_dark": bool,      # 是否过暗
                "is_too_bright": bool,    # 是否过曝
            }
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean()

        return {
            "mean_value": float(mean_val),
            "is_too_dark": bool(mean_val < self.min_brightness),
            "is_too_bright": bool(mean_val > self.max_brightness),
        }

    def check_pose(
        self, landmarks: Dict[str, List[int]]
    ) -> Dict[str, Any]:
        """
        检测人脸倾斜程度（基于双眼连线与水平线的夹角）。

        使用 left_eye 和 right_eye 关键点计算连线角度。
        角度为 0 表示双眼完全水平，正角度表示逆时针倾斜。

        Args:
            landmarks: 人脸关键点字典，至少包含 left_eye 和 right_eye

        Returns:
            Dict: {
                "tilt_angle": float,       # 倾斜角度（0-90 度）
                "is_tilted": bool,         # 是否超过阈值
            }
        """
        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")

        if (
            left_eye is None
            or right_eye is None
            or len(left_eye) != 2
            or len(right_eye) != 2
        ):
            # 无法计算姿态，返回默认值
            return {"tilt_angle": 0.0, "is_tilted": False}

        # 检查是否为无效关键点（partial 标注）
        if (
            (left_eye[0] < 0 and left_eye[1] < 0)
            or (right_eye[0] < 0 and right_eye[1] < 0)
        ):
            return {"tilt_angle": 0.0, "is_tilted": False}

        # 计算双眼连线与水平线的夹角
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = abs(math.degrees(angle_rad))

        # 规范化到 0-90 度
        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        return {
            "tilt_angle": angle_deg,
            "is_tilted": angle_deg > self.max_tilt_angle,
        }

    # ================================================================
    # 单项评分方法（0-1 归一化）
    # ================================================================

    def _score_blur(self, variance: float) -> float:
        """
        将 Laplacian 方差映射为 0-1 的模糊评分。

        使用分段线性映射：
          - variance >= blur_threshold * 1.5 → 1.0（非常清晰）
          - variance 在 [0, threshold*1.5] 之间 → 线性增长
        """
        ref = self.blur_threshold * 1.5
        if ref <= 0:
            return 1.0
        return min(1.0, max(0.0, variance / ref))

    def _score_brightness(self, mean_val: float) -> float:
        """
        将灰度均值映射为 0-1 的亮度评分。

        采用梯形评分模型：中间亮度区间（ideal_low ~ ideal_high）满分，
        两边线性衰减到 min_brightness / max_brightness 时为 0。
        """
        ideal_low = max(self.min_brightness, 80.0)
        ideal_high = min(self.max_brightness, 180.0)

        if ideal_low <= mean_val <= ideal_high:
            return 1.0
        elif mean_val < ideal_low:
            # 暗端：从 min_brightness(0分) 到 ideal_low(1分) 线性过渡
            if mean_val <= self.min_brightness:
                return 0.0
            return (mean_val - self.min_brightness) / (ideal_low - self.min_brightness)
        else:
            # 亮端：从 ideal_high(1分) 到 max_brightness(0分) 线性过渡
            if mean_val >= self.max_brightness:
                return 0.0
            return (self.max_brightness - mean_val) / (
                self.max_brightness - ideal_high
            )

    def _score_pose(self, tilt_angle: float) -> float:
        """
        将倾斜角度映射为 0-1 的姿态评分。

        角度为 0 时满分，线性衰减到 max_tilt_angle 时为 0。
        """
        if self.max_tilt_angle <= 0:
            return 1.0
        return max(0.0, 1.0 - tilt_angle / self.max_tilt_angle)
