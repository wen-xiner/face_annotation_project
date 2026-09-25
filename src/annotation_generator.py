"""
文件名: annotation_generator.py
功能: 标注生成器，整合人脸检测与关键点提取，输出标准 JSON 标注文件
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .face_detector import FaceDetector
from .landmark_extractor import LandmarkExtractor
from .quality_filter import QualityFilter
from .utils import (
    build_annotation_dict,
    load_annotation_json,
    load_image,
    normalize_path,
    save_annotation_json,
)

# 配置模块级日志
logger = logging.getLogger(__name__)


class AnnotationGenerator:
    """
    标注生成器，整合人脸检测与关键点提取，输出标准 JSON 标注文件。

    对每张图片执行：人脸检测 → 关键点提取 → 质量评估(可选) → 构建标注字典 → 保存 JSON。

    Attributes:
        detector: FaceDetector 实例
        extractor: LandmarkExtractor 实例
        output_dir: 标注 JSON 输出根目录
        raw_dir: 原始图片根目录（用于计算相对路径）
        quality_filter: QualityFilter 实例（可选），用于图片质量评估
        quality_threshold: 质量分数阈值，低于此值的标注将被拒绝
        auto_reject: 是否自动拒绝低质量标注
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "output/annotations/auto_labeled",
        raw_dir: Union[str, Path] = "dataset/raw_images",
        quality_filter: Optional[QualityFilter] = None,
        quality_threshold: float = 0.4,
        auto_reject: bool = False,
    ):
        """
        初始化检测器、提取器并设置输出目录。

        Args:
            output_dir: 标注 JSON 文件的输出根目录
            raw_dir: 原始图片根目录
            quality_filter: QualityFilter 实例，为 None 时默认创建（用于质量评估）
            quality_threshold: 质量分数阈值（0-1），低于此值且 auto_reject=True 时将拒绝保存
            auto_reject: 是否自动拒绝低质量标注。启用后，质量分 < quality_threshold 的标注不会保存
        """
        self.output_dir = Path(output_dir)
        self.raw_dir = Path(raw_dir)
        self.quality_threshold = quality_threshold
        self.auto_reject = auto_reject

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化子模块（使用 with 管理生命周期时由外部控制更佳，
        # 但这里选择在 __init__ 中创建以简化调用）
        self.detector = FaceDetector()
        self.extractor = LandmarkExtractor()

        # 质量过滤器
        if quality_filter is not None:
            self.quality_filter = quality_filter
        else:
            self.quality_filter = QualityFilter()

    def annotate(
        self,
        image_path: Union[str, Path],
        save: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        对单张图片执行完整标注流程：检测 → 关键点提取 → 生成 JSON。

        Args:
            image_path: 图片文件路径（绝对或相对路径均可）
            save: 是否自动保存为 JSON 文件

        Returns:
            Optional[Dict]: 标注字典，若人脸检测或关键点提取失败则返回 None
        """
        image_path = Path(image_path).resolve()

        if not image_path.exists():
            logger.error(f"图片文件不存在: {image_path}")
            return None

        logger.info(f"正在标注: {image_path.name}")

        # 步骤 1：人脸检测
        bbox = self.detector.detect(image_path)
        if bbox is None:
            logger.warning(f"未检测到人脸: {image_path.name}")
            return None

        # 步骤 2：关键点提取（传入 bbox 进行 ROI 裁剪）
        landmarks = self.extractor.extract(image_path, bbox=bbox)

        # 步骤 3：构建标注字典
        if landmarks is not None:
            annotator = "auto"
        else:
            # 检测到人脸但关键点提取失败，保存部分标注
            annotator = "auto_partial"
            logger.warning(f"人脸关键点提取失败，保存部分标注: {image_path.name}")
            landmarks = {
                "left_eye": [-1, -1],
                "right_eye": [-1, -1],
                "nose_tip": [-1, -1],
                "left_mouth_corner": [-1, -1],
                "right_mouth_corner": [-1, -1],
            }

        annotation = build_annotation_dict(
            image_path=image_path,
            bbox=bbox,
            landmarks=landmarks,
            annotator=annotator,
            confidence=bbox.get("confidence", None),
        )

        # 步骤 3.5：质量评估（可选）
        quality_result = None
        try:
            image_array = load_image(image_path)
            quality_result = self.quality_filter.assess(image_array, landmarks)
            annotation["quality"] = quality_result

            # 自动拒绝低质量标注
            if self.auto_reject and quality_result["quality_score"] < self.quality_threshold:
                logger.warning(
                    f"图片质量过低 (score={quality_result['quality_score']:.2f}, "
                    f"label={quality_result['quality_label']})，已自动拒绝: {image_path.name}"
                )
                return None
        except Exception as e:
            logger.warning(f"质量评估失败，跳过: {image_path.name} - {e}")

        # 步骤 4：保存 JSON
        if save:
            output_path = self._get_output_path(image_path)
            save_annotation_json(annotation, output_path)
            logger.info(f"标注已保存: {output_path}")

        return annotation

    def annotate_batch(
        self,
        image_paths: List[Path],
        save: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量标注多张图片。

        Args:
            image_paths: 图片路径列表
            save: 是否保存 JSON 文件

        Returns:
            List[Dict]: 每张图片的处理结果摘要
            [{"image": str, "status": "success"/"failed"/"rejected",
              "annotation_path": str/None, "error": str/None, "quality": dict/None}, ...]
        """
        results = []
        for img_path in image_paths:
            result = {
                "image": str(img_path),
                "status": "failed",
                "annotation_path": None,
                "error": None,
                "quality": None,
            }
            try:
                annotation = self.annotate(img_path, save=save)
                if annotation is not None:
                    result["status"] = "success"
                    result["quality"] = annotation.get("quality")
                    if save:
                        result["annotation_path"] = str(
                            self._get_output_path(img_path)
                        )
                else:
                    # 判断失败原因：如果 auto_reject 开启，可能是质量拒绝
                    if self.auto_reject:
                        result["status"] = "rejected"
                        result["error"] = (
                            "图片质量低于阈值 "
                            f"(threshold={self.quality_threshold})，自动拒绝"
                        )
                    else:
                        result["error"] = "人脸检测或关键点提取失败"
            except Exception as e:
                result["error"] = str(e)
                logger.error(f"标注失败 {img_path.name}: {e}")

            results.append(result)

        return results

    def _get_output_path(self, image_path: Path) -> Path:
        """
        根据输入图片路径计算对应的标注 JSON 输出路径。

        保持与 dataset/raw_images/ 相同的子目录结构：
          输入: dataset/raw_images/person_001/image_001.jpg
          输出: output/annotations/auto_labeled/person_001/image_001_annotation.json

        Args:
            image_path: 图片文件的绝对路径

        Returns:
            Path: 标注 JSON 输出路径
        """
        # 尝试计算相对于 raw_dir 的路径
        try:
            rel_path = image_path.relative_to(self.raw_dir.resolve())
        except ValueError:
            # 如果图片不在 raw_dir 下，使用图片所在文件夹名作为分组
            rel_path = Path(image_path.parent.name) / image_path.name

        # 构建输出路径：去掉原扩展名，添加 _annotation.json
        output_name = f"{image_path.stem}_annotation.json"
        output_rel = rel_path.parent / output_name

        return self.output_dir / output_rel

    def annotation_exists(self, image_path: Path) -> bool:
        """
        检查某张图片的标注文件是否已存在。

        Args:
            image_path: 图片文件路径

        Returns:
            bool: 标注文件已存在返回 True
        """
        return self._get_output_path(image_path).exists()

    def close(self):
        """释放内部检测器和提取器的资源。"""
        self.detector.close()
        self.extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
