"""
文件名: validator.py
功能: 标注文件校验器，检查 JSON 标注文件的完整性、格式正确性和数据合理性
      同时提供标注质量评估功能（图片质量 + 标注合理性）
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .utils import (
    ANNOTATION_REQUIRED_KEYS,
    BBOX_REQUIRED_KEYS,
    LANDMARK_REQUIRED_KEYS,
    get_project_root,
    load_image,
)

logger = logging.getLogger(__name__)


class AnnotationValidator:
    """
    标注文件校验器，检查 JSON 标注文件的完整性和格式正确性。

    校验分为两个级别：
    - error（错误）：标注文件存在严重问题，不可使用
    - warning（警告）：标注文件存在轻微问题，建议人工复核

    Attributes:
        strict: True 时将警告视为错误
        errors: 当前校验累计的错误列表
        warnings: 当前校验累计的警告列表
    """

    # 合法的 annotator 值
    VALID_ANNOTATORS = {"auto", "auto_partial", "manual"}

    # 日期格式：YYYY-MM-DD
    DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    # 关键点在 bbox 外的容差比例（20%）
    BBOX_TOLERANCE_RATIO = 0.2

    def __init__(self, strict: bool = False):
        """
        初始化校验器。

        Args:
            strict: True 时将警告也视为错误
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_file(
        self, annotation_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        校验单个标注文件。

        Args:
            annotation_path: 标注 JSON 文件路径

        Returns:
            Dict[str, Any]:
            {
                "file": str,
                "status": "valid" | "invalid",
                "errors": List[str],
                "warnings": List[str]
            }
        """
        self.errors = []
        self.warnings = []
        path = Path(annotation_path)

        # 1. 文件存在且为有效 JSON
        if not path.exists():
            return {
                "file": str(path),
                "status": "invalid",
                "errors": [f"文件不存在: {path}"],
                "warnings": [],
            }

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return {
                "file": str(path),
                "status": "invalid",
                "errors": [f"JSON 格式无效: {e}"],
                "warnings": [],
            }
        except Exception as e:
            return {
                "file": str(path),
                "status": "invalid",
                "errors": [f"无法读取文件: {e}"],
                "warnings": [],
            }

        # 2. 逐项校验
        self._validate_structure(data)
        self._validate_bbox(data.get("bbox", {}))
        self._validate_landmarks(data.get("landmarks", {}))
        self._validate_image_exists(data.get("image_path", ""))
        self._validate_annotator(data.get("annotator", ""))
        self._validate_date(data.get("date", ""))

        # 3. 交叉校验：关键点是否在 bbox 内
        bbox = data.get("bbox")
        landmarks = data.get("landmarks")
        if bbox and landmarks:
            self._validate_bbox_landmark_consistency(bbox, landmarks)

        # 4. 判断最终状态
        has_errors = len(self.errors) > 0
        if self.strict:
            has_errors = has_errors or len(self.warnings) > 0

        return {
            "file": str(path),
            "status": "invalid" if has_errors else "valid",
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def _validate_structure(self, data: Dict) -> None:
        """检查顶层必需字段是否存在且类型正确。"""
        # 检查必需字段
        missing = ANNOTATION_REQUIRED_KEYS - set(data.keys())
        if missing:
            self.errors.append(f"缺少必需字段: {', '.join(sorted(missing))}")

        # 检查字段类型
        if "image_path" in data and not isinstance(data["image_path"], str):
            self.errors.append("image_path 字段必须为字符串")

        if "bbox" in data and not isinstance(data["bbox"], dict):
            self.errors.append("bbox 字段必须为字典")

        if "landmarks" in data and not isinstance(data["landmarks"], dict):
            self.errors.append("landmarks 字段必须为字典")

        if "annotator" in data and not isinstance(data["annotator"], str):
            self.errors.append("annotator 字段必须为字符串")

        if "date" in data and not isinstance(data["date"], str):
            self.errors.append("date 字段必须为字符串")

    def _validate_bbox(self, bbox: Dict) -> None:
        """检查 bbox 字段的完整性和数值合法性。"""
        if not isinstance(bbox, dict):
            return

        # 检查必需子字段
        missing = BBOX_REQUIRED_KEYS - set(bbox.keys())
        if missing:
            self.errors.append(f"bbox 缺少子字段: {', '.join(sorted(missing))}")
            return

        # 检查每个字段的值
        for key in BBOX_REQUIRED_KEYS:
            val = bbox.get(key)
            if not isinstance(val, (int, float)):
                self.errors.append(f"bbox.{key} 必须为数值，当前值: {val}")
                continue
            if val < 0:
                self.errors.append(f"bbox.{key} 不能为负数，当前值: {val}")

        # 宽高必须大于 0
        if isinstance(bbox.get("width"), (int, float)) and bbox["width"] <= 0:
            self.errors.append(f"bbox.width 必须大于 0，当前值: {bbox['width']}")
        if isinstance(bbox.get("height"), (int, float)) and bbox["height"] <= 0:
            self.errors.append(f"bbox.height 必须大于 0，当前值: {bbox['height']}")

    def _validate_landmarks(self, landmarks: Dict) -> None:
        """检查 landmarks 字段的完整性和坐标格式。"""
        if not isinstance(landmarks, dict):
            return

        # 检查必需关键点
        missing = LANDMARK_REQUIRED_KEYS - set(landmarks.keys())
        if missing:
            self.errors.append(
                f"landmarks 缺少关键点: {', '.join(sorted(missing))}"
            )
            return

        for name, coords in landmarks.items():
            if not isinstance(coords, (list, tuple)):
                self.errors.append(
                    f"landmarks.{name} 必须为 [x, y] 列表"
                )
                continue

            if len(coords) != 2:
                self.errors.append(
                    f"landmarks.{name} 必须包含 2 个值 [x, y]，当前长度: {len(coords)}"
                )
                continue

            x, y = coords[0], coords[1]

            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                self.errors.append(
                    f"landmarks.{name} 坐标必须为数值，当前值: {coords}"
                )
                continue

            # 对于 partial 标注，允许 [-1, -1]
            if x == -1 and y == -1:
                self.warnings.append(f"landmarks.{name} 为无效坐标 [-1, -1]（partial 标注）")
            elif x < 0 or y < 0:
                self.errors.append(
                    f"landmarks.{name} 坐标不能为负数，当前值: {coords}"
                )

    def _validate_image_exists(self, image_path: str) -> None:
        """检查标注中引用的图片文件是否存在。"""
        if not image_path:
            return

        # 将相对路径转为绝对路径
        project_root = get_project_root()
        full_path = project_root / image_path

        if not full_path.exists():
            self.warnings.append(f"引用的图片文件不存在: {image_path}")

    def _validate_annotator(self, annotator: str) -> None:
        """检查 annotator 是否为已知值。"""
        if not annotator:
            return
        if annotator not in self.VALID_ANNOTATORS:
            self.warnings.append(
                f"annotator 值 '{annotator}' 不在已知值列表中: "
                f"{', '.join(sorted(self.VALID_ANNOTATORS))}"
            )

    def _validate_date(self, date_str: str) -> None:
        """检查日期格式是否为 YYYY-MM-DD。"""
        if not date_str:
            return

        if not self.DATE_PATTERN.match(date_str):
            self.warnings.append(
                f"date 格式不正确: '{date_str}'，应为 YYYY-MM-DD 格式"
            )
            return

        # 进一步检查是否为合法日期
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            self.warnings.append(f"date 值不是合法日期: '{date_str}'")

    def _validate_bbox_landmark_consistency(
        self, bbox: Dict, landmarks: Dict
    ) -> None:
        """
        检查关键点是否在边界框的合理范围内。

        允许关键点在 bbox 外 20% 的容差范围内（考虑关键点定义差异）。
        """
        bx, by = bbox.get("x", 0), bbox.get("y", 0)
        bw, bh = bbox.get("width", 0), bbox.get("height", 0)

        if bw <= 0 or bh <= 0:
            return

        # 计算容差扩展后的 bbox 范围
        margin_x = bw * self.BBOX_TOLERANCE_RATIO
        margin_y = bh * self.BBOX_TOLERANCE_RATIO

        x_min = bx - margin_x
        x_max = bx + bw + margin_x
        y_min = by - margin_y
        y_max = by + bh + margin_y

        for name, coords in landmarks.items():
            if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                continue

            lx, ly = coords[0], coords[1]

            # 跳过无效坐标
            if lx < 0 and ly < 0:
                continue

            if not (x_min <= lx <= x_max and y_min <= ly <= y_max):
                self.warnings.append(
                    f"landmarks.{name} ({lx}, {ly}) 显著偏离 bbox "
                    f"范围 [{bx},{by}]-[{bx+bw},{by+bh}]"
                )

    def validate_directory(
        self, annotation_dir: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """
        批量校验目录下所有 JSON 标注文件。

        Args:
            annotation_dir: 标注文件目录

        Returns:
            List[Dict]: 每个文件的校验结果
        """
        directory = Path(annotation_dir)

        if not directory.exists():
            return [
                {
                    "file": str(directory),
                    "status": "invalid",
                    "errors": [f"目录不存在: {directory}"],
                    "warnings": [],
                }
            ]

        json_files = sorted(directory.rglob("*_annotation.json"))
        if not json_files:
            print(f"目录中未找到标注 JSON 文件: {directory}")
            return []

        results = []
        for json_path in json_files:
            results.append(self.validate_file(json_path))

        return results

    @staticmethod
    def generate_validation_report(
        results: List[Dict[str, Any]],
    ) -> str:
        """
        生成人类可读的校验报告。

        Args:
            results: validate_file() 或 validate_directory() 的返回结果列表

        Returns:
            str: 格式化的校验报告
        """
        total = len(results)
        valid_count = sum(1 for r in results if r["status"] == "valid")
        invalid_count = total - valid_count

        lines = [
            "=" * 60,
            "          标注文件校验报告",
            "=" * 60,
            f"校验文件总数: {total:>5}",
            f"通过 (valid):   {valid_count:>5}",
            f"未通过 (invalid): {invalid_count:>5}",
            "=" * 60,
        ]

        # 汇总所有错误
        all_errors = []
        all_warnings = []
        for r in results:
            for err in r.get("errors", []):
                all_errors.append((r["file"], err))
            for warn in r.get("warnings", []):
                all_warnings.append((r["file"], warn))

        if all_errors:
            lines.append(f"\n❌ 错误 ({len(all_errors)} 条):")
            lines.append("-" * 40)
            for filepath, err in all_errors:
                filename = Path(filepath).name
                lines.append(f"  [{filename}] {err}")

        if all_warnings:
            lines.append(f"\n⚠️  警告 ({len(all_warnings)} 条):")
            lines.append("-" * 40)
            for filepath, warn in all_warnings:
                filename = Path(filepath).name
                lines.append(f"  [{filename}] {warn}")

        if not all_errors and not all_warnings:
            lines.append("\n✅ 所有标注文件校验通过！")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# 标注质量评估器
# ============================================================


class AnnotationQualityAssessor:
    """
    标注质量评估器，综合评估图片质量和标注合理性。

    在 AnnotationValidator 的结构校验基础上，增加以下评估维度：
      1. 图片质量（模糊度 / 亮度 / 姿态）—— 依赖 QualityFilter
      2. 人脸框面积占比 — bbox 过小或过大时扣分
      3. 关键点相对位置约束 — 验证眼鼻口的垂直层级关系

    Attributes:
        quality_filter: QualityFilter 实例
        validator: AnnotationValidator 实例
        min_face_ratio: 最小人脸面积占比（默认 5%）
        max_face_ratio: 最大人脸面积占比（默认 80%）
    """

    def __init__(
        self,
        quality_filter=None,
        validator: Optional[AnnotationValidator] = None,
        min_face_ratio: float = 0.05,
        max_face_ratio: float = 0.80,
    ):
        """
        初始化质量评估器。

        Args:
            quality_filter: QualityFilter 实例，若为 None 则延迟导入创建
            validator: AnnotationValidator 实例，若为 None 则自动创建
            min_face_ratio: bbox 面积占图片面积的最小比例，低于此值可能漏检
            max_face_ratio: bbox 面积占图片面积的最大比例，高于此值可能误检
        """
        self.validator = validator or AnnotationValidator()
        self.min_face_ratio = min_face_ratio
        self.max_face_ratio = max_face_ratio

        # 延迟导入 QualityFilter 避免循环依赖
        if quality_filter is None:
            from .quality_filter import QualityFilter

            self.quality_filter = QualityFilter()
        else:
            self.quality_filter = quality_filter

    def assess(
        self,
        annotation: Dict[str, Any],
        image: Optional[np.ndarray] = None,
        image_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        综合评估标注质量。

        Args:
            annotation: 标注字典
            image: 可选的 numpy 图片数组（BGR 格式），若为 None 则自动加载
            image_path: 图片路径，仅在 image 为 None 时使用

        Returns:
            Dict[str, Any]:
            {
                "quality_score": float,        # 0-1 综合质量评分
                "quality_label": str,          # "good" / "acceptable" / "poor"
                "image_quality": {...},        # 图片质量评估结果
                "face_ratio_score": float,     # 人脸占比评分 (0-1)
                "landmark_position_score": float,  # 关键点位置评分 (0-1)
                "annotation_issues": [...],    # 标注层面的问题列表
                "verdict": str,                # "accept" / "review" / "reject"
            }
        """
        issues = []
        scores = {}

        # ---- 1. 图片质量评估 ----
        if image is None and image_path is not None:
            try:
                image = load_image(image_path)
            except Exception:
                image = None

        if image is not None:
            landmarks_for_pose = annotation.get("landmarks")
            img_quality = self.quality_filter.assess(image, landmarks_for_pose)
            scores["image_quality_score"] = img_quality["quality_score"]
        else:
            # 无法加载图片，跳过图片质量评估
            img_quality = {
                "quality_score": 1.0,
                "quality_label": "good",
                "blur_score": 1.0,
                "brightness_score": 1.0,
                "pose_score": 1.0,
                "details": {"blur": None, "brightness": None, "pose": None},
            }
            scores["image_quality_score"] = 1.0

        # ---- 2. 人脸框面积占比评估 ----
        face_ratio_score = self._assess_face_ratio(annotation, image)
        scores["face_ratio_score"] = face_ratio_score

        # ---- 3. 关键点相对位置约束 ----
        landmarks = annotation.get("landmarks", {})
        landmark_pos_score = self._assess_landmark_positions(landmarks)
        scores["landmark_position_score"] = landmark_pos_score

        if landmark_pos_score < 0.5:
            issues.append("关键点相对位置异常（眼-鼻-口层级关系不符合预期）")

        # ---- 4. 综合评分 ----
        # 权重：图片质量 0.4，人脸占比 0.3，关键点位置 0.3
        quality_score = (
            0.4 * scores["image_quality_score"]
            + 0.3 * scores["face_ratio_score"]
            + 0.3 * scores["landmark_position_score"]
        )
        quality_score = round(min(1.0, max(0.0, quality_score)), 4)

        # 质量等级
        if quality_score >= 0.7:
            quality_label = "good"
            verdict = "accept"
        elif quality_score >= 0.4:
            quality_label = "acceptable"
            verdict = "review"
        else:
            quality_label = "poor"
            verdict = "reject"

        return {
            "quality_score": quality_score,
            "quality_label": quality_label,
            "image_quality": img_quality,
            "face_ratio_score": round(face_ratio_score, 4),
            "landmark_position_score": round(landmark_pos_score, 4),
            "annotation_issues": issues,
            "verdict": verdict,
        }

    def assess_file(
        self,
        annotation_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        对单个标注文件执行质量评估。

        会自动加载标注 JSON 和对应的图片文件。

        Args:
            annotation_path: 标注 JSON 文件路径

        Returns:
            Dict: 包含校验结果和质量评估结果的完整字典
        """
        path = Path(annotation_path)

        # 基本校验
        validation_result = self.validator.validate_file(path)
        if validation_result["status"] == "invalid":
            return {
                "file": str(path),
                "status": "invalid",
                "validation": validation_result,
                "quality": None,
                "error": "标注文件校验未通过，跳过质量评估",
            }

        # 加载标注
        with open(path, "r", encoding="utf-8") as f:
            annotation = json.load(f)

        # 解析图片路径并加载
        image = None
        image_path = annotation.get("image_path", "")
        if image_path:
            full_image_path = get_project_root() / image_path
            if full_image_path.exists():
                try:
                    image = load_image(full_image_path)
                except Exception as e:
                    logger.warning(f"无法加载图片 {full_image_path}: {e}")

        # 质量评估
        quality_result = self.assess(annotation, image=image)

        return {
            "file": str(path),
            "status": "valid",
            "validation": validation_result,
            "quality": quality_result,
        }

    def assess_directory(
        self,
        annotation_dir: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """
        批量评估目录下所有标注文件的质量。

        Args:
            annotation_dir: 标注文件目录

        Returns:
            List[Dict]: 每个文件的质量评估结果
        """
        directory = Path(annotation_dir)

        if not directory.exists():
            return [
                {
                    "file": str(directory),
                    "status": "invalid",
                    "validation": None,
                    "quality": None,
                    "error": f"目录不存在: {directory}",
                }
            ]

        json_files = sorted(directory.rglob("*_annotation.json"))
        results = []
        for json_path in json_files:
            results.append(self.assess_file(json_path))

        return results

    # ================================================================
    # 内部评估方法
    # ================================================================

    def _assess_face_ratio(
        self,
        annotation: Dict[str, Any],
        image: Optional[np.ndarray] = None,
    ) -> float:
        """
        评估人脸边界框面积占图片面积的比例是否合理。

        Args:
            annotation: 标注字典
            image: 可选的图片数组，若为 None 则仅使用 bbox 尺寸做基本判断

        Returns:
            float: 0-1 的人脸占比评分
        """
        bbox = annotation.get("bbox", {})
        bw = bbox.get("width", 0)
        bh = bbox.get("height", 0)

        if bw <= 0 or bh <= 0:
            return 0.0

        bbox_area = bw * bh

        if image is not None:
            img_h, img_w = image.shape[:2]
            img_area = img_w * img_h
            if img_area > 0:
                ratio = bbox_area / img_area

                # 理想范围：5%-80%
                if self.min_face_ratio <= ratio <= self.max_face_ratio:
                    # 在 10%-40% 之间最优
                    if 0.10 <= ratio <= 0.40:
                        return 1.0
                    else:
                        # 偏离最优区间但仍在可接受范围
                        return 0.8
                elif ratio < self.min_face_ratio:
                    # 人脸太小，线性衰减
                    if ratio <= 0:
                        return 0.0
                    return ratio / self.min_face_ratio * 0.5
                else:
                    # 人脸太大，线性衰减
                    if ratio >= 1.0:
                        return 0.0
                    return (1.0 - ratio) / (1.0 - self.max_face_ratio) * 0.5
            else:
                return 0.5
        else:
            # 无图片信息，仅基于 bbox 绝对尺寸做保守判断
            # 太小的 bbox（< 20x20 像素）可能是误检
            if bw < 20 or bh < 20:
                return 0.3
            return 0.7

    def _assess_landmark_positions(
        self, landmarks: Dict[str, Any]
    ) -> float:
        """
        验证关键点的相对位置约束。

        期望的垂直层级关系（从上到下）：
          eyes（上）→ nose_tip（中）→ mouth_corners（下）

        水平关系：
          left_eye 在 right_eye 的左侧（观察者视角）
          left_mouth_corner 在 right_mouth_corner 的左侧

        Args:
            landmarks: 关键点字典

        Returns:
            float: 0-1 的关键点位置合理性评分
        """
        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")
        nose = landmarks.get("nose_tip")
        left_mouth = landmarks.get("left_mouth_corner")
        right_mouth = landmarks.get("right_mouth_corner")

        # 检查所有关键点是否存在
        required = [left_eye, right_eye, nose, left_mouth, right_mouth]
        if any(p is None or len(p) != 2 for p in required):
            return 0.0

        # 检查是否为 partial 标注（全部为 [-1,-1]）
        all_invalid = all(
            p[0] < 0 and p[1] < 0
            for p in [left_eye, right_eye, nose, left_mouth, right_mouth]
        )
        if all_invalid:
            # partial 标注，跳过位置检查
            return 0.5

        checks_passed = 0
        checks_total = 0

        # 1. 眼鼻垂直关系：eye_y < nose_y（眼睛在鼻子上面）
        if left_eye[1] >= 0 and nose[1] >= 0:
            checks_total += 1
            if left_eye[1] < nose[1]:
                checks_passed += 1

        if right_eye[1] >= 0 and nose[1] >= 0:
            checks_total += 1
            if right_eye[1] < nose[1]:
                checks_passed += 1

        # 2. 鼻嘴垂直关系：nose_y < mouth_y（鼻子在嘴巴上面）
        if nose[1] >= 0 and left_mouth[1] >= 0:
            checks_total += 1
            if nose[1] < left_mouth[1]:
                checks_passed += 1

        if nose[1] >= 0 and right_mouth[1] >= 0:
            checks_total += 1
            if nose[1] < right_mouth[1]:
                checks_passed += 1

        # 3. 水平关系：left_eye.x < right_eye.x（观察者视角）
        if left_eye[0] >= 0 and right_eye[0] >= 0:
            checks_total += 1
            if left_eye[0] < right_eye[0]:
                checks_passed += 1

        # 4. 水平关系：left_mouth.x < right_mouth.x（观察者视角）
        if left_mouth[0] >= 0 and right_mouth[0] >= 0:
            checks_total += 1
            if left_mouth[0] < right_mouth[0]:
                checks_passed += 1

        if checks_total == 0:
            return 0.5

        return checks_passed / checks_total


# ============================================================
# 命令行入口
# ============================================================


def main():
    """标注文件校验工具的命令行入口。"""
    parser = argparse.ArgumentParser(
        description="标注文件校验工具 — 检查 JSON 标注文件的完整性和格式正确性",
    )
    parser.add_argument(
        "path",
        help="单个标注 JSON 文件路径，或标注文件目录路径",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：将警告也视为错误",
    )

    args = parser.parse_args()
    path = Path(args.path)

    validator = AnnotationValidator(strict=args.strict)

    if path.is_dir():
        results = validator.validate_directory(path)
    else:
        results = [validator.validate_file(path)]

    if not results:
        print("没有找到可校验的文件。")
        return

    print(validator.generate_validation_report(results))

    # 退出码：存在 invalid 则非零
    has_invalid = any(r["status"] == "invalid" for r in results)
    sys.exit(1 if has_invalid else 0)


if __name__ == "__main__":
    main()
