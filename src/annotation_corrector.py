"""
文件名: annotation_corrector.py
功能: 标注自动纠错器，基于几何约束自动修正异常关键点坐标

支持的修正策略:
  1. 眼对称修正 — 利用左右眼应近似水平对齐的约束
  2. 鼻尖位置修正 — 鼻尖应在双眼连线中点正下方
  3. 嘴角对齐修正 — 利用嘴角应近似水平对齐的约束
  4. 关键点在 bbox 内的约束 — 修正超出 bbox 的坐标

使用方式:
  corrector = AnnotationCorrector()
  fixed = corrector.correct(annotation)
  # 批量修正
  results = corrector.correct_directory("output/annotations/auto_labeled/")
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .utils import load_annotation_json, save_annotation_json

logger = logging.getLogger(__name__)


class AnnotationCorrector:
    """
    标注自动纠错器。

    基于面部几何约束关系，自动检测并修正异常的关键点坐标。

    Attributes:
        max_eye_y_diff_ratio: 左右眼 Y 坐标差的最大允许比例（相对于眼间距）
        max_mouth_y_diff_ratio: 左右嘴角 Y 坐标差的最大允许比例（相对于嘴宽）
        max_correction_distance: 单次修正允许的最大位移（像素），防止过度修正
    """

    def __init__(
        self,
        max_eye_y_diff_ratio: float = 0.15,
        max_mouth_y_diff_ratio: float = 0.15,
        max_correction_distance: float = 50.0,
    ):
        """
        初始化纠错器。

        Args:
            max_eye_y_diff_ratio: 左右眼 Y 差 / 眼间距 的最大比例
            max_mouth_y_diff_ratio: 左右嘴角 Y 差 / 嘴宽 的最大比例
            max_correction_distance: 单次最大修正距离（像素）
        """
        self.max_eye_y_diff_ratio = max_eye_y_diff_ratio
        self.max_mouth_y_diff_ratio = max_mouth_y_diff_ratio
        self.max_correction_distance = max_correction_distance

    def correct(
        self,
        annotation: Dict[str, Any],
        bbox: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        对单条标注执行自动纠错。

        Args:
            annotation: 标注字典
            bbox: 边界框（若 annotation 中无 bbox 则使用此参数）

        Returns:
            Dict: 修正后的标注字典（深拷贝），包含修改记录
        """
        # 深拷贝
        corrected = json.loads(json.dumps(annotation))
        bbox = bbox or corrected.get("bbox", {})
        landmarks = corrected.get("landmarks", {})

        if not landmarks:
            return corrected

        modifications = []

        # 执行各项修正
        for strategy_name, strategy_fn in [
            ("eye_symmetry", self._correct_eye_symmetry),
            ("nose_position", self._correct_nose_position),
            ("mouth_alignment", self._correct_mouth_alignment),
            ("bbox_bounds", self._correct_bbox_bounds),
        ]:
            changes = strategy_fn(landmarks, bbox)
            for change in changes:
                field, old_val, new_val = change
                modifications.append(
                    {"field": f"landmarks.{field}", "old": old_val, "new": new_val}
                )

        # 记录修改
        if modifications:
            corrected["modifications"] = (
                corrected.get("modifications", []) + modifications
            )
            corrected["annotator"] = "manual"
            corrected["reviewer"] = "auto_corrector"
            corrected["reviewed_from"] = annotation.get("annotator", "auto")
            corrected["date"] = date.today().isoformat()
            logger.info(
                f"自动纠错完成: {len(modifications)} 处修正"
            )
        else:
            logger.debug("无需修正")

        return corrected

    def correct_file(
        self,
        annotation_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        对单个标注文件执行纠错。

        Args:
            annotation_path: 标注 JSON 文件路径
            output_path: 输出路径（默认覆盖原文件）
            save: 是否保存

        Returns:
            Dict: {"file": str, "corrections": int, "corrected": Dict}
        """
        annotation = load_annotation_json(annotation_path)
        corrected = self.correct(annotation)

        corrections = len(corrected.get("modifications", []))

        if save and corrections > 0:
            out = Path(output_path) if output_path else Path(annotation_path)
            save_annotation_json(corrected, out)

        return {
            "file": str(annotation_path),
            "corrections": corrections,
            "corrected": corrected,
        }

    def correct_directory(
        self,
        annotation_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        批量纠错目录下的所有标注文件。

        Args:
            annotation_dir: 标注文件目录
            output_dir: 输出目录（None 则覆盖原文件）
            dry_run: True 则只分析不保存

        Returns:
            Dict: {"total": int, "corrected": int, "corrections_total": int, "details": List}
        """
        directory = Path(annotation_dir)
        json_files = sorted(directory.rglob("*_annotation.json"))

        if not json_files:
            return {"total": 0, "corrected": 0, "corrections_total": 0, "details": []}

        results = {"total": len(json_files), "corrected": 0, "corrections_total": 0, "details": []}

        for json_path in json_files:
            try:
                if output_dir:
                    out_path = Path(output_dir) / json_path.name
                else:
                    out_path = json_path if not dry_run else None

                result = self.correct_file(json_path, out_path, save=not dry_run)
                results["details"].append(result)
                if result["corrections"] > 0:
                    results["corrected"] += 1
                    results["corrections_total"] += result["corrections"]
            except Exception as e:
                logger.error(f"纠错失败 {json_path.name}: {e}")
                results["details"].append(
                    {"file": str(json_path), "corrections": 0, "error": str(e)}
                )

        logger.info(
            f"批量纠错完成: {results['corrected']}/{results['total']} 个文件被修正, "
            f"共 {results['corrections_total']} 处修改"
        )
        return results

    # ================================================================
    # 内部修正策略
    # ================================================================

    def _correct_eye_symmetry(
        self,
        landmarks: Dict[str, List[int]],
        bbox: Dict[str, int],
    ) -> List[tuple]:
        """
        修正左右眼不对称问题。

        约束：左右眼应近似水平（Y 坐标差 ≤ 眼间距 × max_eye_y_diff_ratio）。
        若偏差超过阈值，将两眼 Y 坐标对齐到均值。
        """
        left = landmarks.get("left_eye")
        right = landmarks.get("right_eye")

        if not left or not right or len(left) != 2 or len(right) != 2:
            return []

        # 跳过无效点
        if (left[0] < 0 and left[1] < 0) or (right[0] < 0 and right[1] < 0):
            return []

        eye_dist = abs(right[0] - left[0])
        if eye_dist < 10:
            return []

        y_diff = abs(right[1] - left[1])
        max_allowed = eye_dist * self.max_eye_y_diff_ratio

        if y_diff > max_allowed:
            # 对齐到均值
            mean_y = int(round((left[1] + right[1]) / 2.0))
            changes = []

            if abs(left[1] - mean_y) <= self.max_correction_distance:
                old_left = list(left)
                left[1] = mean_y
                changes.append(("left_eye", old_left, list(left)))

            if abs(right[1] - mean_y) <= self.max_correction_distance:
                old_right = list(right)
                right[1] = mean_y
                changes.append(("right_eye", old_right, list(right)))

            return changes

        return []

    def _correct_nose_position(
        self,
        landmarks: Dict[str, List[int]],
        bbox: Dict[str, int],
    ) -> List[tuple]:
        """
        修正鼻尖位置异常。

        约束：鼻尖 X 坐标应接近双眼 X 坐标的中点；
             鼻尖 Y 坐标应在双眼之下、嘴巴之上。
        """
        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")
        nose = landmarks.get("nose_tip")
        left_mouth = landmarks.get("left_mouth_corner")
        right_mouth = landmarks.get("right_mouth_corner")

        if not all([left_eye, right_eye, nose]) or len(nose) != 2:
            return []

        # 跳过无效点
        if (left_eye[0] < 0 and left_eye[1] < 0) or (right_eye[0] < 0 and right_eye[1] < 0):
            return []
        if nose[0] < 0 and nose[1] < 0:
            return []

        changes = []

        # X 坐标：应在两眼中间
        mid_eye_x = int(round((left_eye[0] + right_eye[0]) / 2.0))
        eye_dist = abs(right_eye[0] - left_eye[0])
        x_dev = abs(nose[0] - mid_eye_x)

        if eye_dist > 10 and x_dev > eye_dist * 0.3:
            if x_dev <= self.max_correction_distance:
                old_nose = list(nose)
                nose[0] = mid_eye_x
                changes.append(("nose_tip", old_nose, list(nose)))

        # Y 坐标：应在眼睛和嘴巴之间
        # 先尝试从已有可信点推断范围
        mouth_y = None
        if left_mouth and right_mouth:
            if not (left_mouth[0] < 0 and left_mouth[1] < 0) and not (right_mouth[0] < 0 and right_mouth[1] < 0):
                mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0

        eye_y = (left_eye[1] + right_eye[1]) / 2.0

        if mouth_y and mouth_y > eye_y:
            expected_nose_y = int(round(eye_y + (mouth_y - eye_y) * 0.55))
            if abs(nose[1] - expected_nose_y) > (mouth_y - eye_y) * 0.3:
                if abs(nose[1] - expected_nose_y) <= self.max_correction_distance:
                    # 标记为已有修改
                    if not changes or changes[-1][0] != "nose_tip":
                        old_nose = list(nose)
                        nose[1] = expected_nose_y
                        changes.append(("nose_tip", old_nose, list(nose)))

        return changes

    def _correct_mouth_alignment(
        self,
        landmarks: Dict[str, List[int]],
        bbox: Dict[str, int],
    ) -> List[tuple]:
        """
        修正嘴角不对齐问题。

        约束：左右嘴角应近似水平（Y 坐标差 ≤ 嘴宽 × max_mouth_y_diff_ratio）。
        """
        left = landmarks.get("left_mouth_corner")
        right = landmarks.get("right_mouth_corner")

        if not left or not right or len(left) != 2 or len(right) != 2:
            return []

        if (left[0] < 0 and left[1] < 0) or (right[0] < 0 and right[1] < 0):
            return []

        mouth_width = abs(right[0] - left[0])
        if mouth_width < 10:
            return []

        y_diff = abs(right[1] - left[1])
        max_allowed = mouth_width * self.max_mouth_y_diff_ratio

        if y_diff > max_allowed:
            mean_y = int(round((left[1] + right[1]) / 2.0))
            changes = []

            if abs(left[1] - mean_y) <= self.max_correction_distance:
                old_left = list(left)
                left[1] = mean_y
                changes.append(("left_mouth_corner", old_left, list(left)))

            if abs(right[1] - mean_y) <= self.max_correction_distance:
                old_right = list(right)
                right[1] = mean_y
                changes.append(("right_mouth_corner", old_right, list(right)))

            return changes

        return []

    def _correct_bbox_bounds(
        self,
        landmarks: Dict[str, List[int]],
        bbox: Dict[str, int],
    ) -> List[tuple]:
        """
        修正关键点超出 bbox 范围的问题。

        将超出 bbox 20% 扩展范围的关键点约束回 bbox 内。
        """
        if not bbox:
            return []

        bx, by = bbox.get("x", 0), bbox.get("y", 0)
        bw, bh = bbox.get("width", 0), bbox.get("height", 0)

        if bw <= 0 or bh <= 0:
            return []

        margin_x = bw * 0.25
        margin_y = bh * 0.25
        x_min = bx - margin_x
        x_max = bx + bw + margin_x
        y_min = by - margin_y
        y_max = by + bh + margin_y

        changes = []
        for name, coords in landmarks.items():
            if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                continue
            if coords[0] < 0 and coords[1] < 0:
                continue

            new_coords = list(coords)
            modified = False

            if coords[0] < x_min and coords[0] >= 0:
                new_coords[0] = int(x_min)
                modified = True
            elif coords[0] > x_max:
                new_coords[0] = int(x_max)
                modified = True

            if coords[1] < y_min and coords[1] >= 0:
                new_coords[1] = int(y_min)
                modified = True
            elif coords[1] > y_max:
                new_coords[1] = int(y_max)
                modified = True

            if modified:
                old = list(coords)
                # 原地修改
                coords[0] = new_coords[0]
                coords[1] = new_coords[1]
                changes.append((name, old, list(coords)))

        return changes


# ============================================================
# 命令行入口
# ============================================================


def main():
    """标注自动纠错工具的命令行入口。"""
    parser = argparse.ArgumentParser(
        description="标注自动纠错工具 — 基于几何约束自动修正异常关键点",
    )
    parser.add_argument(
        "path",
        help="单个标注 JSON 文件或标注文件目录",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出目录（默认覆盖原文件）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式：仅显示将要修正的内容，不保存",
    )
    parser.add_argument(
        "--max-correction",
        type=float,
        default=50.0,
        help="单次最大修正距离（像素），默认 50",
    )

    args = parser.parse_args()
    path = Path(args.path)

    corrector = AnnotationCorrector(max_correction_distance=args.max_correction)

    if path.is_dir():
        result = corrector.correct_directory(
            path,
            output_dir=args.output,
            dry_run=args.dry_run,
        )
        print(f"\n扫描: {result['total']} 个文件")
        print(f"修正: {result['corrected']} 个文件")
        print(f"修改: {result['corrections_total']} 处")

        if args.dry_run and result["details"]:
            print("\n--- 预览 ---")
            for d in result["details"]:
                if d.get("corrections", 0) > 0:
                    print(f"  {Path(d['file']).name}: {d['corrections']} 处修正")
                    mods = d.get("corrected", {}).get("modifications", [])
                    for m in mods:
                        print(f"    {m['field']}: {m['old']} → {m['new']}")
            print("\n使用不带 --dry-run 参数运行以执行修正。")
    else:
        result = corrector.correct_file(
            path,
            output_path=args.output,
            save=not args.dry_run,
        )
        print(f"文件: {Path(result['file']).name}")
        print(f"修正: {result['corrections']} 处")
        if result["corrections"] > 0:
            for m in result["corrected"].get("modifications", []):
                print(f"  {m['field']}: {m['old']} → {m['new']}")


import argparse as _argparse

if __name__ == "__main__":
    main()
