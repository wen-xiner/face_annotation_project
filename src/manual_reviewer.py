"""
文件名: manual_reviewer.py
功能: 人工标注修正器 — 从 auto_labeled 加载标注，支持交互式修正，
      将审核后的结果保存到 manually_reviewed/ 并记录修改历史
"""

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .utils import (
    LANDMARK_REQUIRED_KEYS,
    build_annotation_dict,
    get_project_root,
    load_annotation_json,
    save_annotation_json,
)
from .visualizer import Visualizer


class ManualReviewer:
    """
    人工标注修正器，提供标注数据的查看、修改和审核保存功能。

    工作流：
    1. 从 output/annotations/auto_labeled/ 加载自动标注
    2. 查看/修改 bbox 和 landmarks
    3. 保存到 output/annotations/manually_reviewed/
    4. 自动记录修改历史（modifications 字段）

    Attributes:
        auto_dir: 自动标注文件目录
        reviewed_dir: 人工修正后输出目录
        visualizer: 可视化工具实例
    """

    def __init__(
        self,
        auto_dir: Union[str, Path] = "output/annotations/auto_labeled",
        reviewed_dir: Union[str, Path] = "output/annotations/manually_reviewed",
    ):
        """
        初始化人工修正器。

        Args:
            auto_dir: 自动标注 JSON 文件的输入目录
            reviewed_dir: 人工修正后 JSON 文件的输出目录
        """
        self.auto_dir = Path(auto_dir)
        self.reviewed_dir = Path(reviewed_dir)
        self.reviewed_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = Visualizer()

    def list_pending(self) -> List[Path]:
        """
        列出所有尚未人工审核的自动标注文件。

        判断标准：manually_reviewed/ 下还不存在同名的标注文件。

        Returns:
            List[Path]: 待审核的标注文件路径列表
        """
        if not self.auto_dir.exists():
            return []

        pending = []
        for auto_file in sorted(self.auto_dir.rglob("*_annotation.json")):
            reviewed_file = self.reviewed_dir / auto_file.name
            if not reviewed_file.exists():
                pending.append(auto_file)

        return pending

    def load_for_review(
        self, annotation_path: Union[str, Path]
    ) -> Tuple[Dict[str, Any], Path]:
        """
        加载一个标注文件待人工审核。

        Args:
            annotation_path: 自动标注 JSON 文件路径

        Returns:
            Tuple[Dict, Path]: (标注数据字典, 对应的输出路径)

        Raises:
            FileNotFoundError: 标注文件不存在
        """
        path = Path(annotation_path)
        if not path.exists():
            raise FileNotFoundError(f"标注文件不存在: {path}")

        annotation = load_annotation_json(path)
        output_path = self.reviewed_dir / path.name

        return annotation, output_path

    def review(
        self,
        annotation: Dict[str, Any],
        modifications: Optional[Dict[str, Any]] = None,
        reviewer_name: str = "manual",
    ) -> Dict[str, Any]:
        """
        对标注数据应用人工修正并生成审核后的标注。

        Args:
            annotation: 原始自动标注字典
            modifications: 要修改的字段字典，支持以下键：
                - bbox: Dict[str, int]  新的边界框（全部或部分字段）
                - landmarks: Dict[str, List[int]]  新的关键点（全部或部分点）
                - 其他自定义字段
            reviewer_name: 审核者标识

        Returns:
            Dict: 包含修改记录的完整标注字典
        """
        # 深复制以避免修改原始数据
        reviewed = json.loads(json.dumps(annotation))

        # 记录修改历史
        changes = []

        # 应用 bbox 修改
        if modifications and "bbox" in modifications:
            old_bbox = dict(reviewed.get("bbox", {}))
            new_bbox = {**old_bbox, **modifications["bbox"]}
            reviewed["bbox"] = new_bbox

            for key, new_val in modifications["bbox"].items():
                old_val = old_bbox.get(key)
                if old_val != new_val:
                    changes.append(
                        {"field": f"bbox.{key}", "old": old_val, "new": new_val}
                    )

        # 应用 landmarks 修改
        if modifications and "landmarks" in modifications:
            old_landmarks = dict(reviewed.get("landmarks", {}))
            new_landmarks = {**old_landmarks, **modifications["landmarks"]}
            reviewed["landmarks"] = new_landmarks

            for name, new_coords in modifications["landmarks"].items():
                old_coords = old_landmarks.get(name)
                if old_coords != new_coords:
                    changes.append(
                        {
                            "field": f"landmarks.{name}",
                            "old": old_coords,
                            "new": new_coords,
                        }
                    )

        # 应用其他自定义字段
        if modifications:
            for key, val in modifications.items():
                if key not in ("bbox", "landmarks"):
                    old_val = reviewed.get(key)
                    if old_val != val:
                        changes.append(
                            {"field": key, "old": old_val, "new": val}
                        )
                    reviewed[key] = val

        # 更新元信息
        reviewed["annotator"] = "manual"
        reviewed["reviewer"] = reviewer_name
        reviewed["date"] = date.today().isoformat()
        reviewed["reviewed_from"] = annotation.get("annotator", "auto")

        # 添加修改记录
        if changes:
            reviewed["modifications"] = changes
        else:
            reviewed["modifications"] = []

        return reviewed

    def save_review(
        self,
        reviewed_annotation: Dict[str, Any],
        output_path: Union[str, Path],
        create_viz: bool = True,
    ) -> Path:
        """
        保存人工修正后的标注文件。

        Args:
            reviewed_annotation: 审核后的标注字典
            output_path: 输出 JSON 文件路径
            create_viz: 是否同时生成可视化对比图

        Returns:
            Path: 已保存的 JSON 文件路径
        """
        output = Path(output_path)
        save_annotation_json(reviewed_annotation, output)
        print(f"[OK] 人工修正标注已保存: {output}")

        # 生成可视化图片
        if create_viz:
            try:
                image_path = self._resolve_image_path(reviewed_annotation)
                if image_path and image_path.exists():
                    viz_image = self.visualizer.draw_annotation(
                        image_path, output
                    )
                    viz_dir = get_project_root() / "output" / "visualizations"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    viz_output = viz_dir / f"{output.stem}_reviewed.jpg"
                    self.visualizer.save_visualization(viz_image, viz_output)
            except Exception as e:
                print(f"  ⚠️  可视化生成失败: {e}")

        return output

    def _resolve_image_path(self, annotation: Dict[str, Any]) -> Optional[Path]:
        """
        根据标注中的 image_path 解析原始图片的绝对路径。

        支持 dataset/raw_images/ 和 web_app/uploads/ 两种来源。

        Args:
            annotation: 标注字典

        Returns:
            Optional[Path]: 图片绝对路径，若找不到则返回 None
        """
        image_path = annotation.get("image_path", "")
        if not image_path:
            return None

        project_root = get_project_root()
        candidate = project_root / image_path
        if candidate.exists():
            return candidate

        # 尝试在 web_app/uploads 下查找同名文件
        filename = Path(image_path).name
        alt_path = project_root / "web_app" / "uploads" / filename
        if alt_path.exists():
            return alt_path

        return None

    def batch_review_all(
        self,
        modifications_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        对全部待审核标注应用修正（批量模式）。

        Args:
            modifications_map: {文件名: 修正字典} 的映射，
                              文件名不含路径，如 {"11_annotation.json": {"bbox": {...}}}

        Returns:
            Dict: {"reviewed": int, "skipped": int, "details": List[Dict]}
        """
        pending = self.list_pending()
        results = {"reviewed": 0, "skipped": 0, "details": []}

        for auto_file in pending:
            detail = {
                "file": str(auto_file),
                "status": "skipped",
                "output_path": None,
                "error": None,
            }

            mods = None
            if modifications_map:
                mods = modifications_map.get(auto_file.name)

            try:
                annotation, output_path = self.load_for_review(auto_file)

                if mods is None:
                    # 无修改时直接标记为已审核（信任自动标注）
                    reviewed = self.review(annotation)
                else:
                    reviewed = self.review(annotation, modifications=mods)

                self.save_review(reviewed, output_path)
                detail["status"] = "reviewed"
                detail["output_path"] = str(output_path)
                results["reviewed"] += 1
            except Exception as e:
                detail["error"] = str(e)
                results["skipped"] += 1

            results["details"].append(detail)

        return results

    def compare(
        self, annotation_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """
        对比自动标注与人工修正（如果已存在）之间的差异。

        Args:
            annotation_path: 自动标注 JSON 文件路径

        Returns:
            Optional[Dict]: 对比结果，若人工修正尚未生成则返回 None
        """
        auto_file = Path(annotation_path)
        reviewed_file = self.reviewed_dir / auto_file.name

        if not reviewed_file.exists():
            return None

        auto = load_annotation_json(auto_file)
        reviewed = load_annotation_json(reviewed_file)

        diff = {
            "file": str(auto_file),
            "auto_bbox": auto.get("bbox"),
            "reviewed_bbox": reviewed.get("bbox"),
            "auto_landmarks": auto.get("landmarks"),
            "reviewed_landmarks": reviewed.get("landmarks"),
            "modifications": reviewed.get("modifications", []),
            "auto_annotator": auto.get("annotator"),
            "reviewed_annotator": reviewed.get("annotator"),
        }

        return diff


# ============================================================
# 交互式命令行编辑器
# ============================================================


def _prompt_int(prompt: str, default: int) -> int:
    """提示用户输入整数，直接回车使用默认值。"""
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"    输入无效，使用默认值 {default}")
        return default


def _interactive_edit_bbox(annotation: Dict[str, Any]) -> Dict[str, int]:
    """交互式编辑 bbox。"""
    bbox = annotation.get("bbox", {})
    print("\n  📐 当前边界框 (bbox):")
    print(f"     x={bbox.get('x')}, y={bbox.get('y')}, "
          f"width={bbox.get('width')}, height={bbox.get('height')}")

    print("  （直接回车保留原值，输入新值后回车修改）")
    new_bbox = {}
    for key in ("x", "y", "width", "height"):
        new_val = _prompt_int(f"{key}:", bbox.get(key, 0))
        if new_val != bbox.get(key):
            new_bbox[key] = new_val

    return new_bbox


def _interactive_edit_landmarks(annotation: Dict[str, Any]) -> Dict[str, List[int]]:
    """交互式编辑 landmarks 关键点。"""
    landmarks = annotation.get("landmarks", {})
    labels = {
        "left_eye": "👁 左眼",
        "right_eye": "👁 右眼",
        "nose_tip": "👃 鼻尖",
        "left_mouth_corner": "👄 左嘴角",
        "right_mouth_corner": "👄 右嘴角",
    }

    print("\n  📍 当前关键点 (landmarks):")
    for name in LANDMARK_REQUIRED_KEYS:
        coords = landmarks.get(name, [-1, -1])
        label = labels.get(name, name)
        print(f"     {label}: [{coords[0]}, {coords[1]}]")

    print("\n  输入要修改的关键点编号（直接回车跳过）:")
    print("    1=左眼  2=右眼  3=鼻尖  4=左嘴角  5=右嘴角")
    choice = input("  选择 [1-5]: ").strip()

    key_map = {
        "1": "left_eye",
        "2": "right_eye",
        "3": "nose_tip",
        "4": "left_mouth_corner",
        "5": "right_mouth_corner",
    }

    name = key_map.get(choice)
    if name is None:
        return {}

    old_coords = landmarks.get(name, [-1, -1])
    label = labels.get(name, name)
    print(f"\n  修改 {label}:")
    new_x = _prompt_int("x", old_coords[0])
    new_y = _prompt_int("y", old_coords[1])

    if [new_x, new_y] != old_coords:
        return {name: [new_x, new_y]}
    return {}


def _interactive_edit_annotator(annotation: Dict[str, Any]) -> str:
    """交互式设置审核者标识。"""
    current = annotation.get("annotator", "auto")
    print(f"\n  当前标注者: {current}")
    name = input("  审核者标识（直接回车使用 'manual'）: ").strip()
    return name if name else "manual"


def interactive_review(annotation_path: str, show_viz: bool = True) -> None:
    """
    交互式人工审核一个标注文件。

    在命令行中引导用户逐步查看和修改 bbox、landmarks，
    修改完成后保存到 manually_reviewed/ 目录。

    Args:
        annotation_path: 自动标注 JSON 文件路径
        show_viz: 是否在修改前打开可视化窗口
    """
    reviewer = ManualReviewer()
    path = Path(annotation_path)

    if not path.exists():
        print(f"❌ 标注文件不存在: {path}")
        return

    annotation, output_path = reviewer.load_for_review(path)

    # 显示标注概览
    print("=" * 56)
    print("  人工标注修正工具 — 交互模式")
    print("=" * 56)
    print(f"  文件: {path.name}")
    print(f"  图片路径: {annotation.get('image_path', 'N/A')}")
    print(f"  原始标注者: {annotation.get('annotator', 'N/A')}")
    print(f"  置信度: {annotation.get('confidence', 'N/A')}")

    # 可选：显示可视化窗口
    if show_viz:
        try:
            image_path = reviewer._resolve_image_path(annotation)
            if image_path and image_path.exists():
                viz_image = reviewer.visualizer.draw_annotation(image_path, path)
                print("\n  📷 正在显示可视化窗口（按任意键继续）...")
                reviewer.visualizer.show_visualization(viz_image,
                    window_name=f"原始标注 — {path.name}")
        except Exception as e:
            print(f"  ⚠️  无法显示可视化: {e}")

    # 交互式修改
    mods: Dict[str, Any] = {}

    while True:
        print("\n" + "-" * 40)
        print("  请选择操作:")
        print("    1 = 修改 bbox（边界框）")
        print("    2 = 修改 landmarks（关键点）")
        print("    3 = 设置审核者标识")
        print("    4 = 保存并退出")
        print("    0 = 取消（不保存）")
        choice = input("  选择 [0-4]: ").strip()

        if choice == "0":
            print("\n  ❌ 已取消，未保存任何修改。")
            return

        if choice == "1":
            bbox_changes = _interactive_edit_bbox(annotation)
            if bbox_changes:
                mods.setdefault("bbox", {}).update(bbox_changes)
                print(f"  ✅ bbox 修改已记录: {bbox_changes}")
            else:
                print("  ℹ️  bbox 无变化")

        elif choice == "2":
            lm_changes = _interactive_edit_landmarks(annotation)
            if lm_changes:
                mods.setdefault("landmarks", {}).update(lm_changes)
                print(f"  ✅ landmarks 修改已记录: {lm_changes}")
            else:
                print("  ℹ️  landmarks 无变化")

        elif choice == "3":
            reviewer_name = _interactive_edit_annotator(annotation)
            mods["reviewer_name"] = reviewer_name
            print(f"  ✅ 审核者设置为: {reviewer_name}")

        elif choice == "4":
            # 审核并保存
            reviewer_name = mods.pop("reviewer_name", "manual")
            reviewed = reviewer.review(annotation, modifications=mods, reviewer_name=reviewer_name)
            reviewer.save_review(reviewed, output_path)

            if reviewed.get("modifications"):
                print(f"\n  📝 本次修改记录 ({len(reviewed['modifications'])} 项):")
                for change in reviewed["modifications"]:
                    print(f"     - {change['field']}: {change['old']} → {change['new']}")
            else:
                print("\n  ℹ️  无修改，已标记为人工审核通过")

            print("\n  ✅ 人工修正完成！")
            return

        else:
            print("  ⚠️  无效选择，请重新输入")


# ============================================================
# 命令行入口
# ============================================================


def main():
    """人工标注修正工具的命令行入口。"""
    parser = argparse.ArgumentParser(
        description="人工标注修正工具 — 查看、修改并审核自动标注结果",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ----- list 子命令 -----
    list_parser = subparsers.add_parser("list", help="列出所有待审核的标注文件")
    list_parser.add_argument(
        "--auto-dir",
        default="output/annotations/auto_labeled",
        help="自动标注目录",
    )

    # ----- review 子命令 -----
    review_parser = subparsers.add_parser(
        "review", help="交互式审核单个标注文件"
    )
    review_parser.add_argument("annotation_path", help="标注 JSON 文件路径")
    review_parser.add_argument(
        "--no-viz",
        action="store_true",
        help="不显示可视化窗口",
    )

    # ----- approve 子命令 -----
    approve_parser = subparsers.add_parser(
        "approve", help="直接批准自动标注（不修改，仅标记为已审核）"
    )
    approve_parser.add_argument("annotation_path", help="标注 JSON 文件路径")

    # ----- batch 子命令 -----
    batch_parser = subparsers.add_parser(
        "batch", help="批量批准所有待审核的自动标注"
    )
    batch_parser.add_argument(
        "--auto-dir",
        default="output/annotations/auto_labeled",
        help="自动标注目录",
    )

    # ----- compare 子命令 -----
    compare_parser = subparsers.add_parser(
        "compare", help="对比自动标注与人工修正的差异"
    )
    compare_parser.add_argument("annotation_path", help="自动标注 JSON 文件路径")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    reviewer = ManualReviewer()

    if args.command == "list":
        pending = reviewer.list_pending()
        print(f"\n待审核标注文件: {len(pending)} 个\n")
        if pending:
            for f in pending:
                print(f"  📄 {f}")
        else:
            print("  ✅ 全部已审核！")

    elif args.command == "review":
        interactive_review(args.annotation_path, show_viz=not args.no_viz)

    elif args.command == "approve":
        path = Path(args.annotation_path)
        if not path.exists():
            print(f"❌ 标注文件不存在: {path}")
            sys.exit(1)

        annotation, output_path = reviewer.load_for_review(path)
        reviewed = reviewer.review(annotation)
        reviewer.save_review(reviewed, output_path)
        print("\n✅ 已批准（无修改，标记为已审核）")

    elif args.command == "batch":
        print("\n正在批量批准所有待审核标注...")
        results = reviewer.batch_review_all()
        print(f"\n审核完成: {results['reviewed']} 个通过, {results['skipped']} 个跳过")

    elif args.command == "compare":
        diff = reviewer.compare(args.annotation_path)
        if diff is None:
            print("ℹ️  尚未生成人工修正版本")
        else:
            print(f"\n{'='*50}")
            print(f"  对比: {Path(diff['file']).name}")
            print(f"{'='*50}")
            print(f"  自动标注者: {diff['auto_annotator']}")
            print(f"  审核标注者: {diff['reviewed_annotator']}")
            if diff["modifications"]:
                print(f"\n  修改记录 ({len(diff['modifications'])} 项):")
                for change in diff["modifications"]:
                    print(f"    - {change['field']}: {change['old']} → {change['new']}")
            else:
                print("  无修改（信任自动标注）")


if __name__ == "__main__":
    main()
