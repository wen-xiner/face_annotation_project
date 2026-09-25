"""
文件名: run_review.py
功能: 批量人工审核执行脚本，将自动标注批量标记为人工审核通过

对 output/annotations/auto_labeled/ 下的所有自动标注 JSON 执行：
  1. AnnotationValidator 校验 → 发现结构性问题
  2. ManualReviewer.review() 信任自动结果 → 添加审核元信息
  3. ManualReviewer.save_review() → 保存到 manually_reviewed/

使用方式:
  python scripts/run_review.py              # 执行审核
  python scripts/run_review.py --dry-run    # 预览模式（不实际保存）
  python scripts/run_review.py --skip-viz   # 跳过可视化生成
  python scripts/run_review.py --reviewer "张三"  # 指定审核者名称
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.manual_reviewer import ManualReviewer
from src.validator import AnnotationValidator
from src.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(
        description="批量人工审核脚本 — 将自动标注信任标记为人工审核通过",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式：仅显示将要审核的文件列表，不实际保存",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="跳过标注可视化图片的生成（加速处理）",
    )
    parser.add_argument(
        "--reviewer",
        default="batch_review",
        help="审核者标识（默认: batch_review）",
    )
    parser.add_argument(
        "--auto-dir",
        default=str(PROJECT_ROOT / "output" / "annotations" / "auto_labeled"),
        help="自动标注目录（默认: output/annotations/auto_labeled）",
    )
    parser.add_argument(
        "--reviewed-dir",
        default=str(PROJECT_ROOT / "output" / "annotations" / "manually_reviewed"),
        help="审核后输出目录（默认: output/annotations/manually_reviewed）",
    )

    args = parser.parse_args()

    auto_dir = Path(args.auto_dir)
    reviewed_dir = Path(args.reviewed_dir)

    # ================================================================
    # 初始化工具
    # ================================================================
    print("=" * 60)
    print("      批量人工审核工具")
    print("=" * 60)
    print(f"自动标注目录: {auto_dir}")
    print(f"审核输出目录: {reviewed_dir}")
    print(f"审核者:       {args.reviewer}")
    print(f"预览模式:     {'是' if args.dry_run else '否'}")
    print(f"可视化生成:   {'否' if args.skip_viz else '是'}")
    print("=" * 60)

    if not auto_dir.exists():
        print(f"\n[ERROR] 自动标注目录不存在: {auto_dir}")
        sys.exit(1)

    reviewer = ManualReviewer(
        auto_dir=str(auto_dir),
        reviewed_dir=str(reviewed_dir),
    )
    validator = AnnotationValidator()
    visualizer = Visualizer()

    # ================================================================
    # 查找待审核文件
    # ================================================================
    json_files = sorted(auto_dir.rglob("*_annotation.json"))

    if not json_files:
        print(f"\n[ERROR] 目录中未找到标注文件: {auto_dir}")
        sys.exit(1)

    print(f"\n发现 {len(json_files)} 个自动标注文件")

    if args.dry_run:
        print("\n--- 预览文件列表 ---")
        for f in json_files:
            output_path = reviewed_dir / f.name
            exists = output_path.exists()
            status = "[SKIP] 已有审核" if exists else "[TODO] 待审核"
            print(f"  {status}: {f.name}")
        print(f"\n预览完成。共 {len(json_files)} 个文件。")
        print("使用不带 --dry-run 参数运行以执行实际审核。")
        return

    # ================================================================
    # 执行审核
    # ================================================================
    results = {
        "reviewed": 0,
        "skipped": 0,
        "errors": 0,
        "validation_issues": 0,
        "details": [],
    }

    start_time = datetime.now()

    for i, json_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] {json_path.name}")

        # 检查是否已存在审核结果
        annotation, output_path = reviewer.load_for_review(json_path)
        if output_path.exists():
            print(f"  [SKIP] 已存在审核结果，跳过")
            results["skipped"] += 1
            results["details"].append(
                {"file": str(json_path), "action": "skipped", "reason": "已存在"}
            )
            continue

        # 步骤1：校验
        try:
            validation = validator.validate_file(json_path)
            if validation["status"] == "invalid":
                print(f"  [WARN] 校验发现问题:")
                for err in validation.get("errors", []):
                    print(f"      ERROR: {err}")
                for warn in validation.get("warnings", []):
                    print(f"      WARN:  {warn}")
                results["validation_issues"] += 1
        except Exception as e:
            print(f"  [WARN] 校验失败: {e}")

        # 步骤2：审核（信任模式，无修改）
        try:
            reviewed = reviewer.review(
                annotation,
                modifications=None,
                reviewer_name=args.reviewer,
            )

            # 步骤3：保存
            save_path = reviewer.save_review(
                reviewed,
                output_path,
                create_viz=not args.skip_viz,
            )
            print(f"  [OK] 审核通过 -> {save_path.name}")
            results["reviewed"] += 1
            results["details"].append(
                {
                    "file": str(json_path),
                    "action": "reviewed",
                    "output": str(save_path),
                }
            )
        except Exception as e:
            print(f"  [ERROR] 审核失败: {e}")
            results["errors"] += 1
            results["details"].append(
                {
                    "file": str(json_path),
                    "action": "error",
                    "error": str(e),
                }
            )

    # ================================================================
    # 输出统计报告
    # ================================================================
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("              审核完成 -- 统计报告")
    print("=" * 60)
    print(f"处理文件总数:   {len(json_files):>5}")
    print(f"审核通过:       {results['reviewed']:>5}")
    print(f"跳过 (已存在):  {results['skipped']:>5}")
    print(f"失败:           {results['errors']:>5}")
    print(f"存在校验警告:   {results['validation_issues']:>5}")
    print(f"耗时:           {elapsed:.1f} 秒")
    print("=" * 60)

    # 校验问题的文件列出
    if results["validation_issues"] > 0:
        print("\n[WARN] 以下文件在校验时发现问题，建议人工复核：")
        for d in results["details"]:
            if d.get("action") == "reviewed":
                fname = Path(d["file"]).name
                v = validator.validate_file(Path(d["file"]))
                if v.get("warnings") or v.get("errors"):
                    print(f"  -> {fname}")
                    for w in v.get("warnings", []):
                        print(f"      {w}")

    print(f"\n审核结果保存在: {reviewed_dir}")


if __name__ == "__main__":
    main()
