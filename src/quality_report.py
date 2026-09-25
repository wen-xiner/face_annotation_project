"""
文件名: quality_report.py
功能: 标注质量报告生成器，对批量标注文件进行质量评估并生成统计报告

支持输出：
  1. 控制台文本报告 — 质量分布统计、各维度得分汇总、问题文件列表
  2. JSON 详细报告 — 完整的逐文件评估数据，便于程序化处理
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class QualityReport:
    """
    标注质量报告生成器。

    批量扫描标注目录，执行质量评估，生成统计分布报告。
    """

    def __init__(self):
        """初始化报告生成器，延迟导入评估器。"""
        self._assessor = None

    @property
    def assessor(self):
        """延迟加载 AnnotationQualityAssessor。"""
        if self._assessor is None:
            from .validator import AnnotationQualityAssessor

            self._assessor = AnnotationQualityAssessor()
        return self._assessor

    def generate(
        self,
        annotation_dir: Union[str, Path],
        output_json: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        对标注目录执行质量评估并生成报告。

        Args:
            annotation_dir: 标注文件目录
            output_json: 可选的 JSON 输出路径，用于保存详细报告

        Returns:
            Dict: 包含 summary 和 details 的完整报告
        """
        directory = Path(annotation_dir)
        json_files = sorted(directory.rglob("*_annotation.json"))

        if not json_files:
            print(f"目录中未找到标注 JSON 文件: {directory}")
            return {"summary": {}, "details": []}

        print(f"正在评估 {len(json_files)} 个标注文件...")

        # 逐文件评估
        details = []
        for i, json_path in enumerate(json_files, 1):
            print(f"  [{i}/{len(json_files)}] {json_path.name}")
            try:
                result = self.assessor.assess_file(json_path)
                details.append(result)
            except Exception as e:
                details.append(
                    {
                        "file": str(json_path),
                        "status": "error",
                        "validation": None,
                        "quality": None,
                        "error": str(e),
                    }
                )

        # 汇总统计
        summary = self._build_summary(details)

        report = {
            "generated_at": datetime.now().isoformat(),
            "annotation_dir": str(directory),
            "total_files": len(json_files),
            "summary": summary,
            "details": details,
        }

        # 可选：保存 JSON 详细报告
        if output_json is not None:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n详细报告已保存: {output_path}")

        return report

    def generate_text_report(
        self,
        annotation_dir: Union[str, Path],
    ) -> str:
        """
        生成人类可读的文本质量报告。

        Args:
            annotation_dir: 标注文件目录

        Returns:
            str: 格式化的文本报告
        """
        report = self.generate(annotation_dir)
        return self._format_text_report(report)

    def _build_summary(self, details: List[Dict]) -> Dict[str, Any]:
        """
        从逐文件评估结果中构建汇总统计。

        Args:
            details: 逐文件评估结果列表

        Returns:
            Dict: 汇总统计数据
        """
        total = len(details)
        valid = sum(1 for d in details if d.get("status") == "valid")
        invalid = sum(1 for d in details if d.get("status") == "invalid")
        errors = sum(1 for d in details if d.get("status") == "error")

        # 质量分布统计
        quality_labels = Counter()
        quality_scores = []
        verdicts = Counter()
        blur_scores = []
        brightness_scores = []
        pose_scores = []
        face_ratio_scores = []
        landmark_scores = []
        total_issues = 0
        poor_files = []
        review_files = []

        for d in details:
            q = d.get("quality")
            if q is not None:
                quality_labels[q.get("quality_label", "unknown")] += 1
                quality_scores.append(q.get("quality_score", 0))
                verdicts[q.get("verdict", "unknown")] += 1
                total_issues += len(q.get("annotation_issues", []))

                # 子分数
                img_q = q.get("image_quality", {})
                if img_q:
                    blur_scores.append(img_q.get("blur_score", 0))
                    brightness_scores.append(img_q.get("brightness_score", 0))
                    pose_scores.append(img_q.get("pose_score", 0))
                face_ratio_scores.append(q.get("face_ratio_score", 0))
                landmark_scores.append(q.get("landmark_position_score", 0))

                # 记录问题文件
                if q.get("verdict") == "reject":
                    poor_files.append(
                        {
                            "file": Path(d["file"]).name,
                            "score": q.get("quality_score"),
                            "issues": q.get("annotation_issues", []),
                        }
                    )
                elif q.get("verdict") == "review":
                    review_files.append(
                        {
                            "file": Path(d["file"]).name,
                            "score": q.get("quality_score"),
                            "issues": q.get("annotation_issues", []),
                        }
                    )

        # 计算分布统计
        scores_array = quality_scores

        return {
            "validation": {
                "total": total,
                "valid": valid,
                "invalid": invalid,
                "errors": errors,
            },
            "quality_distribution": {
                "by_label": dict(quality_labels),
                "by_verdict": dict(verdicts),
                "mean_score": (
                    round(sum(scores_array) / len(scores_array), 4)
                    if scores_array
                    else 0
                ),
                "min_score": round(min(scores_array), 4) if scores_array else 0,
                "max_score": round(max(scores_array), 4) if scores_array else 0,
                "total_issues_found": total_issues,
            },
            "sub_scores": {
                "mean_blur": (
                    round(sum(blur_scores) / len(blur_scores), 4)
                    if blur_scores
                    else 0
                ),
                "mean_brightness": (
                    round(sum(brightness_scores) / len(brightness_scores), 4)
                    if brightness_scores
                    else 0
                ),
                "mean_pose": (
                    round(sum(pose_scores) / len(pose_scores), 4)
                    if pose_scores
                    else 0
                ),
                "mean_face_ratio": (
                    round(sum(face_ratio_scores) / len(face_ratio_scores), 4)
                    if face_ratio_scores
                    else 0
                ),
                "mean_landmark_position": (
                    round(sum(landmark_scores) / len(landmark_scores), 4)
                    if landmark_scores
                    else 0
                ),
            },
            "poor_files_count": len(poor_files),
            "review_files_count": len(review_files),
            "poor_files": poor_files[:20],  # 最多展示20个
            "review_files": review_files[:20],
        }

    def _format_text_report(self, report: Dict) -> str:
        """
        将报告格式化为人类可读的文本。

        Args:
            report: generate() 返回的报告字典

        Returns:
            str: 格式化的文本报告
        """
        summary = report.get("summary", {})
        val = summary.get("validation", {})
        qd = summary.get("quality_distribution", {})
        sub = summary.get("sub_scores", {})

        lines = [
            "=" * 65,
            "              标注质量评估报告",
            "=" * 65,
            f"生成时间: {report.get('generated_at', 'N/A')}",
            f"标注目录: {report.get('annotation_dir', 'N/A')}",
            f"文件总数: {report.get('total_files', 0)}",
            "",
            "--- 校验结果 ---",
            f"  通过 (valid):   {val.get('valid', 0):>5}",
            f"  未通过 (invalid): {val.get('invalid', 0):>5}",
            f"  异常 (error):    {val.get('errors', 0):>5}",
            "",
            "--- 质量分布 ---",
            f"  综合平均分:  {qd.get('mean_score', 0):.4f}",
            f"  最低分:      {qd.get('min_score', 0):.4f}",
            f"  最高分:      {qd.get('max_score', 0):.4f}",
            "",
            "  等级分布:",
        ]

        for label, count in sorted(qd.get("by_label", {}).items()):
            bar = "█" * min(count, 40)
            lines.append(f"    {label:<12}: {count:>4}  {bar}")

        lines.append("")
        lines.append("  判定分布:")
        for verdict, count in sorted(qd.get("by_verdict", {}).items()):
            label_map = {"accept": "✅ 接受", "review": "⚠️  复核", "reject": "❌ 拒绝"}
            label = label_map.get(verdict, verdict)
            lines.append(f"    {label:<14}: {count:>4}")

        lines.append("")
        lines.append("--- 各维度平均分 ---")
        lines.append(f"  模糊度:    {sub.get('mean_blur', 0):.4f}")
        lines.append(f"  亮度:      {sub.get('mean_brightness', 0):.4f}")
        lines.append(f"  姿态:      {sub.get('mean_pose', 0):.4f}")
        lines.append(f"  人脸占比:  {sub.get('mean_face_ratio', 0):.4f}")
        lines.append(f"  关键点位置: {sub.get('mean_landmark_position', 0):.4f}")

        lines.append("")
        lines.append(f"发现问题总数: {qd.get('total_issues_found', 0)}")

        # 需复核的文件
        review_files = summary.get("review_files", [])
        if review_files:
            lines.append(f"\n--- ⚠️  建议复核的文件 ({len(review_files)} 个) ---")
            for f in review_files[:10]:
                lines.append(
                    f"  [{f['score']:.2f}] {f['file']}"
                )
                for issue in f.get("issues", []):
                    lines.append(f"         ↳ {issue}")
            if len(review_files) > 10:
                lines.append(f"  ... 及其他 {len(review_files) - 10} 个文件")

        # 建议拒绝的文件
        poor_files = summary.get("poor_files", [])
        if poor_files:
            lines.append(f"\n--- ❌ 建议拒绝的文件 ({len(poor_files)} 个) ---")
            for f in poor_files[:10]:
                lines.append(
                    f"  [{f['score']:.2f}] {f['file']}"
                )
                for issue in f.get("issues", []):
                    lines.append(f"         ↳ {issue}")
            if len(poor_files) > 10:
                lines.append(f"  ... 及其他 {len(poor_files) - 10} 个文件")

        if not review_files and not poor_files:
            lines.append("\n✅ 所有文件质量合格！")

        lines.append("\n" + "=" * 65)
        return "\n".join(lines)


# ============================================================
# 命令行入口
# ============================================================


def main():
    """质量报告生成工具的命令行入口。"""
    parser = argparse.ArgumentParser(
        description="标注质量报告生成工具 — 批量评估标注文件质量并生成统计报告",
    )
    parser.add_argument(
        "annotation_dir",
        nargs="?",
        default="output/annotations/auto_labeled",
        help="标注文件目录（默认: output/annotations/auto_labeled）",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        default=None,
        help="将详细报告保存为 JSON 文件",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="仅输出文本报告（不保存 JSON）",
    )

    args = parser.parse_args()

    reporter = QualityReport()

    if args.output_json and not args.text_only:
        # 生成并保存 JSON
        report = reporter.generate(args.annotation_dir, output_json=args.output_json)
        print(reporter._format_text_report(report))
    else:
        # 仅文本报告
        text = reporter.generate_text_report(args.annotation_dir)
        print(text)

    sys.exit(0)


if __name__ == "__main__":
    main()
