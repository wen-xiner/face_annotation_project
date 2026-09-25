"""
文件名: batch_processor.py
功能: 批量标注处理器，遍历原始图片目录，对所有图片执行自动标注
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .annotation_generator import AnnotationGenerator
from .utils import SUPPORTED_IMAGE_EXTENSIONS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 尝试导入 tqdm，若不可用则回退到简单打印
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class BatchProcessor:
    """
    批量标注处理器，遍历原始图片目录，对所有图片执行自动标注。

    Attributes:
        raw_dir: 原始图片根目录
        output_dir: 标注 JSON 输出根目录
        generator: AnnotationGenerator 实例
    """

    def __init__(
        self,
        raw_dir: Union[str, Path] = "dataset/raw_images",
        output_dir: Union[str, Path] = "output/annotations/auto_labeled",
    ):
        """
        初始化批量处理器。

        Args:
            raw_dir: 原始图片存放的根目录
            output_dir: 标注 JSON 文件的输出根目录
        """
        self.raw_dir = Path(raw_dir).resolve()
        self.output_dir = Path(output_dir).resolve()

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"原始图片目录不存在: {self.raw_dir}")

        self.generator = AnnotationGenerator(
            output_dir=self.output_dir,
            raw_dir=self.raw_dir,
        )

    def discover_images(self) -> List[Path]:
        """
        遍历 raw_dir 下所有子文件夹，收集所有支持的图片文件。

        Returns:
            List[Path]: 所有图片文件的绝对路径列表（按路径排序）
        """
        images = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            # 同时匹配小写和大写扩展名
            images.extend(self.raw_dir.rglob(f"*{ext}"))
            images.extend(self.raw_dir.rglob(f"*{ext.upper()}"))

        # 去重并排序
        unique_images = sorted(set(images))
        logger.info(f"发现 {len(unique_images)} 张图片")
        return unique_images

    def _process_image_list(
        self,
        images: List[Path],
        skip_existing: bool = True,
        progress_desc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理给定的图片列表，对所有图片执行自动标注。

        该方法为 process_all() 和 process_person() 的公共实现，
        消除了两处约 60% 的重复代码。

        Args:
            images: 图片路径列表
            skip_existing: 是否跳过已有标注文件的图片
            progress_desc: 进度条描述文字（tqdm 时使用）

        Returns:
            Dict[str, Any]: 处理结果摘要
        """
        if not images:
            logger.warning("未找到任何图片文件")
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "skipped": 0,
                "details": [],
                "elapsed_seconds": 0.0,
            }

        start_time = time.time()

        # 进度条或简单计数
        if HAS_TQDM:
            desc = progress_desc or "标注进度"
            iterator = tqdm(images, desc=desc, unit="张")
        else:
            iterator = images

        results = {
            "total": len(images),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "details": [],
            "elapsed_seconds": 0.0,
        }

        for img_path in iterator:
            detail = {
                "image": str(img_path),
                "status": "failed",
                "annotation_path": None,
                "error": None,
            }

            # 检查是否跳过已标注的图片
            if skip_existing and self.generator.annotation_exists(img_path):
                detail["status"] = "skipped"
                detail["annotation_path"] = str(
                    self.generator._get_output_path(img_path)
                )
                results["skipped"] += 1
                results["details"].append(detail)
                continue

            # 执行标注
            try:
                annotation = self.generator.annotate(img_path, save=True)
                if annotation is not None:
                    detail["status"] = "success"
                    detail["annotation_path"] = str(
                        self.generator._get_output_path(img_path)
                    )
                    results["success"] += 1
                else:
                    detail["error"] = "未检测到人脸或关键点提取失败"
                    results["failed"] += 1
            except Exception as e:
                detail["error"] = str(e)
                results["failed"] += 1
                logger.error(f"处理失败 {img_path.name}: {e}")

            results["details"].append(detail)

        results["elapsed_seconds"] = round(time.time() - start_time, 1)
        return results

    def process_all(self, skip_existing: bool = True) -> Dict[str, Any]:
        """
        批量处理 raw_dir 下的所有图片。

        Args:
            skip_existing: 是否跳过已有标注文件的图片

        Returns:
            Dict[str, Any]: 处理结果摘要
            {
                "total": int,
                "success": int,
                "failed": int,
                "skipped": int,
                "details": List[Dict],
                "elapsed_seconds": float
            }
        """
        images = self.discover_images()
        return self._process_image_list(
            images,
            skip_existing=skip_existing,
            progress_desc="标注进度",
        )

    def process_person(
        self,
        person_id: str,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        处理指定人物文件夹下的所有图片。

        Args:
            person_id: 人物 ID（文件夹名）
            skip_existing: 是否跳过已有标注文件

        Returns:
            Dict[str, Any]: 处理结果摘要
        """
        person_dir = self.raw_dir / person_id
        if not person_dir.exists():
            raise FileNotFoundError(f"人物文件夹不存在: {person_dir}")

        images = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            images.extend(person_dir.glob(f"*{ext}"))
            images.extend(person_dir.glob(f"*{ext.upper()}"))

        images = sorted(set(images))
        logger.info(f"人物 {person_id}: 发现 {len(images)} 张图片")

        return self._process_image_list(
            images,
            skip_existing=skip_existing,
            progress_desc=f"标注 {person_id}",
        )

    @staticmethod
    def generate_summary_report(results: Dict[str, Any]) -> str:
        """
        生成人类可读的处理摘要报告。

        Args:
            results: process_all() 或 process_person() 的返回值

        Returns:
            str: 格式化的摘要报告
        """
        total = results["total"]
        success = results["success"]
        failed = results["failed"]
        skipped = results["skipped"]
        elapsed = results["elapsed_seconds"]

        lines = [
            "=" * 50,
            "          批量标注处理报告",
            "=" * 50,
            f"总图片数:      {total:>6}",
            f"成功标注:      {success:>6}",
            f"标注失败:      {failed:>6}",
            f"跳过（已标注）: {skipped:>6}",
            f"耗时:          {elapsed:>6.1f} 秒",
            "=" * 50,
        ]

        # 列出失败的图片
        failed_items = [
            d for d in results["details"] if d["status"] == "failed"
        ]
        if failed_items:
            lines.append("\n失败图片详情:")
            lines.append("-" * 40)
            for item in failed_items:
                img_name = Path(item["image"]).name
                lines.append(f"  - {img_name}: {item['error']}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def close(self):
        """释放内部资源。"""
        self.generator.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
# 命令行入口
# ============================================================


def main():
    """批量人脸标注工具的命令行入口。"""
    parser = argparse.ArgumentParser(
        description="批量人脸标注工具 — 对 dataset/raw_images/ 下的所有图片自动标注",
    )
    parser.add_argument(
        "--raw-dir",
        default="dataset/raw_images",
        help="原始图片目录（默认: dataset/raw_images）",
    )
    parser.add_argument(
        "--output-dir",
        default="output/annotations/auto_labeled",
        help="标注输出目录（默认: output/annotations/auto_labeled）",
    )
    parser.add_argument(
        "--person",
        help="仅处理指定人物 ID 的文件夹",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="不跳过已有标注（强制重新标注所有图片）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，仅输出最终报告",
    )

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        processor = BatchProcessor(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
        )

        if args.person:
            results = processor.process_person(
                args.person,
                skip_existing=not args.no_skip,
            )
        else:
            results = processor.process_all(
                skip_existing=not args.no_skip,
            )

        # 始终打印摘要报告
        print(processor.generate_summary_report(results))

        processor.close()

        # 如果有失败的，返回非零退出码
        if results["failed"] > 0:
            sys.exit(1)

    except Exception as e:
        logger.error(f"批量处理异常: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
