"""
文件名: import_data.py
功能: 多源数据采集导入工具

支持以下导入方式:
  1. URL 下载 — 从网络 URL 批量下载图片
  2. 外部文件夹导入 — 将外部目录中的图片复制/链接到项目 dataset/
  3. 公开数据集快速导入 — 支持常见数据集目录结构（按类别分文件夹）

使用方式:
  python scripts/import_data.py --from-url urls.txt --output dataset/raw_images/imported/
  python scripts/import_data.py --from-dir /path/to/photos/ --mode copy
  python scripts/import_data.py --from-dir /path/to/dataset/ --mode organize  # 按子文件夹分类导入
"""

import argparse
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import SUPPORTED_IMAGE_EXTENSIONS, normalize_path

# 用户代理，避免被某些网站拒绝
USER_AGENT = "FaceAnnotationTool/3.0"


def download_image(url: str, output_dir: Path, timeout: int = 30) -> Optional[Path]:
    """
    从 URL 下载单张图片。

    Args:
        url: 图片 URL
        output_dir: 保存目录
        timeout: 超时时间（秒）

    Returns:
        Optional[Path]: 保存的文件路径，失败返回 None
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()

        if len(data) < 1024:
            print(f"  [SKIP] 文件过小 ({len(data)} bytes): {url}")
            return None

        # 从 URL 推断文件名
        parsed = urlparse(url)
        filename = Path(parsed.path).name or f"download_{hash(url) % 100000:05d}.jpg"

        # 检查扩展名
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            filename += ".jpg"

        output_path = output_dir / filename
        # 避免覆盖
        counter = 1
        while output_path.exists():
            stem = Path(filename).stem
            output_path = output_dir / f"{stem}_{counter}{Path(filename).suffix}"
            counter += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        return output_path

    except urllib.error.URLError as e:
        print(f"  [ERROR] 网络错误: {url} - {e.reason}")
        return None
    except urllib.error.HTTPError as e:
        print(f"  [ERROR] HTTP {e.code}: {url}")
        return None
    except Exception as e:
        print(f"  [ERROR] 下载失败: {url} - {e}")
        return None


def import_from_urls(
    urls_file: Path,
    output_dir: Path,
    max_workers: int = 5,
) -> dict:
    """
    从包含 URL 列表的文件批量下载图片。

    文件格式: 每行一个 URL，支持 # 注释行。

    Args:
        urls_file: URL 列表文件路径
        output_dir: 输出目录
        max_workers: 并发下载数

    Returns:
        dict: {"total": int, "success": int, "failed": int, "files": List[Path]}
    """
    if not urls_file.exists():
        print(f"[ERROR] URL 文件不存在: {urls_file}")
        return {"total": 0, "success": 0, "failed": 0, "files": []}

    urls = []
    with open(urls_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    if not urls:
        print("[WARN] URL 文件为空")
        return {"total": 0, "success": 0, "failed": 0, "files": []}

    print(f"下载 {len(urls)} 张图片到 {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"total": len(urls), "success": 0, "failed": 0, "files": []}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image, url, output_dir): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                path = future.result()
                if path:
                    results["success"] += 1
                    results["files"].append(path)
                    print(f"  [{results['success']}/{len(urls)}] {path.name}")
                else:
                    results["failed"] += 1
            except Exception as e:
                results["failed"] += 1
                print(f"  [ERROR] {url}: {e}")

    return results


def import_from_directory(
    source_dir: Path,
    output_dir: Path,
    mode: str = "copy",
    organize_by_folder: bool = True,
) -> dict:
    """
    从外部文件夹导入图片到项目数据集目录。

    Args:
        source_dir: 源目录路径
        output_dir: 目标目录（项目 dataset/raw_images/ 下的子目录）
        mode: "copy"（复制）或 "symlink"（符号链接）
        organize_by_folder: True 时保留源目录的子文件夹结构（按人物分类）

    Returns:
        dict: {"total": int, "copied": int, "skipped": int, "files": List[Path]}
    """
    if not source_dir.exists():
        print(f"[ERROR] 源目录不存在: {source_dir}")
        return {"total": 0, "copied": 0, "skipped": 0, "files": []}

    # 收集所有支持的图片
    image_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(source_dir.rglob(f"*{ext}"))
        image_files.extend(source_dir.rglob(f"*{ext.upper()}"))

    image_files = sorted(set(image_files))

    if not image_files:
        print(f"[WARN] 未在 {source_dir} 下找到支持的图片文件")
        return {"total": 0, "copied": 0, "skipped": 0, "files": []}

    print(f"发现 {len(image_files)} 张图片，模式: {mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"total": len(image_files), "copied": 0, "skipped": 0, "files": []}

    for img_path in image_files:
        if organize_by_folder:
            # 保留子文件夹结构（如按人物名分文件夹）
            try:
                rel = img_path.relative_to(source_dir)
            except ValueError:
                rel = img_path.name
            dest = output_dir / rel
        else:
            # 全部放入同一目录
            dest = output_dir / img_path.name

        # 避免覆盖
        if dest.exists():
            results["skipped"] += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)

        if mode == "symlink":
            try:
                dest.symlink_to(img_path.resolve())
            except OSError:
                # 符号链接不支持（如 Windows 权限不足），回退到复制
                shutil.copy2(img_path, dest)
        else:
            shutil.copy2(img_path, dest)

        results["copied"] += 1
        results["files"].append(dest)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="多源数据采集导入工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 URL 列表下载
  python scripts/import_data.py --from-url urls.txt --output dataset/raw_images/web/

  # 从外部文件夹复制图片
  python scripts/import_data.py --from-dir D:/photos/faces/ --output dataset/raw_images/

  # 从外部文件夹按子文件夹结构导入（每个子文件夹=一个人物）
  python scripts/import_data.py --from-dir D:/dataset/ --output dataset/raw_images/ --organize
        """,
    )
    parser.add_argument(
        "--from-url",
        type=Path,
        help="包含图片 URL 列表的文本文件（每行一个 URL）",
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        help="外部图片文件夹路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "raw_images",
        help="输出目录（默认: dataset/raw_images/）",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="导入模式: copy（复制文件）或 symlink（符号链接，节省空间）",
    )
    parser.add_argument(
        "--organize",
        action="store_true",
        help="按源目录的子文件夹结构组织（用于按人物分类的数据集）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="URL 下载并发数（默认: 5）",
    )

    args = parser.parse_args()

    if not args.from_url and not args.from_dir:
        parser.print_help()
        print("\n[ERROR] 请指定 --from-url 或 --from-dir")
        sys.exit(1)

    start = datetime.now()

    if args.from_url:
        result = import_from_urls(args.from_url, args.output, max_workers=args.workers)

    elif args.from_dir:
        result = import_from_directory(
            args.from_dir,
            args.output,
            mode=args.mode,
            organize_by_folder=args.organize,
        )

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*50}")
    print(f"导入完成 | 总计: {result['total']} | 成功: {result.get('copied', result.get('success', 0))} | 跳过/失败: {result.get('skipped', result.get('failed', 0))} | 耗时: {elapsed:.1f}s")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
