"""
文件名: dataset_organizer.py
功能: 数据集整理工具 — 基于人脸特征签名智能分组，
      将相同人物的图片归到同一文件夹，并更新标注路径

模式:
  --smart   使用 FaceMatcher 提取人脸特征，自动识别同人图片并合并分组（推荐）
  --by-name 仅按文件名规则推断人物 ID（无法判断是否同人）
"""

import argparse
import io
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .face_signature import FaceMatcher, extract_face_signature
from .utils import (
    SUPPORTED_IMAGE_EXTENSIONS,
    get_project_root,
    load_annotation_json,
    load_image,
    save_annotation_json,
)

logger = logging.getLogger(__name__)


# ============================================================
# 图片发现
# ============================================================


def discover_unorganized_images(raw_dir: Path) -> List[Path]:
    """查找 raw_dir 下不在子文件夹中的图片。"""
    images = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        images.extend(raw_dir.glob(f"*{ext}"))
        images.extend(raw_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def find_web_uploads(web_upload_dir: Path) -> List[Path]:
    """查找 web_app/uploads 下的所有图片。"""
    if not web_upload_dir.exists():
        return []
    images = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        images.extend(web_upload_dir.glob(f"*{ext}"))
        images.extend(web_upload_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def suggest_person_id(image_path: Path) -> str:
    """按文件名推断人物 ID（后备方案）。"""
    stem = image_path.stem
    if stem.isdigit():
        return f"person_{int(stem):03d}"
    if len(stem) >= 4 and all(c in "0123456789abcdef" for c in stem[:4].lower()):
        return f"person_{stem[:4]}"
    safe_name = "".join(c for c in stem[:6] if c.isalnum())
    return f"person_{safe_name}" if safe_name else "person_unknown"


# ============================================================
# 基于人脸签名的智能分组
# ============================================================


def group_by_face_signature(
    image_paths: List[Path],
    auto_annotation_dir: Optional[Path] = None,
) -> Dict[str, List[Path]]:
    """
    使用人脸特征签名将图片按人物分组。

    流程：
    1. 加载每张图片对应的标注 JSON，获取 bbox + landmarks
    2. 提取人脸特征签名（几何 + 感知哈希）
    3. 用 FaceMatcher 做聚类分组

    Args:
        image_paths: 图片路径列表
        auto_annotation_dir: 自动标注 JSON 所在目录（用于查找已有标注）

    Returns:
        Dict[str, List[Path]]: {person_id: [图片路径列表]}
    """
    project_root = get_project_root()
    if auto_annotation_dir is None:
        auto_annotation_dir = project_root / "output" / "annotations" / "auto_labeled"

    matcher = FaceMatcher()
    signatures: Dict[str, Dict] = {}

    # 辅助：查找图片对应的标注文件
    def find_annotation(img_path: Path) -> Optional[Path]:
        candidates = [
            auto_annotation_dir / f"{img_path.stem}_annotation.json",
            project_root / "output" / "annotations" / "auto_labeled" / f"{img_path.stem}_annotation.json",
        ]
        # 也搜索子目录
        if auto_annotation_dir.exists():
            for cand in auto_annotation_dir.rglob(f"{img_path.stem}_annotation.json"):
                candidates.append(cand)
        for c in candidates:
            if c.exists():
                return c
        return None

    annotatable = 0
    for img_path in image_paths:
        ann_path = find_annotation(img_path)
        if ann_path is None:
            # 没有标注 → 无法提取人脸签名 → 回退到文件名猜测
            continue

        try:
            ann = load_annotation_json(ann_path)
            bbox = ann.get("bbox")
            landmarks = ann.get("landmarks")

            if bbox is None or landmarks is None:
                continue

            image = load_image(img_path)
            sig = extract_face_signature(image, bbox=bbox, landmarks=landmarks)
            sig["_image_path"] = str(img_path)
            sig["_annotation_path"] = str(ann_path)
            signatures[str(img_path)] = sig
            annotatable += 1
        except Exception:
            continue

    if annotatable < 2:
        return {}

    # 使用 FaceMatcher 聚类
    groups = matcher.find_duplicate_groups(signatures)

    # 分配人物 ID
    result: Dict[str, List[Path]] = {}
    for i, group in enumerate(groups):
        person_id = f"person_{i + 1:03d}"
        result[person_id] = [Path(img_id) for img_id in group]

    return result


# ============================================================
# 合并已有分组与智能分组
# ============================================================


def merge_with_existing_structure(
    smart_groups: Dict[str, List[Path]],
    raw_dir: Path,
) -> Dict[str, List[Path]]:
    """
    将智能分组结果与 raw_dir 下已有的子文件夹结构合并。

    规则：
    - 如果图片已在 person_xxx 文件夹中，保留原分组名
    - 智能分组中的同人图片合并到已有文件夹
    - 全新的图片分配到新 ID
    """
    # 收集已有分组
    existing: Dict[str, List[str]] = {}
    for subdir in sorted(raw_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if not subdir.name.startswith("person_"):
            continue
        for img in sorted(subdir.iterdir()):
            if img.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                existing.setdefault(subdir.name, []).append(img.name)

    # 名字映射：图片名 → 已有 person_id
    name_to_existing: Dict[str, str] = {}
    for pid, names in existing.items():
        for name in names:
            if name not in name_to_existing:
                name_to_existing[name] = pid

    # 合并
    merged: Dict[str, List[Path]] = {}
    new_person_counter = len(existing) + 1

    for person_id, paths in smart_groups.items():
        # 检查这些图片是否属于已有分组
        assigned_pid = None
        for p in paths:
            if p.name in name_to_existing:
                assigned_pid = name_to_existing[p.name]
                break

        if assigned_pid is None:
            assigned_pid = f"person_{new_person_counter:03d}"
            new_person_counter += 1

        merged.setdefault(assigned_pid, []).extend(paths)

    return merged


# ============================================================
# 主整理流程
# ============================================================


def organize_images(
    raw_dir: Path,
    web_upload_dir: Optional[Path] = None,
    sync_web_uploads: bool = True,
    use_smart_grouping: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    整理数据集。

    Args:
        raw_dir: 原始图片根目录
        web_upload_dir: Web 上传目录
        sync_web_uploads: 是否同步 Web 上传图片
        use_smart_grouping: True=人脸特征智能分组, False=文件名猜测
        dry_run: 仅预览
    """
    results: Dict[str, Any] = {
        "organized": [],
        "synced": [],
        "skipped": [],
        "duplicates_found": [],
        "errors": [],
    }

    project_root = get_project_root()

    # === 收集所有待处理图片 ===
    all_collected: List[Path] = list(discover_unorganized_images(raw_dir))

    if sync_web_uploads and web_upload_dir and web_upload_dir.exists():
        for img in find_web_uploads(web_upload_dir):
            target = raw_dir / suggest_person_id(img) / img.name
            if not target.exists():
                all_collected.append(img)

    if not all_collected:
        return results

    # === 智能分组 vs 文件名分组 ===
    if use_smart_grouping:
        print("\n[*] 正在提取人脸特征签名进行智能分组...")
        smart_groups = group_by_face_signature(all_collected)

        if smart_groups:
            print(f"[*] 人脸聚类完成: {len(smart_groups)} 个不同人物")
            # 检测重复图片
            matcher = FaceMatcher()
            sigs = {}
            for img in all_collected:
                ann_path = None
                for cand in (project_root / "output" / "annotations" / "auto_labeled").rglob(f"{img.stem}_annotation.json"):
                    ann_path = cand
                    break
                if ann_path:
                    try:
                        ann = load_annotation_json(ann_path)
                        if ann.get("bbox") and ann.get("landmarks"):
                            image = load_image(img)
                            sigs[str(img)] = extract_face_signature(
                                image, bbox=ann["bbox"], landmarks=ann["landmarks"]
                            )
                    except Exception:
                        pass

            exact_dupes = matcher.find_exact_duplicates(sigs) if sigs else []
            if exact_dupes:
                print(f"[!] 发现 {len(exact_dupes)} 组完全重复的图片:")
                for group in exact_dupes:
                    names = [Path(p).name for p in group]
                    results["duplicates_found"].append(names)
                    print(f"    {'  =  '.join(names)}")

            # 合并已有结构
            person_groups = merge_with_existing_structure(smart_groups, raw_dir)
        else:
            print("[*] 无法提取足够的人脸签名，回退到文件名分组")
            use_smart_grouping = False

    if not use_smart_grouping:
        person_groups = {}
        for img in all_collected:
            pid = suggest_person_id(img)
            person_groups.setdefault(pid, []).append(img)

    # === 执行文件移动/复制 ===
    skipped_items = 0
    for person_id, images in person_groups.items():
        target_dir = raw_dir / person_id
        for img in images:
            target_path = target_dir / img.name

            # 已在目标位置
            if target_path.resolve() == img.resolve():
                skipped_items += 1
                continue

            # 已在其他 person 文件夹中（不用移动）
            if img.parent != raw_dir and img.parent.name.startswith("person_"):
                continue

            is_web_upload = "web_app" in str(img) and "uploads" in str(img)

            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
                if is_web_upload or img.parent != raw_dir:
                    shutil.copy2(str(img), str(target_path))
                else:
                    shutil.move(str(img), str(target_path))
                print(f"  [OK] {img.name} → {person_id}/{img.name}")
            else:
                print(f"  [*] [预览] {img.name} → {person_id}/{img.name}")

            item = {
                "image": str(img),
                "person_id": person_id,
                "new_path": str(target_path),
            }
            if is_web_upload:
                results["synced"].append(item)
            else:
                results["organized"].append(item)

    # === 更新标注 JSON ===
    if not dry_run:
        update_annotations(results, raw_dir)

    return results


def update_annotations(
    organize_results: Dict[str, Any],
    raw_dir: Path,
) -> None:
    """更新标注 JSON 中的 image_path 字段。"""
    project_root = get_project_root()
    auto_dir = project_root / "output" / "annotations" / "auto_labeled"
    reviewed_dir = project_root / "output" / "annotations" / "manually_reviewed"

    path_map: Dict[str, str] = {}
    for item in organize_results.get("organized", []) + organize_results.get("synced", []):
        old_path = Path(item["image"]).resolve()
        new_path = Path(item["new_path"]).resolve()
        try:
            new_rel = new_path.relative_to(project_root).as_posix()
        except ValueError:
            new_rel = f"dataset/raw_images/{item['person_id']}/{new_path.name}"

        path_map[old_path.as_posix()] = new_rel
        path_map[f"web_app/uploads/{old_path.name}"] = new_rel
        path_map[f"dataset/raw_images/{old_path.name}"] = new_rel

    updated_count = 0
    for ann_dir in (auto_dir, reviewed_dir):
        if not ann_dir.exists():
            continue
        for json_file in sorted(ann_dir.rglob("*_annotation.json")):
            try:
                ann = load_annotation_json(json_file)
                old_image_path = ann.get("image_path", "")
                new_image_path = path_map.get(old_image_path)
                if new_image_path is None:
                    for old_key, new_val in path_map.items():
                        if Path(old_key).name == Path(old_image_path).name:
                            new_image_path = new_val
                            break
                if new_image_path and new_image_path != old_image_path:
                    ann["image_path"] = new_image_path
                    save_annotation_json(ann, json_file)
                    print(f"  [log] 更新标注: {json_file.name} → {new_image_path}")
                    updated_count += 1
            except Exception as e:
                print(f"  [WARN] 更新失败 {json_file.name}: {e}")

    if updated_count > 0:
        print(f"\n[log] 共更新 {updated_count} 个标注文件的路径引用")


def generate_organize_report(results: Dict[str, Any]) -> str:
    """生成整理报告。"""
    organized = len(results["organized"])
    synced = len(results["synced"])
    dupes = len(results["duplicates_found"])
    skipped = len(results["skipped"])
    errors = len(results["errors"])

    lines = [
        "=" * 50,
        "       数据集整理报告",
        "=" * 50,
        f"散落图片已归位:  {organized:>5}",
        f"Web图片已同步:   {synced:>5}",
        f"发现重复组:      {dupes:>5}",
        f"已跳过:          {skipped:>5}",
        f"错误:            {errors:>5}",
    ]

    if dupes > 0:
        lines.append(f"\n[!] 重复图片组 ({dupes} 组):")
        for group in results["duplicates_found"]:
            lines.append(f"    {'  =  '.join(group)}")
            lines.append("    → 这些图片已被合并到同一 person 文件夹")

    if organized > 0:
        lines.append(f"\n[+] 归位图片 ({organized} 张):")
        for item in results["organized"]:
            lines.append(f"  {Path(item['image']).name} → {item['person_id']}/")

    if synced > 0:
        lines.append(f"\n[web] 同步图片 ({synced} 张):")
        for item in results["synced"]:
            lines.append(f"  {Path(item['image']).name} → {item['person_id']}/")

    lines.append("=" * 50)
    return "\n".join(lines)


# ============================================================
# 命令行入口
# ============================================================


def main():
    """数据集整理工具的命令行入口。"""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="数据集整理工具 — 智能分组 + 去重，整理原始图片目录",
    )
    parser.add_argument(
        "--raw-dir", default="dataset/raw_images",
        help="原始图片目录（默认: dataset/raw_images）",
    )
    parser.add_argument(
        "--web-upload-dir", default="web_app/uploads",
        help="Web 上传目录（默认: web_app/uploads）",
    )
    parser.add_argument(
        "--no-sync", action="store_true",
        help="不同步 Web 上传的图片",
    )
    parser.add_argument(
        "--by-name", action="store_true",
        help="仅按文件名分组（关闭人脸智能分组）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览，不实际执行",
    )

    args = parser.parse_args()

    project_root = get_project_root()
    raw_dir = project_root / args.raw_dir
    web_upload_dir = project_root / args.web_upload_dir if not args.no_sync else None

    if not raw_dir.exists():
        print(f"[ERROR] 原始图片目录不存在: {raw_dir}")
        sys.exit(1)

    if args.dry_run:
        print("[预览模式] 不会实际修改任何文件\n")

    use_smart = not args.by_name

    results = organize_images(
        raw_dir=raw_dir,
        web_upload_dir=web_upload_dir,
        sync_web_uploads=not args.no_sync,
        use_smart_grouping=use_smart,
        dry_run=args.dry_run,
    )

    print(generate_organize_report(results))

    if args.dry_run:
        print("\n[tip] 去掉 --dry-run 参数以实际执行整理操作。")


if __name__ == "__main__":
    main()
