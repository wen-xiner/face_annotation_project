"""
文件名: format_converter.py
功能: 标注格式转换器，支持将项目标注 JSON 导出为 COCO / YOLO / Pascal VOC 格式

支持的转换方向：
  - 本项目标注 JSON → COCO JSON（含 bbox + keypoints）
  - 本项目标注 JSON → YOLO txt（归一化 bbox，可选关键点文件）
  - 本项目标注 JSON → Pascal VOC XML

使用示例：
  # Python API
  converter = FormatConverter()
  converter.to_coco(annotation_paths, "output/coco.json")
  converter.to_yolo(annotation_paths, "output/yolo/")
  converter.to_voc(annotation_paths, "output/voc/")

  # CLI
  python -m src.format_converter output/annotations/auto_labeled/ --format coco --output output/coco.json
"""

import argparse
import csv
import json
import logging
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2

from .utils import get_project_root, load_annotation_json, save_annotation_json

logger = logging.getLogger(__name__)


class FormatConverter:
    """
    标注格式转换器。

    将本项目的标注 JSON 格式转换为业界通用格式。

    Attributes:
        image_root_dir: 图片根目录，用于解析标注中的相对路径
    """

    def __init__(
        self,
        image_root_dir: Union[str, Path] = ".",
    ):
        """
        初始化格式转换器。

        Args:
            image_root_dir: 图片根目录，标注中 image_path 的参照基准。
                            默认为项目根目录。
        """
        self.image_root_dir = Path(image_root_dir)
        self._image_id_counter = 1
        self._annotation_id_counter = 1

    # ================================================================
    # COCO JSON 格式
    # ================================================================

    def to_coco(
        self,
        annotation_paths: List[Path],
        output_path: Union[str, Path],
        category_name: str = "face",
    ) -> Dict[str, Any]:
        """
        将一批标注转换为单个 COCO JSON 文件。

        COCO 格式包含：
          - images: 图片信息（id, file_name, width, height）
          - annotations: 标注信息（bbox, keypoints, area 等）
          - categories: 类别信息

        Args:
            annotation_paths: 标注 JSON 文件路径列表
            output_path: COCO JSON 输出路径
            category_name: 类别名称，默认 "face"

        Returns:
            Dict: COCO 格式的完整数据字典
        """
        coco = {
            "info": {
                "description": "人脸标注数据集",
                "version": "1.0",
                "year": 2026,
                "contributor": "Face Annotation Tool",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": category_name,
                    "supercategory": "face",
                    "keypoints": [
                        "left_eye",
                        "right_eye",
                        "nose_tip",
                        "left_mouth_corner",
                        "right_mouth_corner",
                    ],
                    "skeleton": [],  # 可选：关键点骨架连接
                }
            ],
        }

        image_id_map = {}  # 图片路径 → COCO image_id

        for json_path in annotation_paths:
            try:
                annotation = load_annotation_json(json_path)
            except Exception as e:
                logger.warning(f"跳过无法读取的文件 {json_path}: {e}")
                continue

            image_path = annotation.get("image_path", "")
            if not image_path:
                logger.warning(f"标注文件缺少 image_path: {json_path}")
                continue

            # 解析图片路径
            full_image_path = self._resolve_image_path(image_path)

            # 获取图片尺寸
            img_w, img_h = self._get_image_size(full_image_path)
            if img_w is None:
                logger.warning(f"无法读取图片尺寸: {image_path}")
                continue

            # 注册图片（去重）
            image_key = str(image_path)
            if image_key not in image_id_map:
                coco_image = {
                    "id": self._image_id_counter,
                    "file_name": str(image_path),
                    "width": img_w,
                    "height": img_h,
                }
                coco["images"].append(coco_image)
                image_id_map[image_key] = self._image_id_counter
                self._image_id_counter += 1

            image_id = image_id_map[image_key]

            # 转换标注
            coco_annotation = self._annotation_to_coco(
                annotation, image_id, img_w, img_h
            )
            if coco_annotation is not None:
                coco["annotations"].append(coco_annotation)

        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)

        logger.info(
            f"COCO 格式导出完成: {len(coco['images'])} 张图片, "
            f"{len(coco['annotations'])} 个标注 → {output_path}"
        )

        return coco

    def _annotation_to_coco(
        self,
        annotation: Dict[str, Any],
        image_id: int,
        img_w: int,
        img_h: int,
    ) -> Optional[Dict[str, Any]]:
        """
        将单条标注转换为 COCO annotation 对象。

        COCO bbox 格式: [x, y, width, height]（整数像素）
        COCO keypoints 格式: [x1, y1, v1, x2, y2, v2, ...]
          v=0: 未标注, v=1: 标注但被遮挡, v=2: 标注且可见
        """
        bbox = annotation.get("bbox", {})
        landmarks = annotation.get("landmarks", {})

        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("width", 0)
        h = bbox.get("height", 0)

        if w <= 0 or h <= 0:
            return None

        area = w * h

        # 构建 keypoints 数组（COCO 顺序与 landmarks 定义顺序一致）
        keypoint_names = [
            "left_eye",
            "right_eye",
            "nose_tip",
            "left_mouth_corner",
            "right_mouth_corner",
        ]
        keypoints = []
        num_keypoints = 0

        for kp_name in keypoint_names:
            coords = landmarks.get(kp_name, [-1, -1])
            kx, ky = coords[0], coords[1]

            if kx >= 0 and ky >= 0:
                keypoints.extend([int(kx), int(ky), 2])  # v=2 可见
                num_keypoints += 1
            else:
                keypoints.extend([0, 0, 0])  # v=0 未标注

        coco_ann = {
            "id": self._annotation_id_counter,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": float(area),
            "iscrowd": 0,
            "keypoints": keypoints,
            "num_keypoints": num_keypoints,
            "score": annotation.get("confidence", 1.0),
        }

        self._annotation_id_counter += 1
        return coco_ann

    # ================================================================
    # YOLO txt 格式
    # ================================================================

    def to_yolo(
        self,
        annotation_paths: List[Path],
        output_dir: Union[str, Path],
        with_keypoints: bool = False,
    ) -> List[Path]:
        """
        将标注转换为 YOLO txt 格式（每张图片一个 txt 文件）。

        YOLO bbox 格式: class_id cx cy w h（全部归一化到 [0, 1]）
        class_id 固定为 0（face）。

        YOLO 关键点格式（--with-keypoints）:
          class_id x1 y1 x2 y2 ... x5 y5（归一化坐标）

        Args:
            annotation_paths: 标注 JSON 文件路径列表
            output_dir: 输出目录
            with_keypoints: 是否额外导出关键点文件（.kpt.txt）

        Returns:
            List[Path]: 生成的 txt 文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        for json_path in annotation_paths:
            try:
                annotation = load_annotation_json(json_path)
            except Exception as e:
                logger.warning(f"跳过无法读取的文件 {json_path}: {e}")
                continue

            image_path = annotation.get("image_path", "")
            if not image_path:
                continue

            # 获取图片尺寸（归一化需要）
            full_image_path = self._resolve_image_path(image_path)
            img_w, img_h = self._get_image_size(full_image_path)
            if img_w is None or img_h is None:
                logger.warning(f"无法获取图片尺寸: {image_path}")
                continue

            # 生成 YOLO bbox 标注
            stem = Path(image_path).stem
            bbox_file = output_dir / f"{stem}.txt"
            bbox_lines = self._annotation_to_yolo(annotation, img_w, img_h)
            if bbox_lines:
                with open(bbox_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(bbox_lines) + "\n")
                generated_files.append(bbox_file)

            # 可选：生成关键点文件
            if with_keypoints:
                kpt_file = output_dir / f"{stem}.kpt.txt"
                kpt_lines = self._landmarks_to_yolo_keypoints(
                    annotation, img_w, img_h
                )
                if kpt_lines:
                    with open(kpt_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(kpt_lines) + "\n")
                    generated_files.append(kpt_file)

        logger.info(
            f"YOLO 格式导出完成: {len(generated_files)} 个文件 → {output_dir}"
        )
        return generated_files

    def _annotation_to_yolo(
        self,
        annotation: Dict[str, Any],
        img_w: int,
        img_h: int,
    ) -> List[str]:
        """将单条标注转换为 YOLO 格式的文本行。"""
        bbox = annotation.get("bbox", {})
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("width", 0)
        h = bbox.get("height", 0)

        if w <= 0 or h <= 0:
            return []

        # 转换为 YOLO 归一化格式：cx, cy 是 bbox 中心点
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h

        # 裁剪到 [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        return [f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]

    def _landmarks_to_yolo_keypoints(
        self,
        annotation: Dict[str, Any],
        img_w: int,
        img_h: int,
    ) -> List[str]:
        """将关键点转换为 YOLO 关键点格式（归一化坐标）。"""
        landmarks = annotation.get("landmarks", {})
        keypoint_names = [
            "left_eye",
            "right_eye",
            "nose_tip",
            "left_mouth_corner",
            "right_mouth_corner",
        ]

        coords = []
        for name in keypoint_names:
            pt = landmarks.get(name, [-1, -1])
            if pt[0] >= 0 and pt[1] >= 0:
                coords.append(f"{pt[0] / img_w:.6f}")
                coords.append(f"{pt[1] / img_h:.6f}")
            else:
                coords.append("0.000000")
                coords.append("0.000000")

        if coords:
            return ["0 " + " ".join(coords)]
        return []

    # ================================================================
    # Pascal VOC XML 格式
    # ================================================================

    def to_voc(
        self,
        annotation_paths: List[Path],
        output_dir: Union[str, Path],
    ) -> List[Path]:
        """
        将标注转换为 Pascal VOC XML 格式（每张图片一个 XML 文件）。

        Args:
            annotation_paths: 标注 JSON 文件路径列表
            output_dir: 输出目录

        Returns:
            List[Path]: 生成的 XML 文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        for json_path in annotation_paths:
            try:
                annotation = load_annotation_json(json_path)
            except Exception as e:
                logger.warning(f"跳过无法读取的文件 {json_path}: {e}")
                continue

            image_path = annotation.get("image_path", "")
            if not image_path:
                continue

            # 获取图片尺寸
            full_image_path = self._resolve_image_path(image_path)
            img_w, img_h = self._get_image_size(full_image_path)
            if img_w is None or img_h is None:
                img_w, img_h = 0, 0

            # 生成 VOC XML
            stem = Path(image_path).stem
            xml_file = output_dir / f"{stem}.xml"
            xml_content = self._annotation_to_voc(annotation, img_w, img_h)
            with open(xml_file, "w", encoding="utf-8") as f:
                f.write(xml_content)
            generated_files.append(xml_file)

        logger.info(
            f"Pascal VOC 格式导出完成: {len(generated_files)} 个文件 → {output_dir}"
        )
        return generated_files

    def _annotation_to_voc(
        self,
        annotation: Dict[str, Any],
        img_w: int,
        img_h: int,
    ) -> str:
        """
        将单条标注转换为 Pascal VOC XML 字符串。

        VOC XML 结构:
          <annotation>
            <folder>...</folder>
            <filename>...</filename>
            <size><width> <height> <depth>
            <object>
              <name>face</name>
              <bndbox><xmin> <ymin> <xmax> <ymax></bndbox>
            </object>
          </annotation>
        """
        image_path = annotation.get("image_path", "")
        filename = Path(image_path).name
        folder = Path(image_path).parent.name if image_path else ""

        bbox = annotation.get("bbox", {})
        xmin = int(bbox.get("x", 0))
        ymin = int(bbox.get("y", 0))
        xmax = xmin + int(bbox.get("width", 0))
        ymax = ymin + int(bbox.get("height", 0))

        # 构建 XML
        annotation_el = ET.Element("annotation")

        ET.SubElement(annotation_el, "folder").text = folder
        ET.SubElement(annotation_el, "filename").text = filename
        ET.SubElement(annotation_el, "path").text = str(image_path)

        source = ET.SubElement(annotation_el, "source")
        ET.SubElement(source, "database").text = "Face Annotation Tool"

        size = ET.SubElement(annotation_el, "size")
        ET.SubElement(size, "width").text = str(img_w)
        ET.SubElement(size, "height").text = str(img_h)
        ET.SubElement(size, "depth").text = "3"

        ET.SubElement(annotation_el, "segmented").text = "0"

        obj = ET.SubElement(annotation_el, "object")
        ET.SubElement(obj, "name").text = "face"
        ET.SubElement(obj, "pose").text = "Frontal"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(max(0, xmin))
        ET.SubElement(bndbox, "ymin").text = str(max(0, ymin))
        ET.SubElement(bndbox, "xmax").text = str(min(img_w if img_w > 0 else 99999, xmax))
        ET.SubElement(bndbox, "ymax").text = str(min(img_h if img_h > 0 else 99999, ymax))

        # 可选：关键点作为扩展信息
        landmarks = annotation.get("landmarks", {})
        if landmarks:
            kpts_el = ET.SubElement(obj, "keypoints")
            for name, coords in landmarks.items():
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    kp = ET.SubElement(kpts_el, name)
                    kp.set("x", str(coords[0]))
                    kp.set("y", str(coords[1]))

        # 格式化输出
        xml_str = ET.tostring(annotation_el, encoding="unicode")
        # 添加 XML 声明并美化
        import xml.dom.minidom as minidom

        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ")

    # ================================================================
    # 便捷方法
    # ================================================================

    def auto_detect_and_convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        format: str = "coco",
    ) -> List[Path]:
        """
        自动检测输入类型（文件或目录），执行批量转换。

        Args:
            input_path: 输入路径（单个标注 JSON 或标注目录）
            output_path: 输出路径（COCO 为 JSON 文件，YOLO/VOC 为目录）
            format: 目标格式 "coco" | "yolo" | "voc"

        Returns:
            List[Path]: 生成的文件路径列表
        """
        input_path = Path(input_path)

        # 收集标注文件
        if input_path.is_file():
            annotation_paths = [input_path]
        elif input_path.is_dir():
            annotation_paths = sorted(input_path.rglob("*_annotation.json"))
        else:
            logger.error(f"输入路径不存在: {input_path}")
            return []

        if not annotation_paths:
            logger.warning(f"未找到标注文件: {input_path}")
            return []

        # 执行转换
        format_lower = format.lower()
        if format_lower == "coco":
            self.to_coco(annotation_paths, output_path)
            return [Path(output_path)]
        elif format_lower == "yolo":
            return self.to_yolo(annotation_paths, output_path)
        elif format_lower == "voc":
            return self.to_voc(annotation_paths, output_path)
        else:
            logger.error(f"不支持的格式: {format}，可选: coco, yolo, voc")
            return []

    # ================================================================
    # 内部工具方法
    # ================================================================

    def _resolve_image_path(self, image_path: str) -> Path:
        """
        解析标注中引用的图片路径为绝对路径。

        Args:
            image_path: 标注中的 image_path 字段

        Returns:
            Path: 绝对路径
        """
        p = Path(image_path)
        if p.is_absolute():
            return p

        # 尝试项目根目录
        project_root = get_project_root()
        full = project_root / image_path
        if full.exists():
            return full

        # 尝试 image_root_dir
        full = self.image_root_dir / image_path
        return full

    @staticmethod
    def _get_image_size(image_path: Path) -> tuple:
        """
        获取图片尺寸（宽度, 高度）。

        Args:
            image_path: 图片文件路径

        Returns:
            (int, int) 或 (None, None)（失败时）
        """
        try:
            if not image_path.exists():
                return (None, None)
            img = cv2.imread(str(image_path))
            if img is None:
                return (None, None)
            h, w = img.shape[:2]
            return (w, h)
        except Exception:
            return (None, None)

    # ================================================================
    # 反向导入：外部格式 → 项目标注 JSON
    # ================================================================

    def from_coco(
        self,
        coco_path: Union[str, Path],
        output_dir: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        """
        将 COCO JSON 标注文件转换为项目的标注 JSON 格式。

        COCO 关键点顺序对应关系:
          COCO keypoints[0:3] → left_eye
          COCO keypoints[3:6] → right_eye
          COCO keypoints[6:9] → nose_tip
          COCO keypoints[9:12] → left_mouth_corner
          COCO keypoints[12:15] → right_mouth_corner

        Args:
            coco_path: COCO JSON 文件路径
            output_dir: 项目标注 JSON 的输出目录
            image_dir: 图片文件所在目录（用于解析 COCO 中的 file_name），
                       None 则尝试从 COCO 的 image.file_name 相对路径解析

        Returns:
            List[Path]: 生成的标注 JSON 文件路径列表
        """
        import json as json_mod
        from datetime import date

        coco_path = Path(coco_path)
        with open(coco_path, "r", encoding="utf-8") as f:
            coco = json_mod.load(f)

        images = {img["id"]: img for img in coco.get("images", [])}
        categories = {cat["id"]: cat for cat in coco.get("categories", [])}
        annotations = coco.get("annotations", [])

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        keypoint_names = [
            "left_eye", "right_eye", "nose_tip",
            "left_mouth_corner", "right_mouth_corner",
        ]

        generated = []
        for ann in annotations:
            image_id = ann.get("image_id")
            img_info = images.get(image_id)
            if img_info is None:
                logger.warning(f"COCO annotation id={ann['id']}: image_id={image_id} 未找到")
                continue

            # 解析图片路径
            file_name = img_info.get("file_name", "")
            if image_dir:
                image_path = str(Path(image_dir) / Path(file_name).name)
            else:
                image_path = file_name

            # 转换 bbox: COCO [x,y,w,h] → 项目格式
            coco_bbox = ann.get("bbox", [0, 0, 0, 0])
            bbox = {
                "x": int(coco_bbox[0]),
                "y": int(coco_bbox[1]),
                "width": int(coco_bbox[2]),
                "height": int(coco_bbox[3]),
            }

            # 转换关键点: COCO [x1,y1,v1, x2,y2,v2, ...] → 项目格式
            coco_kpts = ann.get("keypoints", [])
            landmarks = {}
            all_invalid = True
            for i, name in enumerate(keypoint_names):
                idx = i * 3
                if idx + 2 < len(coco_kpts):
                    kx = int(coco_kpts[idx])
                    ky = int(coco_kpts[idx + 1])
                    kv = coco_kpts[idx + 2]  # visibility: 0=未标注, 1=遮挡, 2=可见
                    if kv > 0 and kx >= 0 and ky >= 0:
                        landmarks[name] = [kx, ky]
                        all_invalid = False
                    else:
                        landmarks[name] = [-1, -1]
                else:
                    landmarks[name] = [-1, -1]

            # 构建标注
            stem = Path(file_name).stem
            annotation = {
                "image_path": image_path,
                "bbox": bbox,
                "landmarks": landmarks,
                "annotator": "auto" if not all_invalid else "auto_partial",
                "date": date.today().isoformat(),
                "confidence": ann.get("score", None),
                "source_format": "coco",
            }

            # 保存
            output_path = output_dir / f"{stem}_annotation.json"
            save_annotation_json(annotation, output_path)
            generated.append(output_path)

        logger.info(
            f"COCO 导入完成: {len(generated)} 个标注 → {output_dir}"
        )
        return generated

    def from_yolo(
        self,
        label_dir: Union[str, Path],
        output_dir: Union[str, Path],
        image_dir: Union[str, Path],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Path]:
        """
        将 YOLO txt 标注文件转换为项目的标注 JSON 格式。

        YOLO 格式: class_id cx cy w h（归一化坐标 [0,1]）
        每个 .txt 文件对应一张图片，每行一个目标。

        Args:
            label_dir: YOLO 标注 txt 文件目录
            output_dir: 项目标注 JSON 的输出目录
            image_dir: 图片文件所在目录（用于解析图片路径和尺寸）
            image_size: 默认图片尺寸 (width, height)，当无法读取图片时使用

        Returns:
            List[Path]: 生成的标注 JSON 文件路径列表
        """
        from datetime import date

        label_dir = Path(label_dir)
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(label_dir.glob("*.txt"))
        # 过滤掉 .kpt.txt 关键点文件
        txt_files = [f for f in txt_files if not f.name.endswith(".kpt.txt")]

        if not txt_files:
            logger.warning(f"未找到 YOLO .txt 文件: {label_dir}")
            return []

        generated = []

        for txt_path in txt_files:
            # 读取 YOLO 标注
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                continue

            # 找对应的图片文件
            stem = txt_path.stem
            image_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
                candidate = image_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                logger.warning(f"未找到对应图片: {stem}.* → 跳过")
                continue

            # 获取图片尺寸
            img_w, img_h = self._get_image_size(image_path)
            if img_w is None and image_size:
                img_w, img_h = image_size
            if img_w is None:
                logger.warning(f"无法获取图片尺寸: {image_path}")
                continue

            # 只取第一行（单个人脸）
            parts = lines[0].split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h

            # YOLO归一化中心点 → 像素左上角
            x = int(cx - w / 2)
            y = int(cy - h / 2)

            bbox = {
                "x": max(0, x),
                "y": max(0, y),
                "width": int(w),
                "height": int(h),
            }

            # YOLO 本身不含关键点，设为基础值
            landmarks = {
                "left_eye": [-1, -1],
                "right_eye": [-1, -1],
                "nose_tip": [-1, -1],
                "left_mouth_corner": [-1, -1],
                "right_mouth_corner": [-1, -1],
            }

            # 构建标注
            annotation = {
                "image_path": str(image_path.relative_to(self.image_root_dir))
                if self.image_root_dir != Path(".")
                else str(image_path),
                "bbox": bbox,
                "landmarks": landmarks,
                "annotator": "auto_partial",
                "date": date.today().isoformat(),
                "source_format": "yolo",
            }

            output_path = output_dir / f"{stem}_annotation.json"
            from .utils import save_annotation_json
            save_annotation_json(annotation, output_path)
            generated.append(output_path)

        logger.info(
            f"YOLO 导入完成: {len(generated)} 个标注 → {output_dir}"
        )
        return generated

    def from_voc(
        self,
        xml_dir: Union[str, Path],
        output_dir: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        """
        将 Pascal VOC XML 标注文件转换为项目的标注 JSON 格式。

        Args:
            xml_dir: VOC XML 标注文件目录
            output_dir: 项目标注 JSON 的输出目录
            image_dir: 图片文件所在目录，None 则从 XML 的 <path> 或 <folder>/<filename> 推断

        Returns:
            List[Path]: 生成的标注 JSON 文件路径列表
        """
        import xml.etree.ElementTree as ET
        from datetime import date

        xml_dir = Path(xml_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        xml_files = sorted(xml_dir.glob("*.xml"))
        if not xml_files:
            logger.warning(f"未找到 XML 文件: {xml_dir}")
            return []

        generated = []

        for xml_path in xml_files:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                logger.warning(f"XML 解析失败 {xml_path}: {e}")
                continue

            # 提取文件名
            filename_el = root.find("filename")
            filename = filename_el.text if filename_el is not None else xml_path.stem

            # 提取 folder
            folder_el = root.find("folder")
            folder = folder_el.text if folder_el is not None else ""

            # 提取路径
            path_el = root.find("path")
            if path_el is not None and path_el.text:
                image_path = path_el.text
            elif image_dir:
                image_path = str(Path(image_dir) / filename)
            elif folder:
                image_path = f"{folder}/{filename}"
            else:
                image_path = filename

            # 提取尺寸
            size_el = root.find("size")
            img_w, img_h = 0, 0
            if size_el is not None:
                w_el = size_el.find("width")
                h_el = size_el.find("height")
                img_w = int(w_el.text) if w_el is not None and w_el.text else 0
                img_h = int(h_el.text) if h_el is not None and h_el.text else 0

            # 提取 bbox
            bbox = {"x": 0, "y": 0, "width": 0, "height": 0}
            obj_el = root.find("object")
            if obj_el is not None:
                bndbox = obj_el.find("bndbox")
                if bndbox is not None:
                    xmin_el = bndbox.find("xmin")
                    ymin_el = bndbox.find("ymin")
                    xmax_el = bndbox.find("xmax")
                    ymax_el = bndbox.find("ymax")
                    xmin = int(xmin_el.text) if xmin_el is not None and xmin_el.text else 0
                    ymin = int(ymin_el.text) if ymin_el is not None and ymin_el.text else 0
                    xmax = int(xmax_el.text) if xmax_el is not None and xmax_el.text else 0
                    ymax = int(ymax_el.text) if ymax_el is not None and ymax_el.text else 0
                    bbox = {
                        "x": xmin,
                        "y": ymin,
                        "width": xmax - xmin,
                        "height": ymax - ymin,
                    }

            # 提取关键点（扩展字段）
            landmarks = {
                "left_eye": [-1, -1],
                "right_eye": [-1, -1],
                "nose_tip": [-1, -1],
                "left_mouth_corner": [-1, -1],
                "right_mouth_corner": [-1, -1],
            }
            kpts_el = obj_el.find("keypoints") if obj_el is not None else None
            if kpts_el is not None:
                for name in landmarks:
                    kp = kpts_el.find(name)
                    if kp is not None:
                        kx = int(kp.get("x", -1))
                        ky = int(kp.get("y", -1))
                        landmarks[name] = [kx, ky]

            # 构建标注
            stem = Path(filename).stem
            all_invalid = all(
                pt[0] < 0 and pt[1] < 0 for pt in landmarks.values()
            )
            annotation = {
                "image_path": image_path,
                "bbox": bbox,
                "landmarks": landmarks,
                "annotator": "auto" if not all_invalid else "auto_partial",
                "date": date.today().isoformat(),
                "source_format": "voc",
            }

            output_path = output_dir / f"{stem}_annotation.json"
            from .utils import save_annotation_json
            save_annotation_json(annotation, output_path)
            generated.append(output_path)

        logger.info(
            f"VOC 导入完成: {len(generated)} 个标注 → {output_dir}"
        )
        return generated

    # ================================================================
    # CSV 格式导出 / 导入
    # ================================================================

    # CSV 列定义（固定顺序，兼容常见人脸关键点数据集格式）
    CSV_COLUMNS = [
        "image_path",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height",
        "left_eye_x", "left_eye_y",
        "right_eye_x", "right_eye_y",
        "nose_tip_x", "nose_tip_y",
        "left_mouth_corner_x", "left_mouth_corner_y",
        "right_mouth_corner_x", "right_mouth_corner_y",
        "confidence",
        "annotator",
        "date",
        "quality_score",
        "quality_label",
    ]

    def to_csv(
        self,
        annotation_paths: List[Path],
        output_path: Union[str, Path],
        include_header: bool = True,
        include_quality: bool = True,
    ) -> Path:
        """
        将标注导出为 CSV 格式（每行一张人脸，关键点坐标平铺为多列）。

        这是许多公开人脸数据集（300-W、AFLW、Kaggle等）的标准分发格式。

        Args:
            annotation_paths: 标注 JSON 文件路径列表
            output_path: CSV 输出文件路径
            include_header: 是否包含列名表头
            include_quality: 是否包含质量评分列

        Returns:
            Path: 生成的 CSV 文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 根据 include_quality 决定实际列
        columns = list(self.CSV_COLUMNS)
        if not include_quality:
            columns = [c for c in columns if not c.startswith("quality_")]

        rows = []
        for json_path in annotation_paths:
            try:
                ann = load_annotation_json(json_path)
            except Exception as e:
                logger.warning(f"跳过无法读取的文件 {json_path}: {e}")
                continue

            bbox = ann.get("bbox", {})
            landmarks = ann.get("landmarks", {})

            # 将嵌套结构拍平为单行
            row = {
                "image_path": ann.get("image_path", ""),
                "bbox_x": bbox.get("x", 0),
                "bbox_y": bbox.get("y", 0),
                "bbox_width": bbox.get("width", 0),
                "bbox_height": bbox.get("height", 0),
                "left_eye_x": landmarks.get("left_eye", [-1, -1])[0],
                "left_eye_y": landmarks.get("left_eye", [-1, -1])[1],
                "right_eye_x": landmarks.get("right_eye", [-1, -1])[0],
                "right_eye_y": landmarks.get("right_eye", [-1, -1])[1],
                "nose_tip_x": landmarks.get("nose_tip", [-1, -1])[0],
                "nose_tip_y": landmarks.get("nose_tip", [-1, -1])[1],
                "left_mouth_corner_x": landmarks.get("left_mouth_corner", [-1, -1])[0],
                "left_mouth_corner_y": landmarks.get("left_mouth_corner", [-1, -1])[1],
                "right_mouth_corner_x": landmarks.get("right_mouth_corner", [-1, -1])[0],
                "right_mouth_corner_y": landmarks.get("right_mouth_corner", [-1, -1])[1],
                "confidence": ann.get("confidence", ""),
                "annotator": ann.get("annotator", ""),
                "date": ann.get("date", ""),
            }

            if include_quality:
                quality = ann.get("quality", {})
                row["quality_score"] = quality.get("quality_score", "")
                row["quality_label"] = quality.get("quality_label", "")

            rows.append(row)

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if include_header:
                writer.writeheader()
            writer.writerows(rows)

        logger.info(
            f"CSV 导出完成: {len(rows)} 行 → {output_path}"
        )
        return output_path

    def from_csv(
        self,
        csv_path: Union[str, Path],
        output_dir: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        """
        从 CSV 文件导入标注，转换为项目标注 JSON 格式。

        支持两种常见的 CSV 列名风格：
          1. 本项目格式: bbox_x, left_eye_x, ...
          2. 通用关键点格式: x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5

        Args:
            csv_path: CSV 文件路径
            output_dir: 项目标注 JSON 输出目录
            image_dir: 图片所在目录（用于解析 image_path），None 则使用 CSV 中的路径

        Returns:
            List[Path]: 生成的标注 JSON 文件路径列表
        """
        from datetime import date

        csv_path = Path(csv_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

        # 检测 CSV 格式：是项目格式还是通用格式
        is_project_format = any("bbox_" in c or "left_eye_x" in c for c in fieldnames)

        # 重新读取数据行
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        generated = []
        for row_idx, row in enumerate(rows):
            try:
                if is_project_format:
                    annotation = self._csv_row_to_annotation(row, image_dir)
                else:
                    annotation = self._generic_csv_row_to_annotation(row, image_dir)

                stem = f"csv_import_{row_idx + 1:04d}"
                if annotation.get("image_path"):
                    stem = Path(annotation["image_path"]).stem

                output_path = output_dir / f"{stem}_annotation.json"
                save_annotation_json(annotation, output_path)
                generated.append(output_path)
            except Exception as e:
                logger.warning(f"CSV 第 {row_idx + 1} 行导入失败: {e}")

        logger.info(
            f"CSV 导入完成: {len(generated)} 个标注 → {output_dir}"
        )
        return generated

    def _csv_row_to_annotation(
        self,
        row: Dict[str, str],
        image_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """将项目格式的 CSV 行转换为标注字典。"""
        from datetime import date

        image_path = row.get("image_path", "")
        if image_dir:
            image_path = str(Path(image_dir) / Path(image_path).name)

        annotation = {
            "image_path": image_path,
            "bbox": {
                "x": int(float(row.get("bbox_x", 0) or 0)),
                "y": int(float(row.get("bbox_y", 0) or 0)),
                "width": int(float(row.get("bbox_width", 0) or 0)),
                "height": int(float(row.get("bbox_height", 0) or 0)),
            },
            "landmarks": {
                "left_eye": [
                    int(float(row.get("left_eye_x", -1) or -1)),
                    int(float(row.get("left_eye_y", -1) or -1)),
                ],
                "right_eye": [
                    int(float(row.get("right_eye_x", -1) or -1)),
                    int(float(row.get("right_eye_y", -1) or -1)),
                ],
                "nose_tip": [
                    int(float(row.get("nose_tip_x", -1) or -1)),
                    int(float(row.get("nose_tip_y", -1) or -1)),
                ],
                "left_mouth_corner": [
                    int(float(row.get("left_mouth_corner_x", -1) or -1)),
                    int(float(row.get("left_mouth_corner_y", -1) or -1)),
                ],
                "right_mouth_corner": [
                    int(float(row.get("right_mouth_corner_x", -1) or -1)),
                    int(float(row.get("right_mouth_corner_y", -1) or -1)),
                ],
            },
            "annotator": row.get("annotator", "auto"),
            "date": row.get("date", date.today().isoformat()),
            "confidence": float(row["confidence"]) if row.get("confidence") else None,
            "source_format": "csv",
        }
        return annotation

    def _generic_csv_row_to_annotation(
        self,
        row: Dict[str, str],
        image_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        将通用格式的 CSV 行转换为标注字典。

        通用格式列名推测规则：
          - image_path / filename / file → image_path
          - x, y, w, h / xmin, ymin, xmax, ymax → bbox
          - x1~x5, y1~y5 → 5个关键点
        """
        from datetime import date

        # 解析图片路径
        image_path = ""
        for key in ("image_path", "filename", "file", "image"):
            if key in row and row[key]:
                image_path = row[key]
                break
        if image_dir and image_path:
            image_path = str(Path(image_dir) / Path(image_path).name)

        # 解析 bbox（支持多种命名风格）
        bbox = {"x": 0, "y": 0, "width": 0, "height": 0}
        if all(k in row for k in ("bbox_x", "bbox_y", "bbox_width", "bbox_height")):
            bbox = {
                "x": int(float(row["bbox_x"] or 0)),
                "y": int(float(row["bbox_y"] or 0)),
                "width": int(float(row["bbox_width"] or 0)),
                "height": int(float(row["bbox_height"] or 0)),
            }
        elif all(k in row for k in ("xmin", "ymin", "xmax", "ymax")):
            bbox = {
                "x": int(float(row["xmin"] or 0)),
                "y": int(float(row["ymin"] or 0)),
                "width": int(float(row["xmax"] or 0)) - int(float(row["xmin"] or 0)),
                "height": int(float(row["ymax"] or 0)) - int(float(row["ymin"] or 0)),
            }
        elif all(k in row for k in ("x", "y", "w", "h")):
            bbox = {
                "x": int(float(row["x"] or 0)),
                "y": int(float(row["y"] or 0)),
                "width": int(float(row["w"] or 0)),
                "height": int(float(row["h"] or 0)),
            }

        # 解析关键点（x1~x5, y1~y5 风格）
        kp_names = [
            "left_eye", "right_eye", "nose_tip",
            "left_mouth_corner", "right_mouth_corner",
        ]
        landmarks = {}
        for i, name in enumerate(kp_names, 1):
            xi = row.get(f"x{i}")
            yi = row.get(f"y{i}")
            if xi is not None and yi is not None:
                landmarks[name] = [
                    int(float(xi)) if xi else -1,
                    int(float(yi)) if yi else -1,
                ]
            else:
                landmarks[name] = [-1, -1]

        # 如果没有找到 x1~y5 格式的关键点，全部置为 sentinel
        if all(v == [-1, -1] for v in landmarks.values()):
            pass  # 保持 partial 状态

        annotation = {
            "image_path": image_path,
            "bbox": bbox,
            "landmarks": landmarks,
            "annotator": "auto_partial" if all(
                v == [-1, -1] for v in landmarks.values()
            ) else "auto",
            "date": date.today().isoformat(),
            "source_format": "csv",
        }
        return annotation


# ============================================================
# 命令行入口
# ============================================================


def main():
    """格式转换工具的命令行入口。"""
    parser = argparse.ArgumentParser(
        description="标注格式转换工具 — 将标注 JSON 转换为 COCO / YOLO / Pascal VOC 格式",
    )
    parser.add_argument(
        "input",
        help="单个标注 JSON 文件或标注文件目录",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["coco", "yolo", "voc", "csv"],
        default="coco",
        help="目标格式（默认: coco）",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="输出路径（COCO 为 .json 文件，YOLO/VOC 为目录）",
    )
    parser.add_argument(
        "--with-keypoints",
        action="store_true",
        help="YOLO 导出时额外生成 .kpt.txt 关键点文件",
    )
    parser.add_argument(
        "--image-root",
        default=".",
        help="图片根目录（默认: 当前目录）",
    )

    args = parser.parse_args()

    converter = FormatConverter(image_root_dir=args.image_root)

    if args.format == "coco":
        # COCO 输出为单个 JSON
        input_path = Path(args.input)
        if input_path.is_dir():
            annotation_paths = sorted(input_path.rglob("*_annotation.json"))
        else:
            annotation_paths = [input_path]

        converter.to_coco(annotation_paths, args.output)
        print(f"COCO 格式已导出到: {args.output}")

    elif args.format == "yolo":
        # YOLO 输出为目录
        input_path = Path(args.input)
        if input_path.is_dir():
            annotation_paths = sorted(input_path.rglob("*_annotation.json"))
        else:
            annotation_paths = [input_path]

        files = converter.to_yolo(
            annotation_paths, args.output, with_keypoints=args.with_keypoints
        )
        print(f"YOLO 格式已导出 {len(files)} 个文件到: {args.output}")

    elif args.format == "voc":
        # VOC 输出为目录
        input_path = Path(args.input)
        if input_path.is_dir():
            annotation_paths = sorted(input_path.rglob("*_annotation.json"))
        else:
            annotation_paths = [input_path]

        files = converter.to_voc(annotation_paths, args.output)
        print(f"Pascal VOC 格式已导出 {len(files)} 个文件到: {args.output}")

    elif args.format == "csv":
        input_path = Path(args.input)
        if input_path.is_dir():
            annotation_paths = sorted(input_path.rglob("*_annotation.json"))
        else:
            annotation_paths = [input_path]

        out = converter.to_csv(annotation_paths, args.output)
        print(f"CSV 格式已导出到: {out}")

    sys.exit(0)


if __name__ == "__main__":
    main()
