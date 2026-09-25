"""
文件名: visualizer.py
功能: 标注可视化工具，在图片上绘制边界框（bbox）和关键点（landmarks）
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .utils import load_annotation_json, load_image


class Visualizer:
    """
    标注可视化工具，在图片上绘制边界框和关键点。

    支持读取标注 JSON 文件，将 bbox（绿色矩形）和 5 个简化关键点（红色圆+标签）
    绘制在原图上，并支持窗口显示和保存。

    Attributes:
        bbox_color: 边界框颜色 (BGR)，默认绿色
        landmark_color: 关键点颜色 (BGR)，默认红色
        line_thickness: 边界框线宽
        landmark_radius: 关键点圆半径
        font_scale: 标签字体大小
    """

    # 关键点对应的标签
    LANDMARK_LABELS = {
        "left_eye": "LE",
        "right_eye": "RE",
        "nose_tip": "N",
        "left_mouth_corner": "LM",
        "right_mouth_corner": "RM",
    }

    def __init__(
        self,
        bbox_color: Tuple[int, int, int] = (0, 255, 0),       # 绿色
        landmark_color: Tuple[int, int, int] = (0, 0, 255),   # 红色
        line_thickness: int = 2,
        landmark_radius: int = 4,
        font_scale: float = 0.5,
    ):
        """
        初始化可视化工具及绘制样式。

        Args:
            bbox_color: 边界框颜色 (B, G, R)，默认绿色
            landmark_color: 关键点颜色 (B, G, R)，默认红色
            line_thickness: 边界框线宽（像素）
            landmark_radius: 关键点圆半径（像素）
            font_scale: 标签文字缩放比例
        """
        self.bbox_color = bbox_color
        self.landmark_color = landmark_color
        self.line_thickness = line_thickness
        self.landmark_radius = landmark_radius
        self.font_scale = font_scale

    def draw_annotation(
        self,
        image_path: Union[str, Path],
        annotation_path: Union[str, Path],
    ) -> np.ndarray:
        """
        读取图片和标注文件，在图片上绘制标注可视化。

        Args:
            image_path: 原始图片路径
            annotation_path: 标注 JSON 文件路径

        Returns:
            np.ndarray: 绘制了 bbox 和 landmarks 的 BGR 图片数组
        """
        # 加载图片和标注
        image = load_image(image_path)
        annotation = load_annotation_json(annotation_path)

        # 绘制 bbox
        bbox = annotation.get("bbox")
        if bbox:
            self.draw_bbox(image, bbox)

        # 绘制 landmarks
        landmarks = annotation.get("landmarks")
        if landmarks:
            self.draw_landmarks(image, landmarks)

        return image

    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: Dict[str, int],
        label: Optional[str] = None,
    ) -> np.ndarray:
        """
        在图片上绘制矩形边界框（原地修改）。

        Args:
            image: BGR 图片数组
            bbox: {"x": int, "y": int, "width": int, "height": int}
            label: 可选标签文本（绘制在 bbox 左上角上方）

        Returns:
            np.ndarray: 同一图片数组（原地修改后的引用）
        """
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        # 绘制矩形
        cv2.rectangle(image, (x, y), (x + w, y + h), self.bbox_color, self.line_thickness)

        # 绘制标签
        if label:
            label_y = y - 5 if y > 15 else y + h + 20
            cv2.putText(
                image,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.bbox_color,
                self.line_thickness,
                cv2.LINE_AA,
            )

        return image

    def draw_landmarks(
        self,
        image: np.ndarray,
        landmarks: Dict[str, List[int]],
        draw_labels: bool = True,
    ) -> np.ndarray:
        """
        在图片上绘制关键点及标签（原地修改）。

        对于坐标为 [-1, -1] 的无效关键点会自动跳过。

        Args:
            image: BGR 图片数组
            landmarks: {"left_eye": [x,y], ...}
            draw_labels: 是否绘制文字标签

        Returns:
            np.ndarray: 同一图片数组（原地修改后的引用）
        """
        for name, (lx, ly) in landmarks.items():
            # 跳过无效关键点（partial 标注可能为 -1）
            if lx < 0 or ly < 0:
                continue

            center = (lx, ly)

            # 绘制实心圆
            cv2.circle(
                image,
                center,
                self.landmark_radius,
                self.landmark_color,
                thickness=-1,  # -1 表示填充
            )

            # 绘制标签文字
            if draw_labels:
                label = self.LANDMARK_LABELS.get(name, name[:2].upper())
                text_x = lx + self.landmark_radius + 2
                text_y = ly - self.landmark_radius - 2

                cv2.putText(
                    image,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    self.landmark_color,
                    self.line_thickness,
                    cv2.LINE_AA,
                )

        return image

    @staticmethod
    def save_visualization(
        image: np.ndarray,
        output_path: Union[str, Path],
    ) -> None:
        """
        保存可视化结果图片。

        Args:
            image: BGR 图片数组
            output_path: 输出文件路径
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(path), image)
        if not success:
            raise IOError(f"无法保存图片到: {path}")
        print(f"可视化结果已保存到: {path}")

    @staticmethod
    def show_visualization(
        image: np.ndarray,
        window_name: str = "Annotation Viewer",
        max_width: int = 1200,
        max_height: int = 900,
    ) -> None:
        """
        在窗口中显示可视化结果，按任意键关闭。

        图片过大时会自动缩放以适应屏幕。

        Args:
            image: BGR 图片数组
            window_name: 窗口标题
            max_width: 窗口最大宽度
            max_height: 窗口最大高度
        """
        h, w = image.shape[:2]

        # 若图片过大则缩放显示
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            display_w = int(w * scale)
            display_h = int(h * scale)
            display_image = cv2.resize(image, (display_w, display_h))
        else:
            display_image = image

        cv2.imshow(window_name, display_image)
        print(f"正在显示窗口 '{window_name}'，按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ============================================================
# 命令行入口
# ============================================================


def main():
    """标注可视化工具的命令行入口。"""
    parser = argparse.ArgumentParser(
        description="标注可视化工具 — 在图片上绘制边界框和关键点",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="原始图片路径",
    )
    parser.add_argument(
        "--annotation",
        required=True,
        help="标注 JSON 文件路径",
    )
    parser.add_argument(
        "--output",
        help="保存可视化结果的图片路径（可选）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="显示可视化窗口（默认开启）",
    )
    parser.add_argument(
        "--no-show",
        action="store_false",
        dest="show",
        help="不显示窗口",
    )

    args = parser.parse_args()

    vis = Visualizer()
    result = vis.draw_annotation(args.image, args.annotation)

    if args.output:
        vis.save_visualization(result, Path(args.output))

    if args.show:
        vis.show_visualization(result)


if __name__ == "__main__":
    main()
