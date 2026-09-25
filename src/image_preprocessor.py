"""
文件名: image_preprocessor.py
功能: 图片预处理管道，提供去噪、增强、格式统一等清洗操作

支持的预处理操作:
  1. 高斯滤波去噪 — 去除高斯噪声，保留边缘
  2. CLAHE 亮度增强 — 自适应直方图均衡化，改善光照不均
  3. 中值滤波 — 去除椒盐噪声
  4. 格式统一 — 统一尺寸、色彩空间、文件格式
  5. 管道模式 — 链式组合多个预处理操作

使用示例:
  # 单步操作
  preprocessor = ImagePreprocessor()
  cleaned = preprocessor.denoise(image)
  enhanced = preprocessor.enhance_brightness(image)

  # 管道模式
  pipe = PreprocessPipeline()
  pipe.add_gaussian_blur(kernel_size=(3,3))
  pipe.add_clahe(clip_limit=2.0)
  result = pipe.apply(image)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class ImagePreprocessor:
    """
    图片预处理器，提供各类清洗操作的统一入口。

    Attributes:
        default_size: 统一缩放的目标尺寸 (width, height)，None 表示不缩放
        default_format: 统一保存的格式扩展名，如 ".jpg"
    """

    def __init__(
        self,
        default_size: Optional[Tuple[int, int]] = None,
        default_format: str = ".jpg",
    ):
        """
        初始化预处理器。

        Args:
            default_size: 统一缩放目标尺寸，如 (1280, 720)，None 则不缩放
            default_format: 统一保存格式
        """
        self.default_size = default_size
        self.default_format = default_format

    # ================================================================
    # 去噪操作
    # ================================================================

    def denoise_gaussian(
        self,
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (3, 3),
        sigma: float = 1.0,
    ) -> np.ndarray:
        """
        高斯滤波去噪，适用于去除高斯噪声。

        Args:
            image: BGR 图片数组
            kernel_size: 高斯核大小，必须为奇数，如 (3,3) (5,5)
            sigma: 标准差

        Returns:
            np.ndarray: 去噪后的图片
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def denoise_median(
        self,
        image: np.ndarray,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """
        中值滤波去噪，适用于去除椒盐噪声。

        Args:
            image: BGR 图片数组
            kernel_size: 核大小，必须为奇数

        Returns:
            np.ndarray: 去噪后的图片
        """
        return cv2.medianBlur(image, kernel_size)

    def denoise_bilateral(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75,
    ) -> np.ndarray:
        """
        双边滤波去噪，保留边缘的同时去除噪声（适合人脸图片）。

        Args:
            image: BGR 图片数组
            d: 像素邻域直径
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差

        Returns:
            np.ndarray: 去噪后的图片
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def denoise_nlmeans(
        self,
        image: np.ndarray,
        h: float = 10,
        template_window: int = 7,
        search_window: int = 21,
    ) -> np.ndarray:
        """
        非局部均值去噪，效果最好但速度较慢。

        Args:
            image: BGR 图片数组
            h: 滤波强度，越大去噪越强但细节丢失越多
            template_window: 模板窗口大小（奇数）
            search_window: 搜索窗口大小（奇数）

        Returns:
            np.ndarray: 去噪后的图片
        """
        return cv2.fastNlMeansDenoisingColored(
            image, None, h, h, template_window, search_window
        )

    # ================================================================
    # 亮度增强
    # ================================================================

    def enhance_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        CLAHE 自适应直方图均衡化，改善光照不均匀问题。

        相比全局直方图均衡化，CLAHE 在局部区域做均衡化，
        不会过度放大噪声，适合人脸图片。

        Args:
            image: BGR 图片数组
            clip_limit: 对比度限制阈值
            tile_grid_size: 网格大小

        Returns:
            np.ndarray: 增强后的图片
        """
        # 转换到 LAB 色彩空间（仅在亮度通道做 CLAHE）
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )
        l_eq = clahe.apply(l)

        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def enhance_brightness(
        self,
        image: np.ndarray,
        alpha: float = 1.2,
        beta: float = 10,
    ) -> np.ndarray:
        """
        简单的亮度/对比度调整。

        output = alpha * image + beta

        Args:
            image: BGR 图片数组
            alpha: 对比度系数 (>1 增强对比度)
            beta: 亮度增量 (>0 提亮)

        Returns:
            np.ndarray: 调整后的图片
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # ================================================================
    # 格式统一
    # ================================================================

    def resize(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        keep_aspect: bool = True,
    ) -> np.ndarray:
        """
        统一图片尺寸。

        Args:
            image: BGR 图片数组
            target_size: 目标尺寸 (width, height)，None 则使用 default_size
            keep_aspect: 是否保持宽高比（若保持，则缩放后做居中裁剪或填充）

        Returns:
            np.ndarray: 调整后的图片
        """
        size = target_size or self.default_size
        if size is None:
            return image

        tw, th = size
        h, w = image.shape[:2]

        if keep_aspect:
            # 等比缩放使短边匹配目标，然后居中裁剪
            scale = max(tw / w, th / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))

            # 居中裁剪
            x1 = (new_w - tw) // 2
            y1 = (new_h - th) // 2
            return resized[y1:y1 + th, x1:x1 + tw]
        else:
            return cv2.resize(image, (tw, th))

    def normalize_color(
        self,
        image: np.ndarray,
        target_space: str = "BGR",
    ) -> np.ndarray:
        """
        统一色彩空间。

        Args:
            image: 输入图片数组
            target_space: 目标色彩空间，"BGR" 或 "RGB"

        Returns:
            np.ndarray: 转换后的图片
        """
        target = target_space.upper()
        if target == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif target == "BGR":
            # 假设输入是 BGR，不需要转换
            return image
        else:
            raise ValueError(f"不支持的目标色彩空间: {target_space}")

    def save_unified(
        self,
        image: np.ndarray,
        output_path: Union[str, Path],
        format: Optional[str] = None,
        quality: int = 92,
    ) -> Path:
        """
        以统一格式保存图片。

        Args:
            image: BGR 图片数组
            output_path: 输出路径
            format: 格式扩展名，None 则使用 default_format
            quality: JPEG 质量 (1-100)

        Returns:
            Path: 实际保存路径
        """
        out = Path(output_path)
        fmt = format or self.default_format or out.suffix
        if not fmt.startswith("."):
            fmt = f".{fmt}"
        out = out.with_suffix(fmt)
        out.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return out


class PreprocessPipeline:
    """
    预处理管道，支持链式组合多个预处理操作。

    Example:
        pipe = PreprocessPipeline()
        pipe.add(ImagePreprocessor().denoise_gaussian, kernel_size=(3,3))
        pipe.add(ImagePreprocessor().enhance_clahe, clip_limit=2.0)
        result = pipe.apply(image)
    """

    def __init__(self):
        """初始化空管道。"""
        self._steps: List[Dict[str, Any]] = []

    def add(self, operation, **kwargs):
        """
        向管道添加一个预处理步骤。

        Args:
            operation: 可调用对象，接受 np.ndarray 作为第一个参数，返回 np.ndarray
            **kwargs: 传递给 operation 的额外关键字参数

        Returns:
            self: 支持链式调用
        """
        self._steps.append({"op": operation, "kwargs": kwargs})
        return self

    def add_gaussian_blur(self, kernel_size=(3, 3), sigma=1.0):
        """添加高斯模糊步骤。"""
        pp = ImagePreprocessor()
        return self.add(pp.denoise_gaussian, kernel_size=kernel_size, sigma=sigma)

    def add_median_blur(self, kernel_size=3):
        """添加中值滤波步骤。"""
        pp = ImagePreprocessor()
        return self.add(pp.denoise_median, kernel_size=kernel_size)

    def add_clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """添加 CLAHE 增强步骤。"""
        pp = ImagePreprocessor()
        return self.add(pp.enhance_clahe, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    def add_brightness(self, alpha=1.2, beta=10):
        """添加亮度调整步骤。"""
        pp = ImagePreprocessor()
        return self.add(pp.enhance_brightness, alpha=alpha, beta=beta)

    def add_resize(self, target_size, keep_aspect=True):
        """添加尺寸统一步骤。"""
        pp = ImagePreprocessor()
        return self.add(pp.resize, target_size=target_size, keep_aspect=keep_aspect)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        对图片依次应用管道中的所有步骤。

        Args:
            image: 输入图片数组

        Returns:
            np.ndarray: 处理后的图片
        """
        result = image
        for step in self._steps:
            try:
                result = step["op"](result, **step["kwargs"])
            except Exception as e:
                # 某个步骤失败时跳过，继续执行后续步骤
                print(f"  [WARN] 预处理步骤失败: {e}，跳过")
                continue
        return result

    def apply_to_file(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save: bool = True,
    ) -> np.ndarray:
        """
        对文件应用预处理管道。

        Args:
            image_path: 输入图片路径
            output_path: 输出路径，None 则覆盖原文件
            save: 是否保存结果

        Returns:
            np.ndarray: 处理后的图片
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        result = self.apply(img)

        if save:
            out = Path(output_path) if output_path else Path(image_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), result)

        return result

    @property
    def steps_description(self) -> List[str]:
        """返回管道的可读描述列表。"""
        desc = []
        for step in self._steps:
            op_name = step["op"].__self__.__class__.__name__ if hasattr(step["op"], "__self__") else step["op"].__name__
            kwargs_str = ", ".join(f"{k}={v}" for k, v in step["kwargs"].items())
            desc.append(f"{op_name}({kwargs_str})")
        return desc


# ============================================================
# 推荐预设
# ============================================================


def create_face_cleaning_pipeline() -> PreprocessPipeline:
    """
    创建推荐的人脸图片清洗管道。

    包含: 双边滤波（保边去噪）→ CLAHE（亮度均衡）

    Returns:
        PreprocessPipeline: 预配置的管道
    """
    pipe = PreprocessPipeline()
    # 双边滤波：去除噪声同时保留人脸边缘
    pipe.add(ImagePreprocessor().denoise_bilateral, d=7, sigma_color=50, sigma_space=50)
    # CLAHE：改善光照不均
    pipe.add(ImagePreprocessor().enhance_clahe, clip_limit=2.0, tile_grid_size=(8, 8))
    return pipe


def create_light_cleaning_pipeline() -> PreprocessPipeline:
    """
    创建轻度清洗管道（仅 CLAHE 亮度增强，不去噪）。

    适用于质量较好的图片，仅改善光照。

    Returns:
        PreprocessPipeline: 预配置的管道
    """
    pipe = PreprocessPipeline()
    pipe.add_clahe(clip_limit=1.5)
    return pipe
