"""
文件名: __init__.py
功能: src 包初始化文件，使人脸标注相关模块可作为 Python 包导入

模块索引:
  - FaceDetector       人脸检测
  - LandmarkExtractor  关键点提取
  - AnnotationGenerator 自动标注生成
  - BatchProcessor      批量处理
  - ManualReviewer      人工修正审核
  - FaceSignature       人脸特征签名与比对
  - DatasetOrganizer    数据集整理迁移（支持智能分组）
  - Visualizer          标注可视化
  - AnnotationValidator 标注校验
"""

from .face_detector import FaceDetector
from .landmark_extractor import LandmarkExtractor
from .annotation_generator import AnnotationGenerator
from .batch_processor import BatchProcessor
from .manual_reviewer import ManualReviewer
from .visualizer import Visualizer
from .validator import AnnotationValidator
from .quality_filter import QualityFilter
from .format_converter import FormatConverter
from .image_preprocessor import ImagePreprocessor, PreprocessPipeline
from .annotation_corrector import AnnotationCorrector

__all__ = [
    "FaceDetector",
    "LandmarkExtractor",
    "AnnotationGenerator",
    "BatchProcessor",
    "ManualReviewer",
    "Visualizer",
    "AnnotationValidator",
    "QualityFilter",
    "FormatConverter",
    "ImagePreprocessor",
    "PreprocessPipeline",
    "AnnotationCorrector",
]
