"""
文件名: face_signature.py
功能: 基于 insightface ArcFace (buffalo_l) 的深度学习人脸识别
      提取 512 维 embedding，用余弦相似度判断是否同一个人

依赖: insightface + onnxruntime
模型: buffalo_l (需手动下载 ONNX 文件到 ~/.insightface/models/buffalo_l/)

相似度参考（ArcFace 余弦相似度，已 L2 归一化）:
  >= 0.60  → 几乎肯定同一人
  0.40-0.60 → 不确定区域
  < 0.40  → 几乎肯定不同人
  < 0  → 完全不同的两个人
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .utils import load_image

# ============================================================
# 全局单例
# ============================================================

_face_app = None


def _get_face_app():
    """获取 insightface FaceAnalysis 单例（懒加载）。"""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis

        _face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0)  # CPU
    return _face_app


# ============================================================
# Embedding 提取
# ============================================================


def extract_embedding(
    image: Union[str, Path, np.ndarray],
) -> Optional[np.ndarray]:
    """
    提取人脸 512 维 ArcFace embedding（L2 归一化）。

    Args:
        image: 图片路径或 BGR 数组

    Returns:
        Optional[np.ndarray]: 512 维归一化 embedding，无人脸返回 None
    """
    if isinstance(image, (str, Path)):
        image = load_image(image)

    app = _get_face_app()
    faces = app.get(image)

    if not faces:
        return None

    # 取面积最大的人脸
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = best.embedding  # shape: (512,)

    # L2 归一化
    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm

    return emb


# ============================================================
# 人脸比对
# ============================================================


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """两个已归一化 embedding 的余弦相似度。"""
    return float(np.dot(emb1, emb2))


class FaceMatcher:
    """
    基于 ArcFace embedding 的人脸比对器。

    阈值建议:
    - strict:  0.55 — 宁可不合并，减少误合并
    - normal:  0.45 — 平衡模式
    - relaxed: 0.35 — 宁可多合并，减少漏合并

    ArcFace 余弦相似度的典型分布:
    - 同一人不同照片: 通常 0.55-0.90
    - 不同人: 通常 -0.10-0.40（大部分在 0.20 以下）
    """

    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return cosine_similarity(emb1, emb2)

    def is_same_person(
        self, emb1: np.ndarray, emb2: np.ndarray
    ) -> Tuple[bool, float]:
        score = self.compare(emb1, emb2)
        return (score >= self.threshold, score)

    def group_by_identity(
        self,
        embeddings: Dict[str, np.ndarray],
        threshold: Optional[float] = None,
    ) -> List[List[str]]:
        """并查集聚类。"""
        thresh = threshold if threshold is not None else self.threshold
        ids = list(embeddings.keys())
        n = len(ids)

        if n <= 1:
            return [[ids[0]]] if n == 1 else []

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if self.compare(embeddings[ids[i]], embeddings[ids[j]]) >= thresh:
                    union(i, j)

        groups: Dict[int, List[str]] = {}
        for i, img_id in enumerate(ids):
            root = find(i)
            groups.setdefault(root, []).append(img_id)

        return sorted(groups.values(), key=len, reverse=True)

    def compute_similarity_matrix(
        self, embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """生成完整相似度矩阵。"""
        ids = list(embeddings.keys())
        matrix: Dict[str, Dict[str, float]] = {}
        for i, id1 in enumerate(ids):
            matrix[id1] = {}
            for j, id2 in enumerate(ids):
                if i == j:
                    matrix[id1][id2] = 1.0
                elif j > i:
                    sim = self.compare(embeddings[id1], embeddings[id2])
                    matrix[id1][id2] = sim
                    matrix.setdefault(id2, {})[id1] = sim
        return matrix

    def find_duplicate_groups(
        self,
        signatures: Dict[str, dict],
        threshold: Optional[float] = None,
    ) -> List[List[str]]:
        """
        基于人脸特征签名对图片做聚类分组（用于智能分组）。

        从每个签名中提取 embedding，然后调用 group_by_identity() 做并查集聚类。
        同一人物的不同照片会被归入同一组。

        Args:
            signatures: Dict[图片路径, 签名dict]，每个签名 dict 必须包含 "embedding" 键
            threshold: 相似度阈值，默认使用实例阈值

        Returns:
            List[List[str]]: 按组大小降序排列的图片路径分组
        """
        # 提取 embedding
        embeddings: Dict[str, np.ndarray] = {}
        for path, sig in signatures.items():
            emb = sig.get("embedding")
            if emb is not None and isinstance(emb, np.ndarray) and len(emb) > 0:
                embeddings[path] = emb

        if len(embeddings) < 2:
            # 少于2张图片，无法聚类
            key_list = list(signatures.keys())
            return [key_list] if key_list else []

        return self.group_by_identity(embeddings, threshold=threshold)

    def find_exact_duplicates(
        self,
        signatures: Dict[str, dict],
    ) -> List[List[str]]:
        """
        发现完全重复的图片（人脸 embedding 几乎相同，余弦相似度 > 0.995）。

        用于检测同一张图片被多次导入的情况。

        Args:
            signatures: Dict[图片路径, 签名dict]，每个签名 dict 必须包含 "embedding" 键

        Returns:
            List[List[str]]: 完全重复的图片分组列表，每组至少2张，按组大小降序
        """
        # 提取 embedding
        embeddings: Dict[str, np.ndarray] = {}
        for path, sig in signatures.items():
            emb = sig.get("embedding")
            if emb is not None and isinstance(emb, np.ndarray) and len(emb) > 0:
                embeddings[path] = emb

        if len(embeddings) < 2:
            return []

        # 使用极高阈值（0.995）来判断完全重复
        ids = list(embeddings.keys())
        n = len(ids)

        exact_groups: Dict[int, List[str]] = {}
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compare(embeddings[ids[i]], embeddings[ids[j]])
                if sim >= 0.995:
                    union(i, j)

        for i, img_id in enumerate(ids):
            root = find(i)
            exact_groups.setdefault(root, []).append(img_id)

        # 只返回包含至少2张图片的组
        result = [g for g in exact_groups.values() if len(g) >= 2]
        return sorted(result, key=len, reverse=True)


# ============================================================
# 便捷函数（保持与旧接口兼容）
# ============================================================


def extract_face_signature(
    image: Union[str, Path, np.ndarray],
    bbox: Optional[Dict[str, int]] = None,
    landmarks: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    """
    提取人脸签名（兼容旧接口，实际只返回 embedding）。
    bbox 和 landmarks 参数保留以兼容，但 insightface 内部自行检测。
    """
    emb = extract_embedding(image)
    return {"embedding": emb, "bbox": bbox}


def compare_two_images(
    image1: Union[str, Path],
    image2: Union[str, Path],
    bbox1: Optional[Dict[str, int]] = None,
    bbox2: Optional[Dict[str, int]] = None,
    landmarks1: Optional[Dict[str, List[int]]] = None,
    landmarks2: Optional[Dict[str, List[int]]] = None,
    threshold: float = 0.45,
) -> Tuple[bool, float]:
    """便捷函数：比较两张图片中的人脸。"""
    emb1 = extract_embedding(image1)
    emb2 = extract_embedding(image2)
    if emb1 is None or emb2 is None:
        return (False, 0.0)
    matcher = FaceMatcher(threshold=threshold)
    return matcher.is_same_person(emb1, emb2)


def describe_match(score: float, threshold: float = 0.45) -> str:
    """相似度分数的可读描述。"""
    if score >= 0.60:
        return f"高度匹配 ({score:.3f}) — 几乎肯定同一人"
    elif score >= threshold:
        return f"匹配 ({score:.3f}) — 很可能是同一人"
    elif score >= 0.30:
        return f"不确定 ({score:.3f}) — 可能需要人工判断"
    else:
        return f"不匹配 ({score:.3f}) — 很可能是不同人"
