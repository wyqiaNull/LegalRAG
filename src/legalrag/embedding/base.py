"""Embedder 抽象入口 —— 契约唯一来源在 core.interfaces。

本文件按目录约定保留，仅做再导出，避免契约多处定义。
"""

from ..core.interfaces import EmbedResult, Embedder  # noqa: F401

__all__ = ["Embedder", "EmbedResult"]
