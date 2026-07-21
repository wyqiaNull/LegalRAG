"""Reranker 抽象入口 —— 契约唯一来源在 core.interfaces（SPEC §4.2）。"""

from ..core.interfaces import Reranker  # noqa: F401

__all__ = ["Reranker"]
