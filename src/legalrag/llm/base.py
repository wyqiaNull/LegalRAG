"""LLMClient 抽象入口 —— 契约唯一来源在 core.interfaces。"""

from ..core.interfaces import LLMClient  # noqa: F401

__all__ = ["LLMClient"]
