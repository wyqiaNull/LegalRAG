"""摄取能力层（SPEC §3）：load → chunk → meta → embed → store。

导入本包即注册全部具体 loader / chunker。
"""

from __future__ import annotations

from . import chunkers, loaders  # noqa: F401 —— 触发注册副作用
from .pipeline import IngestPipeline

__all__ = ["IngestPipeline"]
