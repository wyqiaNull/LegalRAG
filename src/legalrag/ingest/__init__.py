"""摄取能力层：load → chunk → meta → embed → store。

导入本包即注册全部具体 loader / chunker。
"""

from __future__ import annotations

from . import chunkers, loaders  # noqa: F401 —— 触发注册副作用
from .pipeline import IngestPipeline
from .versioning import IngestResult, IngestStats

__all__ = ["IngestPipeline", "IngestResult", "IngestStats"]
