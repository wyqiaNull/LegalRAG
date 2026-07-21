"""检索器集合。

TODO[v0.1]：hybrid_rerank.py —— BM25+向量双路 → RRF 融合 → cross-encoder 重排。
"""

from __future__ import annotations

from . import vector_only  # noqa: F401 —— 触发注册副作用
