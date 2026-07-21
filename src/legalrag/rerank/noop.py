"""NoopReranker（SPEC §3）—— 占位，原样返回候选并截断，保持链路形状。

真实 cross-encoder 重排见 v0.1 的 bge_api.py。
"""

from __future__ import annotations

from ..core import registry
from ..core.interfaces import Candidate, Reranker
from ..core.models import Query


class NoopReranker(Reranker):
    def rerank(
        self, query: Query, candidates: list[Candidate], top_k: int
    ) -> list[Candidate]:
        return candidates[:top_k]


registry.register("reranker", "noop", NoopReranker)
