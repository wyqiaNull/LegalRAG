"""纯向量单阶段检索（SPEC §6，FR-3 MVP）。

query 文本 → 向量 → 向量库 ANN。filters 参数恒在，MVP 传空（v0.2 塞权限/版本条件）。
双阶段混合检索见 v0.1 的 hybrid_rerank.py。
"""

from __future__ import annotations

from ...core import registry
from ...core.errors import RetrievalError
from ...core.interfaces import Candidate, Embedder, Filter, Retriever, VectorStore
from ...core.models import Query


class VectorOnlyRetriever(Retriever):
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self, query: Query, filters: Filter | None, top_k: int
    ) -> list[Candidate]:
        result = self.embedder.embed([query.text])
        if not result.dense:
            raise RetrievalError("query 向量化返回为空")
        return self.vector_store.search(result.dense[0], filters, top_k)


registry.register("retriever", "vector_only", VectorOnlyRetriever)
