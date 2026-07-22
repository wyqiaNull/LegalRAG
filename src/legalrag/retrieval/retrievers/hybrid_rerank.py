"""BM25 与稠密向量双路召回，经 RRF 融合后返回候选。"""

from __future__ import annotations

from typing import Callable, cast

from ...core import registry
from ...core.errors import RetrievalError
from ...core.interfaces import Candidate, Embedder, Filter, Retriever, VectorStore
from ...core.models import Query
from ..fusion import reciprocal_rank_fusion

SparseSearch = Callable[[str, Filter | None, int], list[Candidate]]


class HybridRetriever(Retriever):
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.rrf_k = rrf_k

    def search(
        self, query: Query, filters: Filter | None, top_k: int
    ) -> list[Candidate]:
        result = self.embedder.embed([query.text])
        if not result.dense:
            raise RetrievalError("query 向量化返回为空")
        sparse_search = getattr(self.vector_store, "search_sparse", None)
        if not callable(sparse_search):
            raise RetrievalError("当前向量存储不支持稀疏检索")
        dense = self.vector_store.search(result.dense[0], filters, top_k)
        sparse = cast(SparseSearch, sparse_search)(query.text, filters, top_k)
        return reciprocal_rank_fusion(
            [dense, sparse], k=self.rrf_k, top_n=top_k
        )


registry.register("retriever", "hybrid_rerank", HybridRetriever)
