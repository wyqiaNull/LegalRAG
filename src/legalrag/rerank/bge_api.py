"""OpenAI 风格服务上的 bge cross-encoder 重排器。"""

from __future__ import annotations

from typing import Any

import httpx

from ..core import registry
from ..core.errors import ConfigError, RetrievalError
from ..core.interfaces import Candidate, Reranker
from ..core.models import CandidateSource, Query


class BgeApiReranker(Reranker):
    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model: str = "BAAI/bge-reranker-v2-m3",
        timeout: float = 30.0,
    ) -> None:
        if not api_base:
            raise ConfigError("bge_api 需配置 RERANK_API_BASE 或 EMBEDDING_API_BASE")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def rerank(
        self, query: Query, candidates: list[Candidate], top_k: int
    ) -> list[Candidate]:
        if not candidates:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query.text,
            "documents": [candidate.chunk.content for candidate in candidates],
            "top_n": min(top_k, len(candidates)),
            "return_documents": False,
        }
        try:
            response = httpx.post(
                f"{self.api_base}/rerank",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = response.json()["results"]
            reranked = [
                candidates[int(item["index"])].model_copy(
                    update={
                        "score": float(item["relevance_score"]),
                        "source": CandidateSource.RERANKED,
                    }
                )
                for item in results
                if 0 <= int(item["index"]) < len(candidates)
            ]
        except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError) as error:
            raise RetrievalError(f"rerank API 调用失败：{error}") from error
        if not reranked:
            raise RetrievalError("rerank API 未返回有效候选")
        reranked.sort(key=lambda candidate: candidate.score, reverse=True)
        return reranked[:top_k]


registry.register("reranker", "bge_api", BgeApiReranker)
