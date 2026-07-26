"""bge-m3 API 实现（走 API，无 GPU 退路）。

调用 OpenAI 兼容的 /embeddings 端点分批取得 dense 向量。
"""

from __future__ import annotations

import httpx

from ..core.errors import ConfigError, RetrievalError
from ..core.interfaces import EmbedResult, Embedder
from ..core import registry
from ..observability import UsageCollector


class BgeM3ApiEmbedder(Embedder):
    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model: str = "bge-m3",
        dim: int = 1024,
        batch_size: int = 32,
        timeout: float = 30.0,
        usage_collector: UsageCollector | None = None,
    ) -> None:
        if not api_base:
            raise ConfigError("bge_m3_api 需配置 EMBEDDING_API_BASE")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        if batch_size < 1:
            raise ConfigError("embedding batch_size 必须大于 0")
        self.batch_size = batch_size
        self.timeout = timeout
        self.usage_collector = usage_collector

    def embed(self, texts: list[str]) -> EmbedResult:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        dense: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                resp = httpx.post(
                    f"{self.api_base}/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": batch},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                body = resp.json()
                if self.usage_collector is not None:
                    self.usage_collector.record(body)
                data = sorted(body["data"], key=lambda item: item.get("index", 0))
                embeddings = [item["embedding"] for item in data]
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
                raise RetrievalError(f"embedding API 调用失败：{e}") from e
            if len(embeddings) != len(batch):
                raise RetrievalError("embedding API 返回数量与输入不一致")
            dense.extend(embeddings)
        return EmbedResult(dense=dense, sparse=None)


registry.register("embedder", "bge_m3_api", BgeM3ApiEmbedder)
