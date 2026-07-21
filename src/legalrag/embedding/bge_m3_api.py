"""bge-m3 API 实现（走 API，无 GPU 退路）。

调用 OpenAI 兼容的 /embeddings 端点取 dense 向量。sparse 预留，v0.1 启用。
"""

from __future__ import annotations

import httpx

from ..core.errors import ConfigError, RetrievalError
from ..core.interfaces import EmbedResult, Embedder
from ..core import registry


class BgeM3ApiEmbedder(Embedder):
    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model: str = "bge-m3",
        dim: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        if not api_base:
            raise ConfigError("bge_m3_api 需配置 EMBEDDING_API_BASE")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout = timeout

    def embed(self, texts: list[str]) -> EmbedResult:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = httpx.post(
                f"{self.api_base}/embeddings",
                headers=headers,
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        except (httpx.HTTPError, KeyError) as e:
            raise RetrievalError(f"embedding API 调用失败：{e}") from e
        dense = [item["embedding"] for item in data]
        return EmbedResult(dense=dense, sparse=None)


registry.register("embedder", "bge_m3_api", BgeM3ApiEmbedder)
