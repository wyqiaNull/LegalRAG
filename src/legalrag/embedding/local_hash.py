"""本地确定性 embedder（仅供离线开发 / 冒烟测试，非生产质量）。

用字符 bigram 的哈希装桶得到定长向量 —— 只捕捉词面重叠，不具语义能力，
但无需 GPU/API 即可让 ingest→query 链路端到端跑通（类比 NoopReranker 的占位角色）。
生产路径请用 bge_m3_api。
"""

from __future__ import annotations

import hashlib
import math
import re

from ..core.interfaces import EmbedResult, Embedder
from ..core import registry

_TOKEN = re.compile(r"[0-9a-zA-Z]+|[一-鿿]")


def _bucket(feature: str, dim: int) -> int:
    # 用 sha1 而非内置 hash()：后者受 PYTHONHASHSEED 影响，跨进程不稳定，
    # 会导致 ingest 与 query 两次调用得到不一致的向量。
    h = hashlib.sha1(feature.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % dim


def _features(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    feats = list(toks)
    feats += [toks[i] + toks[i + 1] for i in range(len(toks) - 1)]  # bigram
    return feats


class LocalHashEmbedder(Embedder):
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for f in _features(text):
            v[_bucket(f, self.dim)] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> EmbedResult:
        return EmbedResult(dense=[self._vec(t) for t in texts], sparse=None)


registry.register("embedder", "local_hash", LocalHashEmbedder)
