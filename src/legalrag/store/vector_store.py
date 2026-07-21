"""VectorStore：标量过滤 + ANN 一体（SPEC §4.2）。

MVP 提供内存实现 MemoryVectorStore，接口与 Milvus 对齐（upsert / 带 filter 的 search），
并落盘到 JSON 以便 ingest 与 query 两次独立 CLI 调用间持久化。量大再切 Milvus，主流程零改动。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..core import registry
from ..core.interfaces import Candidate, Filter, VectorStore
from ..core.models import CandidateSource, Chunk


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _passes(chunk_meta: dict[str, Any], filters: Filter | None) -> bool:
    """MVP 空过滤 → 全通过。给出的条件按 等值 / 列表成员 匹配（v0.2 细化语义）。"""
    if not filters:
        return True
    for key, want in filters.items():
        got = chunk_meta.get(key)
        if isinstance(want, list):
            if got not in want:
                return False
        elif got != want:
            return False
    return True


class MemoryVectorStore(VectorStore):
    def __init__(self, path: str = "data/processed") -> None:
        self._file = Path(path) / "vectors.json"
        # chunk_id -> {"chunk": <dump>, "vector": [...]}
        self._records: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._file.exists():
            return json.loads(self._file.read_text(encoding="utf-8"))
        return {}

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps(self._records, ensure_ascii=False), encoding="utf-8"
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vec in zip(chunks, vectors):
            self._records[chunk.chunk_id] = {
                "chunk": chunk.model_dump(mode="json"),
                "vector": vec,
            }
        self._save()

    def search(
        self, vector: list[float], filters: Filter | None, top_n: int
    ) -> list[Candidate]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for rec in self._records.values():
            if not _passes(rec["chunk"], filters):
                continue
            scored.append((_cosine(vector, rec["vector"]), rec["chunk"]))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            Candidate(
                chunk=Chunk(**meta), score=score, source=CandidateSource.DENSE
            )
            for score, meta in scored[:top_n]
        ]


registry.register("vector_store", "memory", MemoryVectorStore)
