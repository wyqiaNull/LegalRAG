"""VectorStore：标量过滤 + ANN 一体。

内存实现与外部向量库保持相同的 upsert/search 形态，并通过 JSON 支持跨进程查询。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..core import registry
from ..core.interfaces import Candidate, Filter, VectorStore
from ..core.models import CandidateSource, Chunk
from .bm25 import bm25_scores


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _matches_filter_value(got: Any, want: Any) -> bool:
    if not isinstance(want, list):
        return got == want
    if not want:
        return False
    if isinstance(got, list):
        return any(value in want for value in got)
    return got in want


def _passes(chunk_meta: dict[str, Any], filters: Filter | None) -> bool:
    """空过滤全通过；普通字段取交集，``$or`` 分支至少通过一个。"""
    if not filters:
        return True
    for key, want in filters.items():
        if not isinstance(key, str):
            return False
        if key == "$or":
            if (
                not isinstance(want, list)
                or not want
                or any(not isinstance(branch, dict) or not branch for branch in want)
            ):
                return False
            if not any(_passes(chunk_meta, branch) for branch in want):
                return False
            continue
        if key.startswith("$"):
            return False
        if not _matches_filter_value(chunk_meta.get(key), want):
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
        if len(chunks) != len(vectors):
            raise ValueError("chunk 与向量数量必须一致")
        for chunk, vec in zip(chunks, vectors):
            self._records[chunk.chunk_id] = {
                "chunk": chunk.model_dump(mode="json"),
                "vector": vec,
            }
        self._save()

    def get_vectors(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        """按 chunk_id 读取已有向量，供版本升级复用。"""
        return {
            chunk_id: list(self._records[chunk_id]["vector"])
            for chunk_id in chunk_ids
            if chunk_id in self._records
        }

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

    def search_sparse(
        self, query: str, filters: Filter | None, top_n: int
    ) -> list[Candidate]:
        """在过滤后的语料上计算 BM25 排名。"""
        records = [
            rec for rec in self._records.values() if _passes(rec["chunk"], filters)
        ]
        scores = bm25_scores(query, [rec["chunk"]["content"] for rec in records])
        ranked = [
            (score, rec["chunk"])
            for score, rec in zip(scores, records)
            if score > 0.0
        ]
        ranked.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        return [
            Candidate(
                chunk=Chunk(**meta), score=score, source=CandidateSource.SPARSE
            )
            for score, meta in ranked[:top_n]
        ]


registry.register("vector_store", "memory", MemoryVectorStore)
