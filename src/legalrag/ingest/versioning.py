"""文档版本升级与 chunk 级向量复用。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from ..core.errors import IngestError, StorageError
from ..core.interfaces import Embedder, MetadataStore, VectorStore
from ..core.models import Chunk


@dataclass(frozen=True)
class IngestStats:
    total_chunks: int
    embedded_chunks: int
    reused_vectors: int
    superseded_chunks: int


@dataclass(frozen=True)
class IngestResult:
    chunks: list[Chunk]
    stats: IngestStats


def _require_method(component: object, name: str) -> Callable[..., Any]:
    method = getattr(component, name, None)
    if not callable(method):
        raise StorageError(f"启用版本管理时，{type(component).__name__} 必须实现 {name}()")
    return cast(Callable[..., Any], method)


def _version_signature(chunk: Chunk) -> tuple[Any, ...]:
    """同版本重试必须完全一致，避免原地篡改历史事实。"""
    return (
        chunk.chunk_id,
        chunk.content_hash,
        chunk.clause_no,
        chunk.page,
        chunk.department,
        tuple(chunk.allowed_roles),
        chunk.confidentiality,
        chunk.effective_date,
    )


class VersionManager:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    def ingest(self, chunks: list[Chunk]) -> IngestResult:
        if not chunks:
            return IngestResult([], IngestStats(0, 0, 0, 0))

        first = chunks[0]
        if not first.doc_name.strip():
            raise IngestError("启用版本管理时必须提供非空文档名")
        if not first.version.strip():
            raise IngestError("启用版本管理时必须提供非空版本号")

        family = (first.doc_name, first.tenant_id, first.doc_type.value)
        list_current = _require_method(self.metadata_store, "list_current_chunks")
        list_version = _require_method(self.metadata_store, "list_version_chunks")
        replace_current = _require_method(self.metadata_store, "replace_current")
        get_vectors = _require_method(self.vector_store, "get_vectors")

        existing = list_version(*family, first.version)
        if existing:
            if sorted(map(_version_signature, existing)) != sorted(
                map(_version_signature, chunks)
            ):
                raise IngestError(
                    f"文档 {first.doc_name!r} 的版本 {first.version!r} 已存在且内容不同，"
                    "请使用新的版本号"
                )
            return IngestResult(
                existing,
                IngestStats(len(existing), 0, len(existing), 0),
            )

        current = list_current(*family)
        current_doc_ids = {chunk.doc_id for chunk in current}
        if len(current_doc_ids) > 1:
            raise StorageError(f"文档族 {family!r} 存在多个现行版本")
        expected_current = next(iter(current_doc_ids), None)

        old_vectors: dict[str, list[float]] = get_vectors(
            [chunk.chunk_id for chunk in current]
        )
        reusable_by_hash: dict[str, list[float]] = {}
        for chunk in current:
            vector = old_vectors.get(chunk.chunk_id)
            if vector is not None:
                reusable_by_hash.setdefault(chunk.content_hash, vector)

        vectors: list[list[float] | None] = []
        embed_indexes: list[int] = []
        for index, chunk in enumerate(chunks):
            vector = reusable_by_hash.get(chunk.content_hash)
            vectors.append(list(vector) if vector is not None else None)
            if vector is None:
                embed_indexes.append(index)

        if embed_indexes:
            embedded = self.embedder.embed([chunks[index].content for index in embed_indexes])
            if len(embedded.dense) != len(embed_indexes):
                raise IngestError("embedding 返回的向量数量与待处理 chunk 数不一致")
            for index, vector in zip(embed_indexes, embedded.dense, strict=True):
                vectors[index] = vector

        complete_vectors = [cast(list[float], vector) for vector in vectors]
        superseded = [
            chunk.model_copy(
                update={"is_current": False, "superseded_by": first.doc_id}
            )
            for chunk in current
        ]
        superseded_with_vectors = [
            chunk for chunk in superseded if chunk.chunk_id in old_vectors
        ]
        self.vector_store.upsert(
            [*superseded_with_vectors, *chunks],
            [
                *[old_vectors[chunk.chunk_id] for chunk in superseded_with_vectors],
                *complete_vectors,
            ],
        )
        replace_current(chunks, expected_current)

        return IngestResult(
            chunks,
            IngestStats(
                total_chunks=len(chunks),
                embedded_chunks=len(embed_indexes),
                reused_vectors=len(chunks) - len(embed_indexes),
                superseded_chunks=len(current),
            ),
        )
