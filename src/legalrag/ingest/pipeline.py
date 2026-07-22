"""摄取编排：load → chunk → meta → embed → store。

组件由调用方（cli 组装根）经 registry 注入，pipeline 只负责串流程、不 import 具体实现。
"""

from __future__ import annotations

from typing import Any

from ..core.errors import IngestError
from ..core.interfaces import Chunker, Embedder, Loader, MetadataStore, VectorStore
from ..core.models import ROLE_WILDCARD, Chunk, Confidentiality, DocType
from . import metadata
from .versioning import IngestResult, IngestStats, VersionManager


class IngestPipeline:
    def __init__(
        self,
        loader: Loader,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        versions_enabled: bool = False,
        shared_tenant_id: str = "__global__",
        shared_doc_types: list[DocType] | None = None,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.versions_enabled = versions_enabled
        self.shared_tenant_id = shared_tenant_id
        self.shared_doc_types = set(
            [DocType.REGULATION] if shared_doc_types is None else shared_doc_types
        )

    def _validate_shared_document(self, chunks: list[Chunk]) -> None:
        shared_chunks = [
            chunk for chunk in chunks if chunk.tenant_id == self.shared_tenant_id
        ]
        if not shared_chunks:
            return
        if len(shared_chunks) != len(chunks):
            raise IngestError("同一文档的 chunks 不能混用全局与私有租户")
        for chunk in shared_chunks:
            if chunk.confidentiality != Confidentiality.PUBLIC:
                raise IngestError("全局共享文档的密级必须为 public")
            if chunk.doc_type not in self.shared_doc_types:
                raise IngestError(
                    f"文档类型 {chunk.doc_type.value!r} 不在全局共享类型配置中"
                )
            if chunk.allowed_roles != [ROLE_WILDCARD]:
                raise IngestError(
                    "全局共享文档必须对所有角色可见，allowed_roles 只能为 ['*']"
                )

    def run(self, path: str, **doc_meta: Any) -> list[Chunk]:
        """摄取单个文件，返回落库的 chunk 列表。

        doc_meta 可覆盖文档级元数据（doc_type/version/confidentiality/tenant_id 等）。
        """
        return self.run_detailed(path, **doc_meta).chunks

    def run_detailed(self, path: str, **doc_meta: Any) -> IngestResult:
        """摄取单个文件并返回增量处理统计。"""
        raw = self.loader.load(path)
        meta_values = dict(doc_meta)
        doc_name = meta_values.pop("doc_name", raw.doc_name)
        base = metadata.base_meta(doc_name, **meta_values)
        chunks = self.chunker.split(raw, base)
        if not chunks:
            return IngestResult([], IngestStats(0, 0, 0, 0))
        self._validate_shared_document(chunks)
        if self.versions_enabled:
            return VersionManager(
                self.embedder, self.vector_store, self.metadata_store
            ).ingest(chunks)
        vectors = self.embedder.embed([c.content for c in chunks]).dense
        self.vector_store.upsert(chunks, vectors)
        self.metadata_store.save(chunks)
        return IngestResult(
            chunks,
            IngestStats(len(chunks), len(chunks), 0, 0),
        )
