"""摄取编排（SPEC §6，FR-1 MVP）：load → chunk → meta → embed → store。

组件由调用方（cli 组装根）经 registry 注入，pipeline 只负责串流程、不 import 具体实现。
"""

from __future__ import annotations

from typing import Any

from ..core.interfaces import Chunker, Embedder, Loader, MetadataStore, VectorStore
from ..core.models import Chunk
from . import metadata


class IngestPipeline:
    def __init__(
        self,
        loader: Loader,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    def run(self, path: str, **doc_meta: Any) -> list[Chunk]:
        """摄取单个文件，返回落库的 chunk 列表。

        doc_meta 可覆盖文档级元数据（doc_type/version/confidentiality/tenant_id 等）。
        """
        raw = self.loader.load(path)
        base = metadata.base_meta(raw.doc_name, **doc_meta)
        chunks = self.chunker.split(raw, base)
        if not chunks:
            return []
        vectors = self.embedder.embed([c.content for c in chunks]).dense
        self.vector_store.upsert(chunks, vectors)
        self.metadata_store.save(chunks)
        return chunks
