"""MetadataStore 的 JSON/内存实现，供离线链路与测试使用。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import registry
from ..core.errors import StorageError
from ..core.interfaces import MetadataStore
from ..core.models import Chunk


class MemoryMetadataStore(MetadataStore):
    def __init__(
        self,
        path: str = "data/processed",
        acl_policies: list[dict[str, Any]] | None = None,
    ) -> None:
        self._file = Path(path) / "metadata.json"
        self._records: dict[str, dict[str, Any]] = self._load()
        self._acl_policies = {
            policy["role"]: policy for policy in (acl_policies or [])
        }

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._file.exists():
            return json.loads(self._file.read_text(encoding="utf-8"))
        return {}

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps(self._records, ensure_ascii=False), encoding="utf-8"
        )

    def save(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._records[chunk.chunk_id] = chunk.model_dump(mode="json")
        self._save()

    def get_acl(self, role: str) -> Any:
        return self._acl_policies.get(role)

    def list_current_chunks(
        self,
        doc_name: str,
        tenant_id: str,
        doc_type: str,
    ) -> list[Chunk]:
        return [
            Chunk(**record)
            for record in self._records.values()
            if record["doc_name"] == doc_name
            and record["tenant_id"] == tenant_id
            and record["doc_type"] == doc_type
            and record["is_current"]
        ]

    def list_version_chunks(
        self,
        doc_name: str,
        tenant_id: str,
        doc_type: str,
        version: str,
    ) -> list[Chunk]:
        return [
            Chunk(**record)
            for record in self._records.values()
            if record["doc_name"] == doc_name
            and record["tenant_id"] == tenant_id
            and record["doc_type"] == doc_type
            and record["version"] == version
        ]

    def replace_current(
        self, chunks: list[Chunk], expected_current_doc_id: str | None
    ) -> None:
        if not chunks:
            return
        first = chunks[0]
        current = self.list_current_chunks(
            first.doc_name, first.tenant_id, first.doc_type.value
        )
        current_ids = {chunk.doc_id for chunk in current}
        actual_current = next(iter(current_ids), None) if len(current_ids) <= 1 else None
        if len(current_ids) > 1 or actual_current != expected_current_doc_id:
            raise StorageError("文档现行版本已发生变化，请重试摄取")
        for chunk in current:
            self._records[chunk.chunk_id] = chunk.model_copy(
                update={"is_current": False, "superseded_by": first.doc_id}
            ).model_dump(mode="json")
        for chunk in chunks:
            self._records[chunk.chunk_id] = chunk.model_dump(mode="json")
        self._save()


registry.register("metadata_store", "memory", MemoryMetadataStore)
