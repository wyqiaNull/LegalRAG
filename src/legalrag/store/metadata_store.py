"""MetadataStore 的 JSON/内存实现，供离线链路与测试使用。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import registry
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


registry.register("metadata_store", "memory", MemoryMetadataStore)
