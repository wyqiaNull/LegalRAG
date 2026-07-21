"""MetadataStore：保存 chunk 元数据、提供 ACL 策略（SPEC §4.2）。

MVP 内存实现 + JSON 落盘；ACL 为占位（返回 None），v0.2 接 PostgresStore 与 acl_policies。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import registry
from ..core.interfaces import MetadataStore
from ..core.models import Chunk


class MemoryMetadataStore(MetadataStore):
    def __init__(self, path: str = "data/processed") -> None:
        self._file = Path(path) / "metadata.json"
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

    def save(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._records[chunk.chunk_id] = chunk.model_dump(mode="json")
        self._save()

    def get_acl(self, role: str) -> Any:
        # 占位：MVP 全放行；v0.2 由 acl_policies 返回角色可见密级/类型策略。
        return None


registry.register("metadata_store", "memory", MemoryMetadataStore)
