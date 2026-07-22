"""PostgreSQL 元数据存储。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.rows import dict_row

from ..core import registry
from ..core.errors import ConfigError, StorageError
from ..core.interfaces import MetadataStore
from ..core.models import Chunk

ConnectionFactory = Callable[..., Any]

_DOCUMENTS_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    doc_name TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    department TEXT NOT NULL DEFAULT '',
    allowed_roles TEXT[] NOT NULL,
    confidentiality TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '',
    effective_date DATE,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_CHUNKS_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    doc_name TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    department TEXT NOT NULL DEFAULT '',
    allowed_roles TEXT[] NOT NULL,
    confidentiality TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '',
    effective_date DATE,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_by TEXT,
    clause_no TEXT NOT NULL DEFAULT '',
    page INTEGER,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_ACL_DDL = """
CREATE TABLE IF NOT EXISTS acl_policies (
    role TEXT PRIMARY KEY,
    allowed_confidentiality TEXT[] NOT NULL,
    allowed_doc_types TEXT[] NOT NULL,
    tenant_scope TEXT NOT NULL DEFAULT 'own',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_UPSERT_DOCUMENT = """
INSERT INTO documents (
    doc_id, doc_name, doc_type, tenant_id, department, allowed_roles,
    confidentiality, version, effective_date, is_current, superseded_by
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (doc_id) DO UPDATE SET
    doc_name = EXCLUDED.doc_name,
    doc_type = EXCLUDED.doc_type,
    tenant_id = EXCLUDED.tenant_id,
    department = EXCLUDED.department,
    allowed_roles = EXCLUDED.allowed_roles,
    confidentiality = EXCLUDED.confidentiality,
    version = EXCLUDED.version,
    effective_date = EXCLUDED.effective_date,
    is_current = EXCLUDED.is_current,
    superseded_by = EXCLUDED.superseded_by,
    updated_at = CURRENT_TIMESTAMP
"""

_UPSERT_CHUNK = """
INSERT INTO chunks (
    chunk_id, doc_id, doc_name, doc_type, tenant_id, department, allowed_roles,
    confidentiality, version, effective_date, is_current, superseded_by,
    clause_no, page, content, content_hash
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id = EXCLUDED.doc_id,
    doc_name = EXCLUDED.doc_name,
    doc_type = EXCLUDED.doc_type,
    tenant_id = EXCLUDED.tenant_id,
    department = EXCLUDED.department,
    allowed_roles = EXCLUDED.allowed_roles,
    confidentiality = EXCLUDED.confidentiality,
    version = EXCLUDED.version,
    effective_date = EXCLUDED.effective_date,
    is_current = EXCLUDED.is_current,
    superseded_by = EXCLUDED.superseded_by,
    clause_no = EXCLUDED.clause_no,
    page = EXCLUDED.page,
    content = EXCLUDED.content,
    content_hash = EXCLUDED.content_hash,
    updated_at = CURRENT_TIMESTAMP
"""

_UPSERT_ACL = """
INSERT INTO acl_policies (
    role, allowed_confidentiality, allowed_doc_types, tenant_scope
) VALUES (%s, %s, %s, %s)
ON CONFLICT (role) DO UPDATE SET
    allowed_confidentiality = EXCLUDED.allowed_confidentiality,
    allowed_doc_types = EXCLUDED.allowed_doc_types,
    tenant_scope = EXCLUDED.tenant_scope,
    updated_at = CURRENT_TIMESTAMP
"""

_CHUNK_FIELDS = """
chunk_id, doc_id, doc_name, doc_type, tenant_id, department, allowed_roles,
confidentiality, version, effective_date, is_current, superseded_by,
clause_no, page, content, content_hash
"""


def _document_values(chunk: Chunk) -> tuple[Any, ...]:
    return (
        chunk.doc_id,
        chunk.doc_name,
        chunk.doc_type.value,
        chunk.tenant_id,
        chunk.department,
        chunk.allowed_roles,
        chunk.confidentiality.value,
        chunk.version,
        chunk.effective_date,
        chunk.is_current,
        chunk.superseded_by,
    )


def _chunk_values(chunk: Chunk) -> tuple[Any, ...]:
    return (
        chunk.chunk_id,
        *_document_values(chunk),
        chunk.clause_no,
        chunk.page,
        chunk.content,
        chunk.content_hash,
    )


def _chunk_from_row(row: dict[str, Any]) -> Chunk:
    values = dict(row)
    if values["effective_date"] is not None:
        values["effective_date"] = values["effective_date"].isoformat()
    return Chunk(**values)


class PostgresMetadataStore(MetadataStore):
    def __init__(
        self,
        dsn: str,
        acl_policies: list[dict[str, Any]] | None = None,
        connection_factory: ConnectionFactory = psycopg.connect,
    ) -> None:
        if not dsn:
            raise ConfigError("使用 PostgreSQL 元数据存储时必须配置 POSTGRES_DSN")
        self.dsn = dsn
        self._connect = connection_factory
        self._initialize(acl_policies or [])

    def _connection(self):
        return self._connect(self.dsn, row_factory=dict_row)

    def _initialize(self, acl_policies: list[dict[str, Any]]) -> None:
        try:
            with self._connection() as conn, conn.cursor() as cur:
                cur.execute(_DOCUMENTS_DDL)
                cur.execute(_CHUNKS_DDL)
                cur.execute(_ACL_DDL)
                if acl_policies:
                    cur.executemany(
                        _UPSERT_ACL,
                        [
                            (
                                policy["role"],
                                policy["allowed_confidentiality"],
                                policy["allowed_doc_types"],
                                policy.get("tenant_scope", "own"),
                            )
                            for policy in acl_policies
                        ],
                    )
        except psycopg.Error as exc:
            raise StorageError(f"初始化 PostgreSQL 元数据存储失败：{exc}") from exc

    def save(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        documents = {chunk.doc_id: chunk for chunk in chunks}
        try:
            with self._connection() as conn, conn.cursor() as cur:
                cur.executemany(
                    _UPSERT_DOCUMENT,
                    [_document_values(chunk) for chunk in documents.values()],
                )
                cur.executemany(_UPSERT_CHUNK, [_chunk_values(chunk) for chunk in chunks])
        except psycopg.Error as exc:
            raise StorageError(f"保存 PostgreSQL 元数据失败：{exc}") from exc

    def get_acl(self, role: str) -> dict[str, Any] | None:
        try:
            with self._connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, allowed_confidentiality, allowed_doc_types, tenant_scope
                    FROM acl_policies WHERE role = %s
                    """,
                    (role,),
                )
                return cur.fetchone()
        except psycopg.Error as exc:
            raise StorageError(f"查询 PostgreSQL ACL 失败：{exc}") from exc

    def list_current_chunks(
        self,
        doc_name: str,
        tenant_id: str,
        doc_type: str,
    ) -> list[Chunk]:
        try:
            with self._connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {_CHUNK_FIELDS}
                    FROM chunks
                    WHERE doc_name = %s AND tenant_id = %s
                      AND doc_type = %s AND is_current = TRUE
                    ORDER BY chunk_id
                    """,
                    (doc_name, tenant_id, doc_type),
                )
                return [_chunk_from_row(row) for row in cur.fetchall()]
        except psycopg.Error as exc:
            raise StorageError(f"查询 PostgreSQL 现行文档失败：{exc}") from exc


registry.register("metadata_store", "postgres", PostgresMetadataStore)
