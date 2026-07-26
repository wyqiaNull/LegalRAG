"""服务层 PostgreSQL 持久化：审计、反馈与文档管理。"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.rows import dict_row

from ..core.errors import ConfigError, StorageError
from .auth import Principal
from .schemas import AuditRecord, DocumentRecord, FeedbackRequest

ConnectionFactory = Callable[..., Any]

_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS query_audits (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    query_sha256 CHAR(64) NOT NULL,
    route TEXT,
    refused BOOLEAN,
    reason TEXT,
    chunk_ids TEXT[] NOT NULL DEFAULT '{}',
    latency_ms DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL,
    error_code TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    estimated_cost_usd NUMERIC(18, 8),
    usage_complete BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
_FEEDBACK_DDL = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES query_audits(request_id),
    user_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    value TEXT NOT NULL CHECK (value IN ('helpful', 'unhelpful')),
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (request_id, user_id)
)
"""
_DOCUMENT_DELETED_DDL = (
    "ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ"
)
_CHUNK_DELETED_DDL = (
    "ALTER TABLE IF EXISTS chunks ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ"
)


class ServiceRepository:
    def __init__(
        self,
        dsn: str,
        connection_factory: ConnectionFactory = psycopg.connect,
    ) -> None:
        if not dsn:
            raise ConfigError("服务 API 必须配置 POSTGRES_DSN")
        self.dsn = dsn
        self._connect = connection_factory
        self._initialize()

    def _connection(self):
        return self._connect(self.dsn, row_factory=dict_row)

    def _initialize(self) -> None:
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute(_AUDIT_DDL)
                cursor.execute(_FEEDBACK_DDL)
                cursor.execute(_DOCUMENT_DELETED_DDL)
                cursor.execute(_CHUNK_DELETED_DDL)
        except psycopg.Error as exc:
            raise StorageError(f"初始化服务层 PostgreSQL 表失败：{exc}") from exc

    def save_audit(self, record: AuditRecord) -> None:
        values = record.model_dump(mode="python")
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO query_audits (
                        request_id, user_id, role, tenant_id, query_sha256, route,
                        refused, reason, chunk_ids, latency_ms, status, error_code,
                        prompt_tokens, completion_tokens, total_tokens,
                        estimated_cost_usd, usage_complete
                    ) VALUES (
                        %(request_id)s, %(user_id)s, %(role)s, %(tenant_id)s,
                        %(query_sha256)s, %(route)s, %(refused)s, %(reason)s,
                        %(chunk_ids)s, %(latency_ms)s, %(status)s, %(error_code)s,
                        %(prompt_tokens)s, %(completion_tokens)s, %(total_tokens)s,
                        %(estimated_cost_usd)s, %(usage_complete)s
                    )
                    """,
                    values,
                )
        except psycopg.Error as exc:
            raise StorageError(f"写入查询审计失败：{exc}") from exc

    def create_feedback(
        self,
        principal: Principal,
        request: FeedbackRequest,
    ) -> str:
        feedback_id = str(uuid.uuid4())
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO feedback (
                        feedback_id, request_id, user_id, tenant_id, value, comment
                    )
                    SELECT %s, request_id, %s, %s, %s, %s
                    FROM query_audits
                    WHERE request_id = %s AND user_id = %s AND tenant_id = %s
                    ON CONFLICT (request_id, user_id) DO UPDATE SET
                        value = EXCLUDED.value,
                        comment = EXCLUDED.comment,
                        created_at = CURRENT_TIMESTAMP
                    RETURNING feedback_id
                    """,
                    (
                        feedback_id,
                        principal.user_id,
                        principal.tenant_id,
                        request.value.value,
                        request.comment,
                        request.request_id,
                        principal.user_id,
                        principal.tenant_id,
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    raise LookupError("找不到属于当前用户的查询记录")
                return row["feedback_id"]
        except psycopg.Error as exc:
            raise StorageError(f"写入查询反馈失败：{exc}") from exc

    def list_documents(
        self,
        tenant_id: str,
        *,
        include_global: bool,
        page: int,
        page_size: int,
    ) -> list[DocumentRecord]:
        tenants = [tenant_id, "__global__"] if include_global else [tenant_id]
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT doc_id, doc_name, doc_type, tenant_id, version, is_current,
                           effective_date, confidentiality, created_at, updated_at
                    FROM documents
                    WHERE tenant_id = ANY(%s) AND deleted_at IS NULL
                    ORDER BY updated_at DESC, doc_id
                    LIMIT %s OFFSET %s
                    """,
                    (tenants, page_size, (page - 1) * page_size),
                )
                return [
                    DocumentRecord(
                        **{
                            **row,
                            "effective_date": (
                                row["effective_date"].isoformat()
                                if row["effective_date"] is not None
                                else None
                            ),
                        }
                    )
                    for row in cursor.fetchall()
                ]
        except psycopg.Error as exc:
            raise StorageError(f"查询文档列表失败：{exc}") from exc

    def document_chunk_ids(
        self,
        doc_id: str,
        tenant_id: str,
        *,
        allow_global: bool,
    ) -> list[str]:
        tenants = [tenant_id, "__global__"] if allow_global else [tenant_id]
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT c.chunk_id
                    FROM chunks c
                    JOIN documents d ON d.doc_id = c.doc_id
                    WHERE d.doc_id = %s AND d.tenant_id = ANY(%s)
                      AND d.deleted_at IS NULL AND c.deleted_at IS NULL
                    ORDER BY c.chunk_id
                    """,
                    (doc_id, tenants),
                )
                return [row["chunk_id"] for row in cursor.fetchall()]
        except psycopg.Error as exc:
            raise StorageError(f"查询待删除文档失败：{exc}") from exc

    def mark_document_deleted(self, doc_id: str, chunk_ids: list[str]) -> None:
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE documents
                    SET deleted_at = CURRENT_TIMESTAMP, is_current = FALSE,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE doc_id = %s AND deleted_at IS NULL
                    """,
                    (doc_id,),
                )
                cursor.execute(
                    """
                    UPDATE chunks
                    SET deleted_at = CURRENT_TIMESTAMP, is_current = FALSE,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE chunk_id = ANY(%s) AND deleted_at IS NULL
                    """,
                    (chunk_ids,),
                )
        except psycopg.Error as exc:
            raise StorageError(f"标记文档删除失败：{exc}") from exc

    def ping(self) -> bool:
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute("SELECT 1 AS ok")
                return cursor.fetchone()["ok"] == 1
        except psycopg.Error:
            return False
