"""元数据 schema 组装。

摄取时一次建全 —— 禁止事后补字段。权限/版本字段 MVP 用安全默认值，
结构就位，v0.2 直接启用过滤逻辑，无需重建索引。
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from ..core.models import ROLE_WILDCARD, Chunk, Confidentiality, DocType

# uuid5 命名空间，保证同名文档得到稳定 doc_id（re-ingest 幂等）。
_DOC_NS = uuid.UUID("00000000-0000-0000-0000-00000000face")


def content_hash(text: str) -> str:
    """正文 sha256（增量更新用；MVP 就落库，零成本预留）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_id_for(
    doc_name: str,
    version: str = "",
    tenant_id: str = "default",
    doc_type: DocType = DocType.REGULATION,
) -> str:
    key = f"{doc_name}@{version}"
    if tenant_id != "default" or doc_type != DocType.REGULATION:
        key = f"{tenant_id}:{doc_type.value}:{key}"
    return str(uuid.uuid5(_DOC_NS, key))


def base_meta(
    doc_name: str,
    *,
    doc_type: DocType = DocType.REGULATION,
    version: str = "",
    tenant_id: str = "default",
    department: str = "",
    allowed_roles: list[str] | None = None,
    confidentiality: Confidentiality = Confidentiality.PUBLIC,
    effective_date: str | None = None,
) -> dict[str, Any]:
    """文档级公共元数据（chunk 级字段由 assemble_chunk 补齐）。"""
    return {
        "doc_id": doc_id_for(doc_name, version, tenant_id, doc_type),
        "doc_name": doc_name,
        "doc_type": doc_type,
        "version": version,
        "tenant_id": tenant_id,
        "department": department,
        "allowed_roles": [ROLE_WILDCARD] if allowed_roles is None else allowed_roles,
        "confidentiality": confidentiality,
        "effective_date": effective_date,
    }


def assemble_chunk(
    content: str,
    base: dict[str, Any],
    *,
    seq: int,
    page: int | None = None,
    clause_no: str = "",
) -> Chunk:
    """把一段正文 + 文档级元数据组装为携带全字段的 Chunk。

    权限/版本未显式给出的字段走 Chunk 的模型默认值（allowed_roles=[*]、
    is_current=true 等），保证 schema 一次建全。
    """
    doc_id = base["doc_id"]
    return Chunk(
        chunk_id=f"{doc_id}:{seq:04d}",
        content=content,
        content_hash=content_hash(content),
        page=page,
        clause_no=clause_no,
        **{k: v for k, v in base.items() if k != "doc_id"},
        doc_id=doc_id,
    )
