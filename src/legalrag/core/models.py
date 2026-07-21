"""领域模型（SPEC §4.1 / §5，字段对齐 tech-design §2.1）。

字段一次建全 —— MVP 用不到的权限/版本/引用字段也先带上并给安全默认值，
避免后期回填重刷索引（哲学六）。后续版本只加实现、不改这些字段。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Confidentiality(str, Enum):
    """密级（tech-design §2.1）。三级密级正是权限隔离的载体。"""

    PUBLIC = "公开"
    INTERNAL = "内部"
    SECRET = "机密"


class DocType(str, Enum):
    """三级合规文档类型（知识库只存这三类，合同不入库）。"""

    REGULATION = "法规"
    POLICY = "制度"
    LEGAL_OPINION = "法律意见"


class Route(str, Enum):
    """路由决策（SPEC §4.1）。MVP 只用 NORMAL，枚举先定全供 v0.3 使用。"""

    CHITCHAT = "chitchat"
    NORMAL = "normal"
    OUT_OF_SCOPE = "out_of_scope"
    NEED_CLARIFY = "need_clarify"
    CONTRACT_REVIEW = "contract_review"


class CandidateSource(str, Enum):
    """候选来源，保留以便消融分析（SPEC §4.1）。"""

    DENSE = "dense"
    SPARSE = "sparse"
    FUSED = "fused"
    RERANKED = "reranked"


# 通配角色：MVP 默认所有 chunk 对所有角色可见（v0.2 起收紧）。
ROLE_WILDCARD = "*"


class Identity(BaseModel):
    """请求者身份（SPEC §4.1）。MVP 默认给最高权限，字段先就位供 v0.2 启用。"""

    user_id: str = "anonymous"
    role: str = ROLE_WILDCARD
    tenant_id: str = "default"
    allowed_confidentiality: list[Confidentiality] = Field(
        default_factory=lambda: list(Confidentiality)
    )


class Query(BaseModel):
    """一次查询请求（SPEC §4.1）。"""

    text: str
    session_id: str | None = None
    identity: Identity = Field(default_factory=Identity)
    top_k: int = 5


class Chunk(BaseModel):
    """知识块 —— 携带 §5 全部元数据字段 + 正文。

    权限/版本字段 MVP 用合理默认值填（confidentiality=公开, is_current=true,
    allowed_roles=[*]），结构就位，v0.2 直接启用过滤逻辑，无需重建索引。
    """

    # —— 标识与溯源 ——
    chunk_id: str
    doc_id: str
    doc_name: str
    doc_type: DocType = DocType.REGULATION
    # —— 权限（FR-5，v0.2 启用过滤）——
    tenant_id: str = "default"
    department: str = ""
    allowed_roles: list[str] = Field(default_factory=lambda: [ROLE_WILDCARD])
    confidentiality: Confidentiality = Confidentiality.PUBLIC
    # —— 版本（FR-6，v0.2 启用）——
    version: str = ""
    effective_date: str | None = None  # ISO date 字符串，None 表示未知
    is_current: bool = True
    superseded_by: str | None = None
    # —— 引用定位（FR-4）——
    clause_no: str = ""
    page: int | None = None
    # —— 正文 ——
    content: str
    content_hash: str = ""


class Candidate(BaseModel):
    """检索候选：一个 chunk + 其得分 + 来源（SPEC §4.1）。"""

    chunk: Chunk
    score: float = 0.0
    source: CandidateSource = CandidateSource.DENSE


class Citation(BaseModel):
    """引用溯源信息（SPEC §4.1，tech-design §5 响应体）。"""

    doc_name: str
    version: str = ""
    clause_no: str = ""
    page: int | None = None


class Answer(BaseModel):
    """一次查询的最终回答（结构对齐 tech-design §5 响应体）。"""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    reason: str | None = None  # 拒答原因（如 out_of_scope），refused=True 时填
    route: Route = Route.NORMAL
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
