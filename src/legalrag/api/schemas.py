"""REST API 请求与响应模型。"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field

from ..contract import ContractReviewReport
from ..core.models import Citation, Route


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = Field(default=None, max_length=256)
    version: str | None = Field(default=None, max_length=128)


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[Citation]
    refused: bool
    reason: str | None
    route: Route
    retrieved_chunk_ids: list[str]


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IngestAccepted(BaseModel):
    task_id: str
    status: TaskState = TaskState.PENDING


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskState
    result: dict[str, int] | None = None
    error_code: str | None = None
    failure_reason: str | None = None


class FeedbackValue(str, Enum):
    HELPFUL = "helpful"
    UNHELPFUL = "unhelpful"


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    value: FeedbackValue
    comment: str | None = Field(default=None, max_length=2_000)


class FeedbackResponse(BaseModel):
    feedback_id: str


class DocumentRecord(BaseModel):
    doc_id: str
    doc_name: str
    doc_type: str
    tenant_id: str
    version: str
    is_current: bool
    effective_date: str | None = None
    confidentiality: str
    created_at: datetime
    updated_at: datetime


class DocumentPage(BaseModel):
    items: list[DocumentRecord]
    page: int
    page_size: int


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    deleted_chunk_count: int


class AuditRecord(BaseModel):
    request_id: str
    user_id: str
    role: str
    tenant_id: str
    query_sha256: str
    route: str | None = None
    refused: bool | None = None
    reason: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: float
    status: str
    error_code: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None
    usage_complete: bool = False


class ContractReviewResponse(ContractReviewReport):
    pass
