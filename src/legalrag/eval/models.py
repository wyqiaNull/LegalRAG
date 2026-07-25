"""Evaluation-only data contracts.

These models intentionally live outside ``core`` so the stable runtime contracts do not
change merely to support experiments.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..core.models import Citation, Identity, Route

CaseKind = Literal["normal", "refusal", "multiturn", "red_team", "benchmark"]
ReviewStatus = Literal["draft", "approved"]


class GoldCitation(BaseModel):
    doc_name: str
    version: str = ""
    clause_no: str

    @field_validator("doc_name", "clause_no")
    @classmethod
    def non_empty_location(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("gold 引用的文档名和条款号不能为空")
        return normalized

    def key(self) -> tuple[str, str, str]:
        return (self.doc_name, self.version, self.clause_no)


class EvalTurn(BaseModel):
    question: str
    expected_answer: str = ""
    expected_refusal: bool = False
    expected_citations: list[GoldCitation] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def non_empty_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("评测问题不能为空")
        return normalized


class EvalCase(BaseModel):
    case_id: str
    source: str
    source_version: str
    kind: CaseKind
    turns: list[EvalTurn] = Field(min_length=1)
    identity: Identity = Field(
        default_factory=lambda: Identity(
            user_id="evaluation",
            role="legal_staff",
            tenant_id="evaluation",
        )
    )
    tags: list[str] = Field(default_factory=list)

    @field_validator("case_id", "source", "source_version")
    @classmethod
    def non_empty_metadata(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("case_id、source 和 source_version 不能为空")
        return normalized

    @model_validator(mode="after")
    def validate_expected_behavior(self) -> "EvalCase":
        if self.kind == "multiturn" and len(self.turns) < 2:
            raise ValueError("multiturn 样本必须至少包含两轮")
        if self.kind == "normal":
            if any(turn.expected_refusal for turn in self.turns):
                raise ValueError("normal 样本不能期望拒答")
            if any(not turn.expected_citations for turn in self.turns):
                raise ValueError("normal 样本必须标注至少一个必需条款")
        if self.kind in {"refusal", "red_team"} and any(
            not turn.expected_refusal for turn in self.turns
        ):
            raise ValueError(f"{self.kind} 样本的每一轮都必须期望拒答")
        return self


class ReviewManifest(BaseModel):
    dataset_sha256: str
    review_csv_sha256: str = ""
    status: ReviewStatus = "draft"
    reviewer: str = ""
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def approved_review_has_auditor(self) -> "ReviewManifest":
        if self.status == "approved" and (
            not self.reviewer.strip()
            or self.reviewed_at is None
            or not self.review_csv_sha256.strip()
        ):
            raise ValueError(
                "approved 审核清单必须包含 reviewer、reviewed_at 和 review_csv_sha256"
            )
        return self


class EvalContext(BaseModel):
    chunk_id: str
    doc_name: str
    version: str = ""
    clause_no: str = ""
    content: str
    score: float = 0.0

    def key(self) -> tuple[str, str, str]:
        return (self.doc_name, self.version, self.clause_no)

    def matches(self, gold: GoldCitation) -> bool:
        if self.doc_name != gold.doc_name or self.version != gold.version:
            return False
        return self.clause_no == gold.clause_no or gold.clause_no in self.content


class EvalRetrievalSignal(BaseModel):
    question: str
    candidate_count: int = Field(ge=0)
    top_score: float | None = None


class EvalObservation(BaseModel):
    case_id: str
    turn_index: int = Field(ge=0)
    profile: str
    question: str
    retrieval_question: str
    retrieval_attempts: list[str] = Field(default_factory=list)
    retrieval_signals: list[EvalRetrievalSignal] = Field(default_factory=list)
    answer: str
    refused: bool
    reason: str | None = None
    route: Route
    contexts: list[EvalContext] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)
    error: str | None = None


class RefusalMetrics(BaseModel):
    accuracy: float
    recall: float
    false_refusal_rate: float
    macro_f1: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


class DeterministicMetrics(BaseModel):
    sample_count: int
    context_recall: float
    citation_precision: float
    citation_recall: float
    citation_f1: float
    refusal: RefusalMetrics
    red_team_attack_success_rate: float
    failed_observations: int
