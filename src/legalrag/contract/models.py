"""v0.4 合同审查专用数据模型，不扩张稳定的 core 契约。"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..core.models import Citation


class ContractType(str, Enum):
    LABOR = "labor"


class ClauseCategory(str, Enum):
    CONTRACT_TERM = "contract_term"
    PROBATION = "probation"
    WORKING_TIME = "working_time"
    COMPENSATION = "compensation"
    SOCIAL_INSURANCE = "social_insurance"
    CONFIDENTIALITY = "confidentiality"
    NON_COMPETE = "non_compete"
    TERMINATION = "termination"
    LIABILITY = "liability"
    DISPUTE_RESOLUTION = "dispute_resolution"
    OTHER = "other"


class ClauseCompliance(str, Enum):
    COMPLIANT = "compliant"
    RISK = "risk"
    NO_MATCH = "no_match"


class ContractClause(BaseModel):
    clause_no: str
    content: str
    page: int | None = None

    @field_validator("clause_no", "content")
    @classmethod
    def non_empty_clause(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("合同条款编号和正文不能为空")
        return normalized


class ClauseReview(BaseModel):
    clause_no: str
    content: str
    page: int | None = None
    category: ClauseCategory
    retrieval_query: str
    status: ClauseCompliance
    reason: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("clause_no", "content", "retrieval_query", "reason")
    @classmethod
    def non_empty_review_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("条款审查文本字段不能为空")
        return normalized

    @model_validator(mode="after")
    def evidence_matches_status(self) -> "ClauseReview":
        if self.status is ClauseCompliance.NO_MATCH:
            if self.citations or self.retrieved_chunk_ids:
                raise ValueError("no_match 条款不能携带引用或命中 chunk")
        elif not self.citations or not self.retrieved_chunk_ids:
            raise ValueError("合规或风险条款必须携带引用和命中 chunk")
        return self


class ContractReviewSummary(BaseModel):
    total: int = Field(ge=0)
    compliant: int = Field(ge=0)
    risk: int = Field(ge=0)
    no_match: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_match_total(self) -> "ContractReviewSummary":
        if self.compliant + self.risk + self.no_match != self.total:
            raise ValueError("合同审查汇总数量不一致")
        return self


class ContractReviewReport(BaseModel):
    contract_name: str
    contract_type: ContractType
    summary: ContractReviewSummary
    clauses: list[ClauseReview] = Field(min_length=1)

    @field_validator("contract_name")
    @classmethod
    def non_empty_contract_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("合同名称不能为空")
        return normalized

    @model_validator(mode="after")
    def summary_matches_clauses(self) -> "ContractReviewReport":
        numbers = [clause.clause_no for clause in self.clauses]
        if len(numbers) != len(set(numbers)):
            raise ValueError("合同审查报告条款编号重复")
        expected = {
            "total": len(self.clauses),
            "compliant": sum(
                clause.status is ClauseCompliance.COMPLIANT for clause in self.clauses
            ),
            "risk": sum(clause.status is ClauseCompliance.RISK for clause in self.clauses),
            "no_match": sum(
                clause.status is ClauseCompliance.NO_MATCH for clause in self.clauses
            ),
        }
        if self.summary.model_dump() != expected:
            raise ValueError("合同审查汇总与逐条款结论不一致")
        return self


class GoldCitation(BaseModel):
    doc_name: str
    version: str = ""
    clause_no: str

    @field_validator("doc_name", "clause_no")
    @classmethod
    def non_empty_location(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("gold 引用文档名和条款号不能为空")
        return normalized

    def key(self) -> tuple[str, str, str]:
        return (self.doc_name, self.version, self.clause_no)


class ContractGoldClause(BaseModel):
    clause_no: str
    expected_status: ClauseCompliance
    expected_citations: list[GoldCitation] = Field(default_factory=list)

    @field_validator("clause_no")
    @classmethod
    def non_empty_clause_no(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("gold 条款编号不能为空")
        return normalized


class ContractGold(BaseModel):
    contract_name: str
    contract_type: ContractType
    source: str
    review_status: Literal["draft", "approved"] = "draft"
    clauses: list[ContractGoldClause] = Field(min_length=1)

    @field_validator("contract_name", "source")
    @classmethod
    def non_empty_gold_metadata(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("合同 gold 名称和来源不能为空")
        return normalized

    @model_validator(mode="after")
    def unique_clause_numbers(self) -> "ContractGold":
        numbers = [clause.clause_no for clause in self.clauses]
        if len(numbers) != len(set(numbers)):
            raise ValueError("合同 gold 条款编号重复")
        return self


class ContractReviewMetrics(BaseModel):
    review_status: Literal["draft", "approved"]
    gold_clause_count: int = Field(ge=0)
    matched_clause_count: int = Field(ge=0)
    missing_clause_count: int = Field(ge=0)
    unexpected_clause_count: int = Field(ge=0)
    status_accuracy: float = Field(ge=0.0, le=1.0)
    citation_precision: float = Field(ge=0.0, le=1.0)
    citation_recall: float = Field(ge=0.0, le=1.0)
