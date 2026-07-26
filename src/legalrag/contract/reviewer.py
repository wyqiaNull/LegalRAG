"""劳动合同审查编排：拆条款、分类、反查法规、逐条判定和汇总。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ..core.errors import ContractReviewError
from ..core.interfaces import (
    Chunker,
    LLMClient,
    Loader,
    PermissionFilter,
    RawDoc,
    Reflector,
    Reranker,
    Retriever,
    VersionFilter,
)
from ..core.models import Candidate, DocType, Identity, Query
from ..generation.citation import assemble_citations, validate_citations
from .models import (
    ClauseCategory,
    ClauseCompliance,
    ClauseReview,
    ContractClause,
    ContractReviewReport,
    ContractReviewSummary,
    ContractType,
)

_PROMPT_DIR = Path(__file__).parent / "prompts"


class _StrictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _TypeDecision(_StrictResponse):
    contract_type: Literal["labor", "unsupported"]


class _ClauseRoute(_StrictResponse):
    category: ClauseCategory
    retrieval_query: str

    @field_validator("retrieval_query")
    @classmethod
    def non_empty_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("法规检索问题不能为空")
        return normalized


class _ClauseDecision(_StrictResponse):
    status: ClauseCompliance
    reason: str
    evidence_indices: list[int] = Field(default_factory=list)

    @field_validator("reason")
    @classmethod
    def non_empty_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("条款审查理由不能为空")
        return normalized

    @model_validator(mode="after")
    def evidence_matches_status(self) -> "_ClauseDecision":
        if len(self.evidence_indices) != len(set(self.evidence_indices)):
            raise ValueError("证据序号不能重复")
        if self.status is ClauseCompliance.NO_MATCH and self.evidence_indices:
            raise ValueError("no_match 不能携带证据序号")
        if self.status is not ClauseCompliance.NO_MATCH and not self.evidence_indices:
            raise ValueError("合规或风险结论必须携带证据序号")
        return self


def _parse_json_model(output: str, model: type[BaseModel], label: str) -> BaseModel:
    try:
        payload = json.loads(output.strip())
        if not isinstance(payload, dict):
            raise ValueError("顶层必须是 JSON 对象")
        return model.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ContractReviewError(f"{label}输出不是合法的严格 JSON：{exc}") from exc


def _format_context(candidates: list[Candidate]) -> str:
    blocks: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        chunk = candidate.chunk
        location = f"[{index}] 《{chunk.doc_name}》"
        if chunk.clause_no:
            location += f" {chunk.clause_no}"
        if chunk.version:
            location += f"（版本：{chunk.version}）"
        blocks.append(f"{location}\n{chunk.content}")
    return "\n\n".join(blocks)


class ContractReviewer:
    def __init__(
        self,
        loader: Loader,
        chunker: Chunker,
        llm: LLMClient,
        retriever: Retriever,
        reranker: Reranker,
        top_n: int,
        top_k: int,
        max_context_chunks: int = 5,
        permission_filter: PermissionFilter | None = None,
        version_filter: VersionFilter | None = None,
        reflector: Reflector | None = None,
        max_retries: int = 1,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.top_n = top_n
        self.top_k = top_k
        self.max_context_chunks = max_context_chunks
        self.permission_filter = permission_filter
        self.version_filter = version_filter
        self.reflector = reflector
        self.max_retries = max_retries
        self.type_template = (_PROMPT_DIR / "contract_type.txt").read_text(encoding="utf-8")
        self.route_template = (_PROMPT_DIR / "clause_route.txt").read_text(encoding="utf-8")
        self.review_template = (_PROMPT_DIR / "clause_review.txt").read_text(encoding="utf-8")

    def review(self, path: str, identity: Identity | None = None) -> ContractReviewReport:
        raw = self.loader.load(path)
        if not raw.text.strip():
            raise ContractReviewError("待审合同为空")
        self._validate_contract_type(raw.doc_name, raw.text)
        clauses = self._split_clauses(raw.doc_name, raw)
        filters = self._build_filters(identity or Identity())
        reviews = [self._review_clause(clause, identity or Identity(), filters) for clause in clauses]
        return ContractReviewReport(
            contract_name=raw.doc_name,
            contract_type=ContractType.LABOR,
            summary=ContractReviewSummary(
                total=len(reviews),
                compliant=sum(item.status is ClauseCompliance.COMPLIANT for item in reviews),
                risk=sum(item.status is ClauseCompliance.RISK for item in reviews),
                no_match=sum(item.status is ClauseCompliance.NO_MATCH for item in reviews),
            ),
            clauses=reviews,
        )

    def _validate_contract_type(self, name: str, text: str) -> None:
        output = self.llm.complete(
            self.type_template.format(contract_name=name, contract=text[:4000]),
            temperature=0.0,
        )
        decision = _parse_json_model(output, _TypeDecision, "合同类型识别")
        assert isinstance(decision, _TypeDecision)
        if decision.contract_type != ContractType.LABOR.value:
            raise ContractReviewError("当前仅支持劳动合同审查")

    def _split_clauses(self, name: str, raw: RawDoc) -> list[ContractClause]:
        meta = {
            "doc_id": "contract-draft-not-indexed",
            "doc_name": name,
            "doc_type": DocType.REGULATION,
        }
        chunks = self.chunker.split(raw, meta)
        clauses = [
            ContractClause(
                clause_no=chunk.clause_no,
                content=chunk.content,
                page=chunk.page,
            )
            for chunk in chunks
            if chunk.clause_no
        ]
        if not clauses:
            raise ContractReviewError("未识别到“第X条”格式的合同条款")
        return clauses

    def _build_filters(self, identity: Identity) -> dict[str, object]:
        filters: dict[str, object] = {}
        if self.permission_filter is not None:
            filters.update(self.permission_filter.build(identity))
        if self.version_filter is not None:
            filters.update(self.version_filter.build(only_current=True))
        regulation = DocType.REGULATION.value
        allowed_doc_types = filters.get("doc_type")
        if isinstance(allowed_doc_types, list) and regulation not in allowed_doc_types:
            filters["chunk_id"] = []
        elif isinstance(allowed_doc_types, str) and allowed_doc_types != regulation:
            filters["chunk_id"] = []
        filters["doc_type"] = regulation
        return filters

    def _review_clause(
        self,
        clause: ContractClause,
        identity: Identity,
        filters: dict[str, object],
    ) -> ClauseReview:
        route_output = self.llm.complete(
            self.route_template.format(
                clause_no=clause.clause_no,
                clause=clause.content,
            ),
            temperature=0.0,
        )
        route = _parse_json_model(route_output, _ClauseRoute, "条款分类")
        assert isinstance(route, _ClauseRoute)
        query = Query(text=route.retrieval_query, identity=identity, top_k=self.top_k)
        candidates = self._retrieve(query, filters)
        if not candidates:
            return ClauseReview(
                **clause.model_dump(),
                category=route.category,
                retrieval_query=route.retrieval_query,
                status=ClauseCompliance.NO_MATCH,
                reason="未检索到足以支持合规判断的现行法规。",
            )

        used = candidates[: self.max_context_chunks]
        review_output = self.llm.complete(
            self.review_template.format(
                clause_no=clause.clause_no,
                clause=clause.content,
                context=_format_context(used),
            ),
            temperature=0.0,
        )
        decision = _parse_json_model(review_output, _ClauseDecision, "条款审查")
        assert isinstance(decision, _ClauseDecision)
        if any(index < 1 or index > len(used) for index in decision.evidence_indices):
            raise ContractReviewError("条款审查证据序号超出本次检索上下文范围")
        evidence = [used[index - 1] for index in decision.evidence_indices]
        citations = validate_citations(assemble_citations(evidence), evidence)
        return ClauseReview(
            **clause.model_dump(),
            category=route.category,
            retrieval_query=route.retrieval_query,
            status=decision.status,
            reason=decision.reason,
            citations=citations,
            retrieved_chunk_ids=[item.chunk.chunk_id for item in evidence],
        )

    def _retrieve(
        self,
        initial_query: Query,
        filters: dict[str, object],
    ) -> list[Candidate]:
        query = initial_query
        retries = 0
        while True:
            candidates = self.retriever.search(query, filters, self.top_n)
            candidates = self.reranker.rerank(query, candidates, self.top_k)
            if self.reflector is None or not self.reflector.should_retry(candidates):
                return candidates
            if retries >= self.max_retries:
                return []
            rewritten = self.reflector.rewrite(query)
            if rewritten.text.strip() == query.text.strip():
                return []
            query = rewritten
            retries += 1
