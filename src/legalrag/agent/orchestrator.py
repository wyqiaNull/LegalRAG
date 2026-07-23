"""Agentic 查询编排：指代消解、检索前路由与正常 RAG 链路。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core.interfaces import (
    Generator,
    PermissionFilter,
    Reranker,
    Retriever,
    Router,
    VersionFilter,
)
from ..core.models import Answer, Candidate, Identity, Query, Route


class CoreferenceResolver(Protocol):
    def resolve(self, question: str, history: list[str]) -> str: ...


class ConversationStore(Protocol):
    def get(self, session_id: str) -> list[str]: ...

    def append(self, session_id: str, standalone_question: str) -> None: ...


@dataclass(frozen=True)
class QueryExecution:
    answer: Answer
    candidates: list[Candidate]
    retrieval_question: str


_BRANCH_ANSWERS: dict[Route, Answer] = {
    Route.CHITCHAT: Answer(
        text="您好，我可以协助查询中国法律法规、企业制度和法律意见。请问有什么需要？",
        route=Route.CHITCHAT,
    ),
    Route.OUT_OF_SCOPE: Answer(
        text="当前知识库未覆盖该问题，无法提供可靠回答。",
        refused=True,
        reason=Route.OUT_OF_SCOPE.value,
        route=Route.OUT_OF_SCOPE,
    ),
    Route.NEED_CLARIFY: Answer(
        text="信息不足，请补充具体法规、制度、合同或问题背景。",
        route=Route.NEED_CLARIFY,
    ),
    Route.CONTRACT_REVIEW: Answer(
        text="已识别为合同审查请求，当前版本尚未开放合同审查功能。",
        refused=True,
        reason="contract_review_unavailable",
        route=Route.CONTRACT_REVIEW,
    ),
}


class AgentOrchestrator:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        top_n: int,
        top_k: int,
        router: Router | None = None,
        permission_filter: PermissionFilter | None = None,
        version_filter: VersionFilter | None = None,
        coreference: CoreferenceResolver | None = None,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_n = top_n
        self.top_k = top_k
        self.router = router
        self.permission_filter = permission_filter
        self.version_filter = version_filter
        self.coreference = coreference
        self.conversation_store = conversation_store

    def run(
        self,
        text: str,
        session_id: str | None = None,
        identity: Identity | None = None,
        version: str | None = None,
    ) -> QueryExecution:
        retrieval_question = text
        if (
            session_id
            and self.coreference is not None
            and self.conversation_store is not None
        ):
            history = self.conversation_store.get(session_id)
            retrieval_question = self.coreference.resolve(text, history)

        query = Query(
            text=retrieval_question,
            session_id=session_id,
            identity=identity or Identity(),
            top_k=self.top_k,
        )
        route = self.router.route(query) if self.router is not None else Route.NORMAL
        if route is not Route.NORMAL:
            return QueryExecution(
                answer=_BRANCH_ANSWERS[route].model_copy(deep=True),
                candidates=[],
                retrieval_question=retrieval_question,
            )

        filters: dict[str, object] = {}
        if self.permission_filter is not None:
            filters.update(self.permission_filter.build(query.identity))
        if self.version_filter is not None:
            filters.update(self.version_filter.build(only_current=version is None))
            if version is not None:
                filters["version"] = version

        candidates = self.retriever.search(query, filters or None, self.top_n)
        candidates = self.reranker.rerank(query, candidates, self.top_k)
        answer = self.generator.generate(query, candidates)
        if session_id and self.coreference is not None and self.conversation_store is not None:
            self.conversation_store.append(session_id, retrieval_question)
        return QueryExecution(answer, candidates, retrieval_question)
