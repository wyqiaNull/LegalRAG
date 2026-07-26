"""Agentic 查询编排：指代消解、检索前路由、反思改写与正常 RAG 链路。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core.interfaces import (
    ConversationStore,
    Generator,
    PermissionFilter,
    Reflector,
    Reranker,
    Retriever,
    Router,
    VersionFilter,
)
from ..core.models import Answer, Candidate, Identity, Query, Route
from .refusal import RefusalReason, RefusalResponder


class CoreferenceResolver(Protocol):
    def resolve(self, question: str, history: list[str]) -> str: ...


@dataclass(frozen=True)
class QueryExecution:
    answer: Answer
    candidates: list[Candidate]
    retrieval_question: str
    retrieval_attempts: tuple[str, ...] = ()
    retrieval_signals: tuple["RetrievalSignal", ...] = ()


@dataclass(frozen=True)
class RetrievalSignal:
    question: str
    candidate_count: int
    top_score: float | None


_CHITCHAT_ANSWER = Answer(
    text="您好，我可以协助查询中国法律法规、企业制度和法律意见。请问有什么需要？",
    route=Route.CHITCHAT,
)

_ROUTE_REFUSALS: dict[Route, RefusalReason] = {
    Route.OUT_OF_SCOPE: RefusalReason.OUT_OF_SCOPE,
    Route.NEED_CLARIFY: RefusalReason.INSUFFICIENT_INFORMATION,
    Route.CONTRACT_REVIEW: RefusalReason.CONTRACT_FILE_REQUIRED,
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
        refusal_responder: RefusalResponder | None = None,
        reflector: Reflector | None = None,
        max_retries: int = 1,
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
        self.refusal_responder = refusal_responder or RefusalResponder()
        self.reflector = reflector
        self.max_retries = max_retries

    def run(
        self,
        text: str,
        session_id: str | None = None,
        identity: Identity | None = None,
        version: str | None = None,
    ) -> QueryExecution:
        query_identity = identity or Identity()
        retrieval_question = text
        if (
            session_id
            and self.coreference is not None
            and self.conversation_store is not None
        ):
            history = self.conversation_store.get(query_identity, session_id)
            retrieval_question = self.coreference.resolve(text, history)

        original_query = Query(
            text=retrieval_question,
            session_id=session_id,
            identity=query_identity,
            top_k=self.top_k,
        )
        route = (
            self.router.route(original_query)
            if self.router is not None
            else Route.NORMAL
        )
        if route is Route.CHITCHAT:
            return QueryExecution(
                answer=_CHITCHAT_ANSWER.model_copy(deep=True),
                candidates=[],
                retrieval_question=retrieval_question,
            )
        if route is not Route.NORMAL:
            return QueryExecution(
                answer=self.refusal_responder.refuse(
                    _ROUTE_REFUSALS[route], route=route
                ),
                candidates=[],
                retrieval_question=retrieval_question,
            )

        filters: dict[str, object] = {}
        if self.permission_filter is not None:
            filters.update(self.permission_filter.build(original_query.identity))
        if self.version_filter is not None:
            filters.update(self.version_filter.build(only_current=version is None))
            if version is not None:
                filters["version"] = version

        retrieval_query = original_query
        attempts = [retrieval_query.text]
        signals: list[RetrievalSignal] = []
        retries = 0
        while True:
            candidates = self.retriever.search(
                retrieval_query, filters or None, self.top_n
            )
            candidates = self.reranker.rerank(
                retrieval_query, candidates, self.top_k
            )
            signals.append(
                RetrievalSignal(
                    question=retrieval_query.text,
                    candidate_count=len(candidates),
                    top_score=max((candidate.score for candidate in candidates), default=None),
                )
            )
            if self.reflector is None or not self.reflector.should_retry(candidates):
                break
            if retries >= self.max_retries:
                return QueryExecution(
                    answer=self.refusal_responder.refuse(
                        RefusalReason.NO_ACCESSIBLE_CONTEXT
                    ),
                    candidates=candidates,
                    retrieval_question=retrieval_question,
                    retrieval_attempts=tuple(attempts),
                    retrieval_signals=tuple(signals),
                )
            rewritten = self.reflector.rewrite(retrieval_query)
            if rewritten.text.strip() == retrieval_query.text.strip():
                return QueryExecution(
                    answer=self.refusal_responder.refuse(
                        RefusalReason.NO_ACCESSIBLE_CONTEXT
                    ),
                    candidates=candidates,
                    retrieval_question=retrieval_question,
                    retrieval_attempts=tuple(attempts),
                    retrieval_signals=tuple(signals),
                )
            retrieval_query = rewritten
            attempts.append(retrieval_query.text)
            retries += 1

        if not candidates:
            return QueryExecution(
                answer=self.refusal_responder.refuse(
                    RefusalReason.NO_ACCESSIBLE_CONTEXT
                ),
                candidates=[],
                retrieval_question=retrieval_question,
                retrieval_attempts=tuple(attempts),
                retrieval_signals=tuple(signals),
            )
        answer = self.generator.generate(original_query, candidates)
        if session_id and self.coreference is not None and self.conversation_store is not None:
            self.conversation_store.append(
                original_query.identity, session_id, retrieval_question
            )
        return QueryExecution(
            answer,
            candidates,
            retrieval_question,
            tuple(attempts),
            tuple(signals),
        )
