"""LegalRAG 应用服务与组件组装根。"""

from __future__ import annotations

import inspect
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import agent, embedding, generation, llm, rerank, retrieval, store  # noqa: F401
from .agent.orchestrator import AgentOrchestrator, QueryExecution
from .config.settings import Settings, load_settings
from .contract import ContractReviewReport, ContractReviewer
from .core import registry
from .core.errors import ConfigError
from .core.models import Answer, Confidentiality, DocType, Identity
from .ingest import IngestPipeline, IngestResult  # 触发 ingest 包注册
from .observability import UsageCollector


def _build(kind: str, name: str, **candidates: Any) -> Any:
    cls = registry.get(kind, name)
    params = inspect.signature(cls.__init__).parameters
    accepted = {key: value for key, value in candidates.items() if key in params}
    return cls(**accepted)


def build_component(kind: str, name: str, **candidates: Any) -> Any:
    return _build(kind, name, **candidates)


class Components:
    """一次组装好的全套组件，供 CLI、API 与 worker 复用。"""

    def __init__(
        self,
        settings: Settings,
        usage_collector: UsageCollector | None = None,
    ) -> None:
        cfg, sec = settings.config, settings.secrets
        self.settings = settings
        self.usage_collector = usage_collector

        self.embedder = _build(
            "embedder",
            cfg.embedding.impl,
            api_base=sec.embedding_api_base,
            api_key=sec.embedding_api_key,
            model=sec.embedding_model,
            dim=cfg.embedding.dim,
            batch_size=cfg.embedding.batch_size,
            usage_collector=usage_collector,
        )
        self.vector_store = _build(
            "vector_store",
            cfg.store.vector,
            path=cfg.store.path,
            uri=sec.milvus_uri,
            token=sec.milvus_token,
            collection_name=cfg.store.collection,
            dim=cfg.embedding.dim,
        )
        self.metadata_store = _build(
            "metadata_store",
            cfg.store.metadata,
            path=cfg.store.path,
            dsn=sec.postgres_dsn,
            acl_policies=[
                policy.model_dump(mode="json") for policy in cfg.governance.acl_policies
            ],
        )
        self.permission_filter = None
        if cfg.governance.permissions_enabled:
            self.permission_filter = _build(
                "permission_filter",
                cfg.governance.permission_filter,
                metadata_store=self.metadata_store,
                shared_tenant_id=cfg.governance.shared_tenant_id,
                shared_doc_types=cfg.governance.shared_doc_types,
            )
        self.version_filter = None
        if cfg.governance.versions_enabled:
            self.version_filter = _build(
                "version_filter", cfg.governance.version_filter
            )
        self.llm = _build(
            "llm",
            cfg.llm.impl,
            api_base=sec.llm_api_base,
            api_key=sec.llm_api_key,
            model=sec.llm_model,
            usage_collector=usage_collector,
        )
        self.reranker = _build(
            "reranker",
            cfg.rerank.impl,
            api_base=sec.rerank_api_base or sec.embedding_api_base,
            api_key=sec.rerank_api_key or sec.embedding_api_key,
            model=sec.rerank_model,
            timeout=cfg.rerank.timeout,
            usage_collector=usage_collector,
        )
        self.generator = _build(
            "generator",
            cfg.generation.impl,
            llm=self.llm,
            max_context_chunks=cfg.generation.max_context_chunks,
            temperature=cfg.generation.temperature,
        )
        self.retriever = _build(
            "retriever",
            cfg.retrieval.impl,
            embedder=self.embedder,
            vector_store=self.vector_store,
            rrf_k=cfg.retrieval.rrf_k,
        )
        self.coreference = None
        self.conversation_store = None
        if cfg.conversation.enabled:
            self.coreference = _build(
                "coreference",
                cfg.conversation.resolver,
                llm=self.llm,
                history_turns=cfg.conversation.history_turns,
            )
            conversation_path = cfg.conversation.path or str(
                Path(cfg.store.path) / "conversations.json"
            )
            self.conversation_store = _build(
                "conversation_store",
                cfg.conversation.store,
                path=conversation_path,
                url=sec.redis_url,
                history_turns=cfg.conversation.history_turns,
                ttl_seconds=cfg.conversation.ttl_seconds,
                key_prefix=cfg.conversation.key_prefix,
            )
        self.router = None
        if cfg.routing.enabled:
            self.router = _build("router", cfg.routing.impl, llm=self.llm)
        self.reflector = None
        if cfg.reflection.enabled:
            self.reflector = _build(
                "reflector",
                cfg.reflection.impl,
                llm=self.llm,
                min_candidates=cfg.reflection.min_candidates,
                min_score=cfg.reflection.min_score,
            )
        self.query_orchestrator = AgentOrchestrator(
            retriever=self.retriever,
            reranker=self.reranker,
            generator=self.generator,
            top_n=cfg.retrieval.top_n,
            top_k=cfg.retrieval.top_k,
            router=self.router,
            permission_filter=self.permission_filter,
            version_filter=self.version_filter,
            coreference=self.coreference,
            conversation_store=self.conversation_store,
            reflector=self.reflector,
            max_retries=cfg.reflection.max_retries,
        )
        self.contract_reviewer = None
        if cfg.contract.enabled:
            self.contract_reviewer = ContractReviewer(
                loader=_build("loader", "auto"),
                chunker=_build("chunker", "clause"),
                llm=self.llm,
                retriever=self.retriever,
                reranker=self.reranker,
                top_n=cfg.retrieval.top_n,
                top_k=cfg.retrieval.top_k,
                max_context_chunks=cfg.contract.max_context_chunks,
                permission_filter=self.permission_filter,
                version_filter=self.version_filter,
                reflector=self.reflector,
                max_retries=cfg.reflection.max_retries,
            )

    def pipeline(self) -> IngestPipeline:
        cfg = self.settings.config
        loader = _build("loader", cfg.ingest.loader)
        chunker = _build(
            "chunker",
            cfg.ingest.chunker,
            chunk_size=cfg.ingest.chunk_size,
            chunk_overlap=cfg.ingest.chunk_overlap,
        )
        return IngestPipeline(
            loader,
            chunker,
            self.embedder,
            self.vector_store,
            self.metadata_store,
            versions_enabled=cfg.governance.versions_enabled,
            shared_tenant_id=cfg.governance.shared_tenant_id,
            shared_doc_types=cfg.governance.shared_doc_types,
        )


def new_usage_collector(settings: Settings) -> UsageCollector:
    service = settings.config.service
    return UsageCollector(
        Decimal(str(service.input_cost_per_million)),
        Decimal(str(service.output_cost_per_million)),
    )


def build_vector_store(settings: Settings):
    cfg, sec = settings.config, settings.secrets
    return _build(
        "vector_store",
        cfg.store.vector,
        path=cfg.store.path,
        uri=sec.milvus_uri,
        token=sec.milvus_token,
        collection_name=cfg.store.collection,
        dim=cfg.embedding.dim,
    )


def run_ingest(path: str, settings: Settings | None = None, **doc_meta: Any) -> int:
    return len(run_ingest_details(path, settings, **doc_meta).chunks)


def run_ingest_details(
    path: str,
    settings: Settings | None = None,
    **doc_meta: Any,
) -> IngestResult:
    settings = settings or load_settings()
    if isinstance(doc_meta.get("doc_type"), str):
        doc_meta["doc_type"] = DocType(doc_meta["doc_type"])
    if isinstance(doc_meta.get("confidentiality"), str):
        doc_meta["confidentiality"] = Confidentiality(doc_meta["confidentiality"])
    return Components(settings).pipeline().run_detailed(path, **doc_meta)


def run_query_details(
    text: str,
    settings: Settings | None = None,
    session_id: str | None = None,
    identity: Identity | None = None,
    version: str | None = None,
    usage_collector: UsageCollector | None = None,
) -> QueryExecution:
    settings = settings or load_settings()
    if settings.config.governance.permissions_enabled and identity is None:
        raise ConfigError("权限过滤启用时必须提供查询身份")
    if version is not None:
        version = version.strip()
        if not version:
            raise ConfigError("显式查询的版本号不能为空")
        if not settings.config.governance.versions_enabled:
            raise ConfigError("当前配置未启用版本查询")
    components = Components(settings, usage_collector)
    return components.query_orchestrator.run(text, session_id, identity, version)


def run_query(
    text: str,
    settings: Settings | None = None,
    session_id: str | None = None,
    identity: Identity | None = None,
    version: str | None = None,
) -> Answer:
    return run_query_details(text, settings, session_id, identity, version).answer


def run_contract_review(
    path: str,
    settings: Settings | None = None,
    identity: Identity | None = None,
) -> ContractReviewReport:
    settings = settings or load_settings()
    if not settings.config.contract.enabled:
        raise ConfigError("当前配置未启用合同审查")
    if settings.config.governance.permissions_enabled and identity is None:
        raise ConfigError("权限过滤启用时必须提供合同审查身份")
    reviewer = Components(settings).contract_reviewer
    if reviewer is None:
        raise ConfigError("合同审查组件未初始化")
    return reviewer.review(path, identity)
