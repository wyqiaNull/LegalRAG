"""CLI 入口与组件组装根。

本模块是**组装根**：集中 import 各能力实现以触发注册，再按 config 经 registry 注入，
主流程不 import 任何具体实现。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import typer

from . import agent, embedding, generation, llm, rerank, retrieval, store  # noqa: F401
from .agent.session import JsonConversationStore
from .config.settings import Settings, load_settings
from .core import registry
from .core.errors import ConfigError
from .core.models import Answer, Candidate, Confidentiality, DocType, Identity, Query
from .ingest import IngestPipeline  # 触发 ingest 包注册
from .ingest.stats import LengthSummary, analyze_chunks

app = typer.Typer(help="LegalRAG —— 企业级合同法规智能问答系统")


def _build(kind: str, name: str, **candidates: Any) -> Any:
    """按构造签名过滤 kwargs 后实例化 —— 让组装根无需知晓各实现的私有参数差异。"""
    cls = registry.get(kind, name)
    params = inspect.signature(cls.__init__).parameters
    accepted = {k: v for k, v in candidates.items() if k in params}
    return cls(**accepted)


class Components:
    """一次组装好的全套组件（供 ingest 与 query 复用）。"""

    def __init__(self, settings: Settings) -> None:
        cfg, sec = settings.config, settings.secrets
        self.settings = settings

        self.embedder = _build(
            "embedder",
            cfg.embedding.impl,
            api_base=sec.embedding_api_base,
            api_key=sec.embedding_api_key,
            model=sec.embedding_model,
            dim=cfg.embedding.dim,
            batch_size=cfg.embedding.batch_size,
        )
        self.vector_store = _build("vector_store", cfg.store.vector, path=cfg.store.path)
        self.metadata_store = _build(
            "metadata_store",
            cfg.store.metadata,
            path=cfg.store.path,
            dsn=sec.postgres_dsn,
            acl_policies=[
                policy.model_dump(mode="json")
                for policy in cfg.governance.acl_policies
            ],
        )
        self.permission_filter = None
        if cfg.governance.permissions_enabled:
            self.permission_filter = _build(
                "permission_filter",
                cfg.governance.permission_filter,
                metadata_store=self.metadata_store,
            )
        self.llm = _build(
            "llm",
            cfg.llm.impl,
            api_base=sec.llm_api_base,
            api_key=sec.llm_api_key,
            model=sec.llm_model,
        )
        self.reranker = _build(
            "reranker",
            cfg.rerank.impl,
            api_base=sec.rerank_api_base or sec.embedding_api_base,
            api_key=sec.rerank_api_key or sec.embedding_api_key,
            model=sec.rerank_model,
            timeout=cfg.rerank.timeout,
        )
        self.generator = _build(
            "generator",
            cfg.generation.impl,
            llm=self.llm,
            max_context_chunks=cfg.generation.max_context_chunks,
        )
        self.retriever = _build(
            "retriever",
            cfg.retrieval.impl,
            embedder=self.embedder,
            vector_store=self.vector_store,
            rrf_k=cfg.retrieval.rrf_k,
        )
        self.coreference = None
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
        self.conversation_store = JsonConversationStore(
            conversation_path, cfg.conversation.history_turns
        )

    def pipeline(self) -> IngestPipeline:
        loader = _build("loader", self.settings.config.ingest.loader)
        chunker = _build(
            "chunker",
            self.settings.config.ingest.chunker,
            chunk_size=self.settings.config.ingest.chunk_size,
            chunk_overlap=self.settings.config.ingest.chunk_overlap,
        )
        return IngestPipeline(
            loader, chunker, self.embedder, self.vector_store, self.metadata_store
        )


# ============ 可被测试直接调用的编排函数 ============


@dataclass(frozen=True)
class QueryExecution:
    answer: Answer
    candidates: list[Candidate]
    retrieval_question: str


def run_ingest(path: str, settings: Settings | None = None, **doc_meta: Any) -> int:
    settings = settings or load_settings()
    comp = Components(settings)
    chunks = comp.pipeline().run(path, **doc_meta)
    return len(chunks)


def run_query_details(
    text: str,
    settings: Settings | None = None,
    session_id: str | None = None,
    identity: Identity | None = None,
) -> QueryExecution:
    settings = settings or load_settings()
    if settings.config.governance.permissions_enabled and identity is None:
        raise ConfigError("权限过滤启用时必须提供查询身份")
    comp = Components(settings)
    cfg = settings.config.retrieval
    retrieval_question = text
    if session_id and comp.coreference is not None:
        history = comp.conversation_store.get(session_id)
        retrieval_question = comp.coreference.resolve(text, history)
    query = Query(
        text=retrieval_question,
        session_id=session_id,
        identity=identity or Identity(),
        top_k=cfg.top_k,
    )
    filters = None
    if comp.permission_filter is not None:
        filters = comp.permission_filter.build(query.identity)
    candidates = comp.retriever.search(query, filters, cfg.top_n)
    candidates = comp.reranker.rerank(query, candidates, cfg.top_k)
    answer = comp.generator.generate(query, candidates)
    if session_id and comp.coreference is not None:
        comp.conversation_store.append(session_id, retrieval_question)
    return QueryExecution(answer, candidates, retrieval_question)


def run_query(
    text: str,
    settings: Settings | None = None,
    session_id: str | None = None,
    identity: Identity | None = None,
) -> Answer:
    return run_query_details(text, settings, session_id, identity).answer


def _input_documents(path: str) -> list[str]:
    target = Path(path)
    supported = {".pdf", ".docx", ".txt", ".md"}
    if target.is_file() and target.suffix.lower() in supported:
        return [str(target)]
    if target.is_dir():
        return [
            str(candidate)
            for candidate in sorted(target.iterdir())
            if candidate.is_file() and candidate.suffix.lower() in supported
        ]
    raise typer.BadParameter(f"找不到支持的文档：{path}")


def _print_summary(name: str, summary: LengthSummary) -> None:
    typer.echo(
        f"{name}: count={summary.count}, min={summary.minimum}, "
        f"max={summary.maximum}, avg={summary.average:.2f}, "
        f"median={summary.median:g}"
    )


# ============ Typer 命令 ============


@app.command()
def ingest(
    path: str = typer.Argument(..., help="待摄取的文件路径（pdf/docx/txt）"),
    config: str = typer.Option(None, "--config", "-c", help="配置文件路径"),
    doc_name: str | None = typer.Option(
        None, "--doc-name", help="稳定文档名；多版本文档应保持一致"
    ),
    doc_type: DocType = typer.Option(
        DocType.REGULATION, "--doc-type", help="文档类型"
    ),
    tenant_id: str = typer.Option("default", "--tenant-id", help="所属租户"),
    department: str = typer.Option("", "--department", help="所属部门"),
    allowed_role: list[str] = typer.Option(
        [], "--allowed-role", help="允许访问的角色，可重复指定"
    ),
    confidentiality: Confidentiality = typer.Option(
        Confidentiality.PUBLIC, "--confidentiality", help="文档密级"
    ),
    version: str = typer.Option("", "--version", help="文档版本号"),
    effective_date: str | None = typer.Option(
        None, "--effective-date", help="生效日期，格式 YYYY-MM-DD"
    ),
) -> None:
    """摄取一个文档入库。"""
    settings = load_settings(config)
    normalized_date = None
    if effective_date:
        try:
            normalized_date = date.fromisoformat(effective_date).isoformat()
        except ValueError as exc:
            raise typer.BadParameter(
                "生效日期必须使用 YYYY-MM-DD 格式", param_hint="--effective-date"
            ) from exc
    doc_meta: dict[str, Any] = {
        "doc_type": doc_type,
        "tenant_id": tenant_id,
        "department": department,
        "confidentiality": confidentiality,
        "version": version,
        "effective_date": normalized_date,
    }
    if doc_name:
        doc_meta["doc_name"] = doc_name
    if allowed_role:
        doc_meta["allowed_roles"] = list(dict.fromkeys(allowed_role))
    n = run_ingest(path, settings, **doc_meta)
    typer.echo(f"✅ 摄取完成：{path} → {n} 个 chunk 已入库")


@app.command()
def query(
    text: str = typer.Argument(..., help="用户问题"),
    config: str = typer.Option(None, "--config", "-c", help="配置文件路径"),
    session_id: str | None = typer.Option(None, "--session-id", help="多轮会话 ID"),
    show_hits: bool = typer.Option(False, "--show-hits", help="显示检索候选与分数"),
    user_id: str = typer.Option("anonymous", "--user-id", help="用户 ID"),
    role: str | None = typer.Option(None, "--role", help="用户角色"),
    tenant_id: str | None = typer.Option(None, "--tenant-id", help="所属租户"),
    allowed_confidentiality: list[Confidentiality] = typer.Option(
        [],
        "--allowed-confidentiality",
        help="进一步限制可见密级，可重复指定",
    ),
) -> None:
    """检索并生成答案。"""
    settings = load_settings(config)
    permissions_enabled = settings.config.governance.permissions_enabled
    if permissions_enabled and not role:
        raise typer.BadParameter(
            "权限过滤启用时必须提供用户角色", param_hint="--role"
        )
    if permissions_enabled and not tenant_id:
        raise typer.BadParameter(
            "权限过滤启用时必须提供所属租户", param_hint="--tenant-id"
        )
    identity = None
    if role or tenant_id or allowed_confidentiality:
        identity = Identity(
            user_id=user_id,
            role=role or "*",
            tenant_id=tenant_id or "default",
            allowed_confidentiality=(
                list(dict.fromkeys(allowed_confidentiality))
                if allowed_confidentiality
                else list(Confidentiality)
            ),
        )
    result = run_query_details(text, settings, session_id, identity)
    answer = result.answer
    typer.echo(answer.text)
    if answer.retrieved_chunk_ids:
        typer.echo(f"\n[命中 {len(answer.retrieved_chunk_ids)} 个 chunk]")
    if show_hits:
        if result.retrieval_question != text:
            typer.echo(f"[独立检索问题] {result.retrieval_question}")
        for rank, candidate in enumerate(result.candidates, start=1):
            chunk = candidate.chunk
            location = f"《{chunk.doc_name}》 {chunk.clause_no}".strip()
            typer.echo(
                f"{rank}. {location} source={candidate.source.value} "
                f"score={candidate.score:.6f} chunk_id={chunk.chunk_id}"
            )


@app.command("chunk-stats")
def chunk_stats(
    path: str = typer.Argument(..., help="单个文档或文档目录"),
    config: str = typer.Option(None, "--config", "-c", help="配置文件路径"),
) -> None:
    """统计分块完整率及字符长度分布，不执行 embedding 或落库。"""
    settings = load_settings(config)
    loader = _build("loader", settings.config.ingest.loader)
    chunker = _build(
        "chunker",
        settings.config.ingest.chunker,
        chunk_size=settings.config.ingest.chunk_size,
        chunk_overlap=settings.config.ingest.chunk_overlap,
    )
    stats = analyze_chunks(_input_documents(path), loader, chunker)
    typer.echo(
        "法条一对一完整率: "
        f"{stats.one_article_chunks}/{stats.source_articles} "
        f"({stats.one_to_one_ratio:.2%})"
    )
    _print_summary("法条 chunk", stats.clause_chunks)
    _print_summary("全部 chunk", stats.all_chunks)


if __name__ == "__main__":
    app()
