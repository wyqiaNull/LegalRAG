"""CLI 入口与 MVP 组装根（SPEC §3/§9）。

编排层 v0.3 才独立（agent/），MVP 阶段在此线性串起 ingest→检索→精排→生成。
本模块是**组装根**：集中 import 各能力实现以触发注册，再按 config 经 registry 注入，
主流程不 import 任何具体实现（哲学五）。
"""

from __future__ import annotations

import inspect
from typing import Any

import typer

from . import embedding, generation, llm, rerank, retrieval, store  # noqa: F401
from .config.settings import Settings, load_settings
from .core import registry
from .core.models import Answer, Identity, Query
from .ingest import IngestPipeline  # 触发 ingest 包注册

app = typer.Typer(help="LegalRAG —— 企业级合同法规智能问答系统（MVP）")


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
        )
        self.vector_store = _build("vector_store", cfg.store.vector, path=cfg.store.path)
        self.metadata_store = _build(
            "metadata_store", cfg.store.metadata, path=cfg.store.path
        )
        self.reranker = _build("reranker", cfg.rerank.impl)
        self.llm = _build(
            "llm",
            cfg.llm.impl,
            api_base=sec.llm_api_base,
            api_key=sec.llm_api_key,
            model=sec.llm_model,
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


def run_ingest(path: str, settings: Settings | None = None, **doc_meta: Any) -> int:
    settings = settings or load_settings()
    comp = Components(settings)
    chunks = comp.pipeline().run(path, **doc_meta)
    return len(chunks)


def run_query(text: str, settings: Settings | None = None) -> Answer:
    settings = settings or load_settings()
    comp = Components(settings)
    top_k = settings.config.retrieval.top_k
    query = Query(text=text, identity=Identity(), top_k=top_k)
    candidates = comp.retriever.search(query, None, top_k)  # MVP filters 传空
    candidates = comp.reranker.rerank(query, candidates, top_k)
    return comp.generator.generate(query, candidates)


# ============ Typer 命令 ============


@app.command()
def ingest(
    path: str = typer.Argument(..., help="待摄取的文件路径（pdf/docx/txt）"),
    config: str = typer.Option(None, "--config", "-c", help="配置文件路径"),
) -> None:
    """摄取一个文档入库。"""
    settings = load_settings(config)
    n = run_ingest(path, settings)
    typer.echo(f"✅ 摄取完成：{path} → {n} 个 chunk 已入库")


@app.command()
def query(
    text: str = typer.Argument(..., help="用户问题"),
    config: str = typer.Option(None, "--config", "-c", help="配置文件路径"),
) -> None:
    """检索并生成答案。"""
    settings = load_settings(config)
    answer = run_query(text, settings)
    typer.echo(answer.text)
    if answer.retrieved_chunk_ids:
        typer.echo(f"\n[命中 {len(answer.retrieved_chunk_ids)} 个 chunk]")


if __name__ == "__main__":
    app()
