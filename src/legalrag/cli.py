"""CLI 入口与组件组装根。

本模块是**组装根**：集中 import 各能力实现以触发注册，再按 config 经 registry 注入，
主流程不 import 任何具体实现。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import typer

from .application import (
    Components,
    QueryExecution,
    build_component,
    run_contract_review,
    run_ingest,
    run_ingest_details,
    run_query,
    run_query_details,
)
from .config.settings import load_settings
from .contract import (
    ContractReviewMetrics,
    ContractReviewReport,
    evaluate_contract_report,
    load_contract_gold,
)
from .core.models import (
    Candidate,
    Citation,
    Confidentiality,
    DocType,
    Identity,
)
from .generation.citation import is_historical_citation
from .ingest.stats import LengthSummary, analyze_chunks

app = typer.Typer(help="LegalRAG —— 企业级合同法规智能问答系统")

__all__ = [
    "Components",
    "QueryExecution",
    "app",
    "run_contract_review",
    "run_ingest",
    "run_ingest_details",
    "run_query",
    "run_query_details",
]


def _format_citation(citation: Citation, contexts: list[Candidate]) -> str:
    location = f"《{citation.doc_name}》"
    if citation.clause_no:
        location += f" {citation.clause_no}"
    details = []
    if citation.version:
        details.append(f"版本：{citation.version}")
    if is_historical_citation(citation, contexts):
        details.append("历史版本")
    if details:
        location += f"（{'；'.join(details)}）"
    page = f"第{citation.page}页" if citation.page is not None else "页码未知"
    return f"{location}，{page}"


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


def _print_contract_report(report: ContractReviewReport) -> None:
    typer.echo(f"合同：{report.contract_name}（{report.contract_type.value}）")
    for clause in report.clauses:
        typer.echo(
            f"\n{clause.clause_no} [{clause.category.value}] {clause.status.value}"
        )
        typer.echo(clause.content)
        typer.echo(clause.reason)
        if clause.citations:
            typer.echo("引用来源：")
            for index, citation in enumerate(clause.citations, start=1):
                typer.echo(f"[{index}] {_format_citation(citation, [])}")
    summary = report.summary
    typer.echo(
        f"\n汇总：total={summary.total} compliant={summary.compliant} "
        f"risk={summary.risk} no_match={summary.no_match}"
    )


def _print_contract_metrics(metrics: ContractReviewMetrics) -> None:
    typer.echo(
        "合同 gold 指标"
        f"（review_status={metrics.review_status}）："
        f"status_accuracy={metrics.status_accuracy:.4f} "
        f"citation_precision={metrics.citation_precision:.4f} "
        f"citation_recall={metrics.citation_recall:.4f} "
        f"missing={metrics.missing_clause_count} "
        f"unexpected={metrics.unexpected_clause_count}"
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
    result = run_ingest_details(path, settings, **doc_meta)
    stats = result.stats
    typer.echo(
        f"摄取完成：{path}；总块数={stats.total_chunks}，"
        f"重新 embedding={stats.embedded_chunks}，"
        f"向量复用={stats.reused_vectors}，被取代块数={stats.superseded_chunks}"
    )


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
    version: str | None = typer.Option(
        None, "--version", help="显式查询指定版本；历史版本会被标注"
    ),
) -> None:
    """检索并生成答案。"""
    settings = load_settings(config)
    permissions_enabled = settings.config.governance.permissions_enabled
    if version is not None and not settings.config.governance.versions_enabled:
        raise typer.BadParameter(
            "当前配置未启用版本查询", param_hint="--version"
        )
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
    if version is None:
        result = run_query_details(text, settings, session_id, identity)
    else:
        result = run_query_details(
            text, settings, session_id, identity, version=version
        )
    answer = result.answer
    typer.echo(answer.text)
    if answer.citations:
        used_ids = set(answer.retrieved_chunk_ids)
        used_contexts = [
            candidate
            for candidate in result.candidates
            if candidate.chunk.chunk_id in used_ids
        ]
        typer.echo("\n引用来源：")
        for index, citation in enumerate(answer.citations, start=1):
            typer.echo(f"[{index}] {_format_citation(citation, used_contexts)}")
    if answer.retrieved_chunk_ids:
        typer.echo(f"\n[命中 {len(answer.retrieved_chunk_ids)} 个 chunk]")
    if show_hits:
        typer.echo(f"[路由] {answer.route.value}")
        if result.retrieval_question != text:
            typer.echo(f"[独立检索问题] {result.retrieval_question}")
        if len(result.retrieval_attempts) > 1:
            typer.echo(f"[反思重试] {len(result.retrieval_attempts) - 1} 次")
            for index, question in enumerate(result.retrieval_attempts[1:], start=1):
                typer.echo(f"[改写检索问题 {index}] {question}")
        for rank, candidate in enumerate(result.candidates, start=1):
            chunk = candidate.chunk
            location = f"《{chunk.doc_name}》 {chunk.clause_no}".strip()
            typer.echo(
                f"{rank}. {location} source={candidate.source.value} "
                f"score={candidate.score:.6f} version={chunk.version or '-'} "
                f"status={'current' if chunk.is_current else 'history'} "
                f"chunk_id={chunk.chunk_id}"
            )


@app.command("contract-review")
def contract_review(
    path: str = typer.Argument(..., help="待审劳动合同草稿（pdf/docx/txt）"),
    config: str = typer.Option(None, "--config", "-c", help="配置文件路径"),
    output: str | None = typer.Option(None, "--output", "-o", help="JSON 报告输出路径"),
    gold: str | None = typer.Option(None, "--gold", help="可选的条款级 draft gold"),
    user_id: str = typer.Option("anonymous", "--user-id", help="用户 ID"),
    role: str | None = typer.Option(None, "--role", help="用户角色"),
    tenant_id: str | None = typer.Option(None, "--tenant-id", help="所属租户"),
    allowed_confidentiality: list[Confidentiality] = typer.Option(
        [],
        "--allowed-confidentiality",
        help="进一步限制可见密级，可重复指定",
    ),
) -> None:
    """上传劳动合同草稿并输出逐条款合规报告，草稿不会入库。"""
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
    report = run_contract_review(path, settings, identity)
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        typer.echo(f"合同审查报告已写入：{target}")
    else:
        _print_contract_report(report)
    if gold:
        metrics = evaluate_contract_report(report, load_contract_gold(gold))
        _print_contract_metrics(metrics)


@app.command("chunk-stats")
def chunk_stats(
    path: str = typer.Argument(..., help="单个文档或文档目录"),
    config: str = typer.Option(None, "--config", "-c", help="配置文件路径"),
) -> None:
    """统计分块完整率及字符长度分布，不执行 embedding 或落库。"""
    settings = load_settings(config)
    loader = build_component("loader", settings.config.ingest.loader)
    chunker = build_component(
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


from .eval.cli import register as _register_eval  # noqa: E402

_register_eval(app)


if __name__ == "__main__":
    app()
