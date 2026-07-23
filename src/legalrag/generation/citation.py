"""从生成上下文装配并校验结构化引用。"""

from __future__ import annotations

from collections.abc import Iterable

from ..core.models import Candidate, Citation

CitationKey = tuple[str, str, str, int | None]


def citation_key(citation: Citation) -> CitationKey:
    """返回引用在当前稳定契约下的完整定位键。"""
    return (
        citation.doc_name,
        citation.version,
        citation.clause_no,
        citation.page,
    )


def citation_for(candidate: Candidate) -> Citation:
    chunk = candidate.chunk
    return Citation(
        doc_name=chunk.doc_name,
        version=chunk.version,
        clause_no=chunk.clause_no,
        page=chunk.page,
    )


def assemble_citations(contexts: Iterable[Candidate]) -> list[Citation]:
    """按 context 排名稳定去重，引用信息只取自 chunk 元数据。"""
    citations: list[Citation] = []
    seen: set[CitationKey] = set()
    for context in contexts:
        citation = citation_for(context)
        key = citation_key(citation)
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)
    return citations


def validate_citations(
    citations: Iterable[Citation], contexts: Iterable[Candidate]
) -> list[Citation]:
    """仅保留能精确映射回本次生成 context 的引用。"""
    allowed = {citation_key(citation_for(context)) for context in contexts}
    return [citation for citation in citations if citation_key(citation) in allowed]


def historical_versions(contexts: Iterable[Candidate]) -> list[str]:
    """按 context 顺序返回实际参与生成的历史版本。"""
    versions: list[str] = []
    seen: set[str] = set()
    for context in contexts:
        chunk = context.chunk
        if chunk.is_current:
            continue
        version = chunk.version or "未知版本"
        if version not in seen:
            seen.add(version)
            versions.append(version)
    return versions


def is_historical_citation(
    citation: Citation, contexts: Iterable[Candidate]
) -> bool:
    """判断引用是否映射到本次 context 中的历史候选。"""
    key = citation_key(citation)
    return any(
        not context.chunk.is_current and citation_key(citation_for(context)) == key
        for context in contexts
    )
