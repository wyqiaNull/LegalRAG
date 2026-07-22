"""分块结果的可复现统计。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from ..core.interfaces import Chunker, Loader
from .chunkers.clause import article_headings


@dataclass(frozen=True)
class LengthSummary:
    count: int
    minimum: int
    maximum: int
    average: float
    median: float


@dataclass(frozen=True)
class ChunkCorpusStats:
    all_chunks: LengthSummary
    clause_chunks: LengthSummary
    source_articles: int
    one_article_chunks: int

    @property
    def one_to_one_ratio(self) -> float:
        if not self.source_articles:
            return 0.0
        return self.one_article_chunks / self.source_articles


def _summarize(lengths: list[int]) -> LengthSummary:
    if not lengths:
        return LengthSummary(0, 0, 0, 0.0, 0.0)
    return LengthSummary(
        count=len(lengths),
        minimum=min(lengths),
        maximum=max(lengths),
        average=mean(lengths),
        median=median(lengths),
    )


def analyze_chunks(paths: list[str], loader: Loader, chunker: Chunker) -> ChunkCorpusStats:
    all_lengths: list[int] = []
    clause_lengths: list[int] = []
    source_articles = 0
    one_article_chunks = 0
    for path in paths:
        raw = loader.load(path)
        source_articles += len(article_headings(raw.text))
        chunks = chunker.split(raw, {"doc_id": raw.doc_name, "doc_name": raw.doc_name})
        for chunk in chunks:
            all_lengths.append(len(chunk.content))
            if not chunk.clause_no:
                continue
            clause_lengths.append(len(chunk.content))
            if article_headings(chunk.content) == [chunk.clause_no]:
                one_article_chunks += 1
    return ChunkCorpusStats(
        all_chunks=_summarize(all_lengths),
        clause_chunks=_summarize(clause_lengths),
        source_articles=source_articles,
        one_article_chunks=one_article_chunks,
    )
