"""按排名融合多路候选。"""

from __future__ import annotations

from ..core.interfaces import Candidate
from ..core.models import CandidateSource


def reciprocal_rank_fusion(
    rankings: list[list[Candidate]], *, k: int = 60, top_n: int | None = None
) -> list[Candidate]:
    """使用 RRF 融合候选，不直接相加不同检索器的原始分数。"""
    scores: dict[str, float] = {}
    candidates: dict[str, Candidate] = {}
    best_ranks: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0

    for ranking in rankings:
        route_seen: set[str] = set()
        rank = 0
        for candidate in ranking:
            chunk_id = candidate.chunk.chunk_id
            if chunk_id in route_seen:
                continue
            route_seen.add(chunk_id)
            rank += 1
            if chunk_id not in candidates:
                candidates[chunk_id] = candidate
                first_seen[chunk_id] = seen_counter
                seen_counter += 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            best_ranks[chunk_id] = min(best_ranks.get(chunk_id, rank), rank)

    ordered_ids = sorted(
        candidates,
        key=lambda chunk_id: (
            -scores[chunk_id],
            best_ranks[chunk_id],
            first_seen[chunk_id],
            chunk_id,
        ),
    )
    if top_n is not None:
        ordered_ids = ordered_ids[:top_n]
    return [
        candidates[chunk_id].model_copy(
            update={"score": scores[chunk_id], "source": CandidateSource.FUSED}
        )
        for chunk_id in ordered_ids
    ]
