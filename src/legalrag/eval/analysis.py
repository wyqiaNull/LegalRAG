"""Offline threshold calibration and paired ablation summaries."""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean

from .matrix import EvalMatrix
from .models import EvalCase, EvalObservation


def _turn_lookup(cases: list[EvalCase]):
    return {
        (case.case_id, turn_index): (case, turn)
        for case in cases
        for turn_index, turn in enumerate(case.turns)
    }


def _context_recall(turn, observation: EvalObservation) -> float | None:
    if not turn.expected_citations:
        return None
    hits = sum(
        any(context.matches(citation) for context in observation.contexts)
        for citation in turn.expected_citations
    )
    return hits / len(turn.expected_citations)


def _refusal_correct(turn, observation: EvalObservation) -> float:
    return float(turn.expected_refusal == observation.refused)


def _paired_summary(
    cases: list[EvalCase],
    baseline: list[EvalObservation],
    candidate: list[EvalObservation],
    scorer: Callable,
) -> dict:
    expected = _turn_lookup(cases)
    left = {(item.case_id, item.turn_index): item for item in baseline if not item.error}
    right = {(item.case_id, item.turn_index): item for item in candidate if not item.error}
    pairs: list[tuple[float, float]] = []
    for key, (_, turn) in expected.items():
        if key not in left or key not in right:
            continue
        baseline_score = scorer(turn, left[key])
        candidate_score = scorer(turn, right[key])
        if baseline_score is None or candidate_score is None:
            continue
        pairs.append((baseline_score, candidate_score))
    differences = [right_score - left_score for left_score, right_score in pairs]
    return {
        "sample_count": len(pairs),
        "baseline_mean": fmean(left_score for left_score, _ in pairs) if pairs else 0.0,
        "candidate_mean": fmean(right_score for _, right_score in pairs) if pairs else 0.0,
        "mean_difference": fmean(differences) if differences else 0.0,
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
    }


def summarize_ablations(
    matrix: EvalMatrix,
    cases: list[EvalCase],
    observations: dict[str, list[EvalObservation]],
) -> dict[str, list[dict]]:
    summaries: dict[str, list[dict]] = {}
    for experiment in matrix.experiments:
        metric = "refusal_correct" if experiment.name == "agentic" else "context_recall"
        scorer = _refusal_correct if metric == "refusal_correct" else _context_recall
        baseline = experiment.profiles[0]
        comparisons = []
        for candidate in experiment.profiles[1:]:
            comparisons.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    **_paired_summary(
                        cases,
                        observations[baseline],
                        observations[candidate],
                        scorer,
                    ),
                }
            )
        summaries[experiment.name] = comparisons
    return summaries


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _threshold_row(
    cases: list[EvalCase],
    observations: list[EvalObservation],
    *,
    min_candidates: int,
    min_score: float,
) -> dict:
    expected = _turn_lookup(cases)
    actual = {(item.case_id, item.turn_index): item for item in observations}
    tp = tn = fp = fn = 0
    context_hits = context_total = 0
    failures = 0
    for key, (_, turn) in expected.items():
        observation = actual.get(key)
        if observation is None or observation.error:
            failures += 1
            continue
        signal = observation.retrieval_signals[-1] if observation.retrieval_signals else None
        gated = (
            observation.refused
            or signal is None
            or signal.candidate_count < min_candidates
            or signal.top_score is None
            or signal.top_score < min_score
        )
        if turn.expected_refusal and gated:
            tp += 1
        elif turn.expected_refusal:
            fn += 1
        elif gated:
            fp += 1
        else:
            tn += 1

        if turn.expected_citations:
            context_total += len(turn.expected_citations)
            if not gated:
                context_hits += sum(
                    any(context.matches(citation) for context in observation.contexts)
                    for citation in turn.expected_citations
                )

    positive_precision = tp / (tp + fp) if tp + fp else 0.0
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    return {
        "min_candidates": min_candidates,
        "min_score": min_score,
        "macro_f1": (
            _f1(positive_precision, positive_recall)
            + _f1(negative_precision, negative_recall)
        )
        / 2,
        "false_refusal_rate": fp / (fp + tn) if fp + tn else 0.0,
        "context_recall": context_hits / context_total if context_total else 0.0,
        "failed_observations": failures,
    }


def calibrate_thresholds(
    cases: list[EvalCase],
    observations: list[EvalObservation],
    *,
    default_min_candidates: int = 1,
    default_min_score: float = 0.5,
) -> dict:
    rows = [
        _threshold_row(
            cases,
            observations,
            min_candidates=min_candidates,
            min_score=step / 20,
        )
        for min_candidates in (1, 2, 3)
        for step in range(21)
    ]
    default = next(
        row
        for row in rows
        if row["min_candidates"] == default_min_candidates
        and row["min_score"] == default_min_score
    )
    eligible = [
        row
        for row in rows
        if row["failed_observations"] == 0
        and row["context_recall"] >= default["context_recall"]
    ]
    recommended = sorted(
        eligible,
        key=lambda row: (
            -row["macro_f1"],
            row["false_refusal_rate"],
            abs(row["min_candidates"] - default_min_candidates),
            abs(row["min_score"] - default_min_score),
        ),
    )[0] if eligible else default
    changed = (
        recommended["min_candidates"] != default_min_candidates
        or recommended["min_score"] != default_min_score
    )
    return {
        "method": "cached non-agentic observation replay",
        "grid_size": len(rows),
        "default": default,
        "recommended": recommended,
        "changed": changed,
        "apply_default": changed and recommended["context_recall"] >= default["context_recall"],
        "reason": (
            "推荐阈值不降低正常问答条款级近似 Context Recall"
            if changed
            else "默认阈值在拒答宏 F1、误拒率和召回约束下仍为最优或并列最保守值"
        ),
        "rows": rows,
    }
