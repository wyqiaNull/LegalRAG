"""合同审查 draft gold 加载与确定性指标。"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..core.errors import ContractReviewError
from .models import ContractGold, ContractReviewMetrics, ContractReviewReport


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def load_contract_gold(path: str | Path) -> ContractGold:
    target = Path(path)
    if not target.exists():
        raise ContractReviewError(f"合同 gold 不存在：{target}")
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        return ContractGold.model_validate(raw)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ContractReviewError(f"合同 gold 无效：{exc}") from exc


def evaluate_contract_report(
    report: ContractReviewReport,
    gold: ContractGold,
) -> ContractReviewMetrics:
    if report.contract_name != gold.contract_name:
        raise ContractReviewError(
            f"报告与 gold 合同名不一致：{report.contract_name!r} != {gold.contract_name!r}"
        )
    if report.contract_type is not gold.contract_type:
        raise ContractReviewError("报告与 gold 合同类型不一致")

    actual = {clause.clause_no: clause for clause in report.clauses}
    expected = {clause.clause_no: clause for clause in gold.clauses}
    common = set(actual) & set(expected)
    status_matches = sum(
        actual[number].status is expected[number].expected_status for number in common
    )

    citation_hits = citation_predicted = citation_expected = 0
    for number in common:
        actual_keys = {
            (citation.doc_name, citation.version, citation.clause_no)
            for citation in actual[number].citations
        }
        expected_keys = {
            citation.key() for citation in expected[number].expected_citations
        }
        citation_hits += len(actual_keys & expected_keys)
        citation_predicted += len(actual_keys)
        citation_expected += len(expected_keys)

    return ContractReviewMetrics(
        review_status=gold.review_status,
        gold_clause_count=len(expected),
        matched_clause_count=len(common),
        missing_clause_count=len(set(expected) - set(actual)),
        unexpected_clause_count=len(set(actual) - set(expected)),
        status_accuracy=_ratio(status_matches, len(expected)),
        citation_precision=_ratio(citation_hits, citation_predicted),
        citation_recall=_ratio(citation_hits, citation_expected),
    )
