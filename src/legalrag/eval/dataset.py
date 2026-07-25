"""Evaluation dataset loading, hashing and human-review gates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml

from ..core.errors import ConfigError
from .models import EvalCase, ReviewManifest

_NORMAL_DOCUMENT_TAGS = {"劳动法", "劳动合同法", "社会保险法", "公司法", "劳动仲裁"}
_REFUSAL_TAGS = {"out_of_scope", "need_clarify", "no_context"}


def dataset_sha256(path: str | Path) -> str:
    target = Path(path)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_cases(path: str | Path) -> list[EvalCase]:
    target = Path(path)
    if not target.exists():
        raise ConfigError(f"评测集不存在：{target}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            case = EvalCase.model_validate_json(line)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ConfigError(f"评测集第 {line_number} 行无效：{exc}") from exc
        if case.case_id in seen:
            raise ConfigError(f"评测集 case_id 重复：{case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ConfigError("评测集不能为空")
    return cases


def load_review(path: str | Path) -> ReviewManifest:
    target = Path(path)
    if not target.exists():
        raise ConfigError(f"审核清单不存在：{target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    try:
        return ReviewManifest.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"审核清单无效：{exc}") from exc


def validate_minimum_scale(cases: list[EvalCase]) -> dict:
    counts = {
        kind: sum(case.kind == kind for case in cases)
        for kind in ("normal", "refusal", "multiturn", "red_team")
    }
    minimums = {"normal": 30, "refusal": 15, "multiturn": 10, "red_team": 10}
    short = {kind: minimums[kind] - count for kind, count in counts.items() if count < minimums[kind]}
    if short:
        raise ConfigError(f"正式法律评测集未达到最低规模：{short}")
    normal_tags = {
        tag: sum(case.kind == "normal" and tag in case.tags for case in cases)
        for tag in _NORMAL_DOCUMENT_TAGS
    }
    if any(count < 6 for count in normal_tags.values()):
        raise ConfigError(f"正常问答未满足每部法规至少 6 条：{normal_tags}")
    refusal_tags = {
        tag: sum(case.kind == "refusal" and tag in case.tags for case in cases)
        for tag in _REFUSAL_TAGS
    }
    if any(count < 5 for count in refusal_tags.values()):
        raise ConfigError(f"拒答集未满足三类各至少 5 条：{refusal_tags}")
    if any(len(case.turns) != 2 for case in cases if case.kind == "multiturn"):
        raise ConfigError("正式多轮评测集必须是两轮追问")
    return {
        "case_counts": counts,
        "turn_count": sum(len(case.turns) for case in cases),
        "normal_document_counts": normal_tags,
        "refusal_category_counts": refusal_tags,
    }


def validate_review(
    dataset_path: str | Path,
    review_path: str | Path,
    *,
    require_approved: bool,
    review_csv_path: str | Path | None = None,
) -> ReviewManifest:
    manifest = load_review(review_path)
    actual_hash = dataset_sha256(dataset_path)
    if manifest.dataset_sha256 != actual_hash:
        raise ConfigError(
            "评测集已在审核后发生变化："
            f"manifest={manifest.dataset_sha256} actual={actual_hash}"
        )
    if require_approved and manifest.status != "approved":
        raise ConfigError("正式评测要求人工审核状态为 approved")
    if manifest.status == "approved":
        if review_csv_path is None:
            raise ConfigError("approved 审核必须提供人工审核 CSV")
        _validate_review_csv(load_cases(dataset_path), review_csv_path, manifest)
    return manifest


def _validate_review_csv(
    cases: list[EvalCase],
    review_csv_path: str | Path,
    manifest: ReviewManifest,
) -> None:
    target = Path(review_csv_path)
    if not target.exists():
        raise ConfigError(f"人工审核 CSV 不存在：{target}")
    actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual_hash != manifest.review_csv_sha256:
        raise ConfigError(
            "人工审核 CSV 哈希不匹配："
            f"manifest={manifest.review_csv_sha256} actual={actual_hash}"
        )
    expected = {
        (case.case_id, str(turn_index))
        for case in cases
        for turn_index, _ in enumerate(case.turns)
    }
    with target.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual = {(row.get("case_id", ""), row.get("turn_index", "")) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ConfigError("人工审核 CSV 的样本/轮次与评测集不一致")
    rejected = [
        (row.get("case_id", ""), row.get("turn_index", ""))
        for row in rows
        if row.get("review_decision", "").strip().lower() != "approved"
    ]
    if rejected:
        raise ConfigError(f"人工审核 CSV 存在未批准条目：{len(rejected)}")


def write_review_csv(cases: list[EvalCase], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "kind",
                "turn_index",
                "question",
                "expected_refusal",
                "expected_answer",
                "expected_citations",
                "review_decision",
                "review_notes",
            ],
        )
        writer.writeheader()
        for case in cases:
            for turn_index, turn in enumerate(case.turns):
                writer.writerow(
                    {
                        "case_id": case.case_id,
                        "kind": case.kind,
                        "turn_index": turn_index,
                        "question": turn.question,
                        "expected_refusal": str(turn.expected_refusal).lower(),
                        "expected_answer": turn.expected_answer,
                        "expected_citations": " | ".join(
                            "/".join(citation.key()) for citation in turn.expected_citations
                        ),
                        "review_decision": "",
                        "review_notes": "",
                    }
                )
