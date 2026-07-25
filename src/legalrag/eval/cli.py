"""Typer commands for the v0.5 evaluation harness."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import typer

from ..config.settings import load_settings
from ..core.errors import ConfigError
from .analysis import calibrate_thresholds, summarize_ablations
from .benchmarks import (
    benchmark_sanity,
    calibrate_ragtruth,
    load_benchmark_manifest,
    load_benchmark_samples,
    validate_benchmarks,
)
from .dataset import (
    dataset_sha256,
    load_cases,
    validate_minimum_scale,
    validate_review,
    write_review_csv,
)
from .hard_checks import run_hard_checks
from .matrix import matrix_evidence, validate_matrix
from .report import write_report
from .ragas_adapter import score_benchmark_faithfulness
from .runner import run_profile, summarize_profile

eval_app = typer.Typer(help="运行、校验并报告 v0.5 评估")


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() or "unknown"


def _dependency_versions() -> dict[str, str]:
    result = {}
    for package in ("legalrag", "ragas", "langchain-openai", "pydantic", "pytest"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _run_benchmarks(manifest_path: str, secrets) -> dict:
    validated = validate_benchmarks(manifest_path)
    manifest = load_benchmark_manifest(manifest_path)
    specs = {item.name: item for item in manifest.benchmarks}
    ragtruth = load_benchmark_samples(specs["RAGTruth"])
    scores = score_benchmark_faithfulness(ragtruth, secrets)
    calibration = calibrate_ragtruth(
        ragtruth,
        scores,
        threshold=manifest.ragtruth_faithfulness_threshold,
        min_accuracy=manifest.ragtruth_min_accuracy,
    )
    return {
        "judge_calibrated": calibration["judge_calibrated"],
        "RAGTruth": {**validated["RAGTruth"], **calibration},
        "RGB": {
            **validated["RGB"],
            **benchmark_sanity(
                load_benchmark_samples(specs["RGB"]), expected_dataset="RGB"
            ),
        },
        "CRUD-RAG": {
            **validated["CRUD-RAG"],
            **benchmark_sanity(
                load_benchmark_samples(specs["CRUD-RAG"]),
                expected_dataset="CRUD-RAG",
            ),
        },
    }


@eval_app.command("validate")
def validate_command(
    dataset: str = typer.Option("data/eval/legal_gold.jsonl", help="评测 JSONL"),
    review: str = typer.Option("data/eval/review.yaml", help="人工审核清单"),
    matrix: str = typer.Option("config/eval/matrix.yaml", help="消融矩阵"),
    benchmarks: str = typer.Option("data/eval/benchmarks.yaml", help="benchmark 来源清单"),
    formal: bool = typer.Option(False, help="要求 approved 审核状态"),
    review_csv: str | None = typer.Option(None, help="重新导出人工审核 CSV"),
) -> None:
    cases = load_cases(dataset)
    if formal:
        validate_minimum_scale(cases)
    manifest = validate_review(
        dataset,
        review,
        require_approved=formal,
        review_csv_path=str(Path(review).with_name("review.csv")),
    )
    validate_matrix(matrix)
    benchmark_manifest = load_benchmark_manifest(benchmarks)
    if formal:
        validate_benchmarks(benchmarks)
    if review_csv:
        write_review_csv(cases, review_csv)
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.kind] = counts.get(case.kind, 0) + 1
    typer.echo(
        f"评测集有效：cases={len(cases)} sha256={dataset_sha256(dataset)} "
        f"review={manifest.status} kinds={counts}"
    )
    typer.echo(
        "benchmark 来源清单有效："
        + ", ".join(
            f"{item.name}@{item.revision[:12]}({item.sample_count})"
            for item in benchmark_manifest.benchmarks
        )
    )


@eval_app.command("run")
def run_command(
    dataset: str = typer.Option("data/eval/legal_gold.jsonl"),
    review: str = typer.Option("data/eval/review.yaml"),
    matrix: str = typer.Option("config/eval/matrix.yaml"),
    corpus: str = typer.Option("data/eval/corpus.yaml"),
    benchmarks: str = typer.Option("data/eval/benchmarks.yaml"),
    output: str = typer.Option("artifacts/eval/latest"),
    mode: Literal["draft", "formal"] = typer.Option("draft"),
    suite: Literal["legal", "quality", "benchmarks", "hard", "all"] = typer.Option(
        "legal"
    ),
    profiles: str = typer.Option("", help="逗号分隔的 profile 名；空值运行全部"),
    limit: int | None = typer.Option(None, min=1, help="仅 draft smoke 使用的 case 上限"),
    refresh: bool = typer.Option(False),
) -> None:
    formal = mode == "formal"
    if formal and suite != "all":
        raise ConfigError("正式评测必须使用 --suite all，禁止跳过裁判和 benchmark")
    if formal and (limit is not None or profiles.strip()):
        raise ConfigError("正式评测禁止使用 --limit 或筛选 profile")
    cases = load_cases(dataset)
    if formal:
        validate_minimum_scale(cases)
    if limit is not None:
        cases = cases[:limit]
    review_manifest = validate_review(
        dataset,
        review,
        require_approved=formal,
        review_csv_path=str(Path(review).with_name("review.csv")),
    )
    eval_matrix = validate_matrix(matrix)
    profile_lookup = {profile.name: profile.config for profile in eval_matrix.profiles}
    selected_profiles = [item.strip() for item in profiles.split(",") if item.strip()]
    if selected_profiles:
        unknown = set(selected_profiles) - set(profile_lookup)
        if unknown:
            raise ConfigError(f"未知评测 profile：{sorted(unknown)}")
        profile_lookup = {name: profile_lookup[name] for name in selected_profiles}
    base_profile_name = next(iter(profile_lookup))
    base_profile_config = profile_lookup[base_profile_name]
    output_path = Path(output)
    profile_summaries: dict[str, dict] = {}
    profile_observations = {}
    benchmark_results = {
        "judge_calibrated": False,
        "RAGTruth": {"status": "not-run"},
        "RGB": {"status": "not-run"},
        "CRUD-RAG": {"status": "not-run"},
    }
    if suite in {"benchmarks", "all"}:
        benchmark_results = _run_benchmarks(
            benchmarks,
            load_settings(base_profile_config).secrets,
        )
        if formal and not benchmark_results["judge_calibrated"]:
            raise ConfigError("RAGTruth 裁判校准未达到最低准确率，正式评测中止")

    if suite in {"legal", "quality", "all"}:
        for profile_name, config_path in profile_lookup.items():
            typer.echo(f"运行 profile：{profile_name}")
            observations = run_profile(
                profile_name=profile_name,
                config_path=config_path,
                cases=cases,
                dataset_path=dataset,
                corpus_path=corpus,
                output_dir=output_path,
                refresh=refresh,
            )
            profile_observations[profile_name] = observations
            settings = load_settings(config_path)
            profile_summaries[profile_name] = summarize_profile(
                cases,
                observations,
                settings=settings,
                with_ragas=suite in {"quality", "all"},
            )
            if formal and profile_summaries[profile_name]["deterministic"]["failed_observations"]:
                raise ConfigError(f"profile {profile_name} 存在失败样本，正式评测中止")

    required_profiles = {item.name for item in eval_matrix.profiles}
    if set(profile_observations) == required_profiles:
        ablations = summarize_ablations(eval_matrix, cases, profile_observations)
        thresholds = calibrate_thresholds(
            cases, profile_observations["clause_hybrid_no_agentic"]
        )
    else:
        ablations = {}
        thresholds = {}

    hard = (
        run_hard_checks(output_path, include_real_services=formal)
        if suite in {"hard", "all"}
        else {
            "status": "not-run",
            "exit_code": None,
            "real_services": False,
            "unauthorized_candidate_count": None,
        }
    )
    if formal and hard["exit_code"] != 0:
        raise ConfigError("真实硬功能测试未全绿，正式评测中止")
    base_settings = load_settings(base_profile_config)
    summary = {
        "mode": mode,
        "suite": suite,
        "case_count": len(cases),
        "turn_count": sum(len(case.turns) for case in cases),
        "sample_limit": limit,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "dataset_sha256": dataset_sha256(dataset),
        "review": review_manifest.model_dump(mode="json"),
        "baseline": base_profile_name,
        "models": {
            "generator": base_settings.secrets.llm_model,
            "judge": base_settings.secrets.eval_llm_model if suite == "all" else "not-run",
            "embedding": base_settings.secrets.embedding_model,
            "reranker": base_settings.secrets.rerank_model,
        },
        "profiles": profile_summaries,
        "profile_configs": {
            "paths": {item.name: item.config for item in eval_matrix.profiles},
            **matrix_evidence(matrix),
        },
        "matrix": eval_matrix.model_dump(mode="json"),
        "ablations": ablations,
        "threshold_calibration": thresholds,
        "hard_checks": hard,
        "benchmarks": benchmark_results,
        "dependencies": _dependency_versions(),
        "components": {
            "quality_metadata_store": "memory (isolated temporary directory)",
            "quality_conversation_store": "json (isolated temporary directory)",
            "quality_models": "real API" if suite == "all" else "configured pipeline",
            "governance_services": "real PostgreSQL and Redis" if formal else "offline assertions",
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(f"评测完成：{summary_path}")


@eval_app.command("report")
def report_command(
    summary: str = typer.Option("artifacts/eval/latest/summary.json"),
    output: str = typer.Option("project/evaluation/v0.5-report.md"),
    complete: bool = typer.Option(False, help="强制完整验收门禁"),
) -> None:
    write_report(summary, output, require_complete=complete)
    typer.echo(f"报告已生成：{output}")


def register(parent: typer.Typer) -> None:
    parent.add_typer(eval_app, name="eval")
