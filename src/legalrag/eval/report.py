"""Render a provenance-first v0.5 evaluation report."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.errors import ConfigError


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def _decimal(value: float | None) -> str:
    return "-" if value is None else f"{value:+.4f}"


def render_report(summary: dict, *, require_complete: bool) -> str:
    profiles = summary.get("profiles", {})
    hard = summary.get("hard_checks", {})
    failures = sum(
        item.get("deterministic", {}).get("failed_observations", 0)
        for item in profiles.values()
    )
    benchmarks = summary.get("benchmarks", {})
    ragas_complete = bool(profiles) and all(
        item.get("ragas", {}).get("failed_samples") == 0
        and item.get("ragas", {}).get("sample_count")
        == item.get("ragas", {}).get("expected_sample_count")
        for item in profiles.values()
    )
    benchmark_complete = (
        benchmarks.get("judge_calibrated") is True
        and benchmarks.get("RAGTruth", {}).get("status") == "passed"
        and benchmarks.get("RGB", {}).get("status") == "passed"
        and benchmarks.get("CRUD-RAG", {}).get("status") == "passed"
    )
    complete = (
        summary.get("mode") == "formal"
        and len(profiles) == 5
        and summary.get("review", {}).get("status") == "approved"
        and bool(summary.get("review", {}).get("review_csv_sha256"))
        and failures == 0
        and hard.get("exit_code") == 0
        and hard.get("real_services") is True
        and hard.get("unauthorized_candidate_count") == 0
        and hard.get("skipped", 0) == 0
        and ragas_complete
        and benchmark_complete
        and set(summary.get("ablations", {})) == {"chunking", "retrieval", "agentic"}
        and summary.get("threshold_calibration", {}).get("grid_size") == 63
    )
    if require_complete and not complete:
        raise ConfigError("当前结果不满足 v0.5 完整验收门禁，不能生成完成报告")

    status = "已完成" if complete else "草稿/待验收"
    lines = [
        "# LegalRAG v0.5 评估报告",
        "",
        f"> 状态：**{status}**。本报告只陈述实际运行结果，不用离线 mock 代替真实质量证据。",
        "",
        "## 运行信息",
        "",
        f"- Git：`{summary.get('git_sha', 'unknown')}`",
        f"- 数据 SHA256：`{summary.get('dataset_sha256', 'unknown')}`",
        f"- 审核状态：`{summary.get('review', {}).get('status', 'unknown')}`",
        f"- 审核 CSV SHA256：`{summary.get('review', {}).get('review_csv_sha256') or 'not-approved'}`",
        f"- 生成模型：`{summary.get('models', {}).get('generator', 'unknown')}`",
        f"- 独立裁判：`{summary.get('models', {}).get('judge', 'not-run')}`",
        f"- Embedding：`{summary.get('models', {}).get('embedding', 'unknown')}`",
        f"- Reranker：`{summary.get('models', {}).get('reranker', 'unknown')}`",
        "- 质量评测使用隔离的内存元数据与 JSON 会话存储；治理能力由独立硬功能及真实服务测试验证。",
        "",
        "## 六维指标",
        "",
        "| Profile | Context Precision | Context Recall* | Faithfulness | Answer Relevancy | Citation F1 | Refusal Macro-F1 | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in profiles.items():
        deterministic = item.get("deterministic", {})
        ragas = item.get("ragas", {})
        lines.append(
            f"| {name} | {_percent(ragas.get('context_precision'))} | "
            f"{_percent(deterministic.get('context_recall'))} | "
            f"{_percent(ragas.get('faithfulness'))} | "
            f"{_percent(ragas.get('answer_relevancy'))} | "
            f"{_percent(deterministic.get('citation_f1'))} | "
            f"{_percent(deterministic.get('refusal', {}).get('macro_f1'))} | "
            f"{deterministic.get('failed_observations', 0)} |"
        )
    lines.extend(
        [
            "",
            r"\* Context Recall 是按必需法规条款是否进入实际生成 context 计算的近似召回率，"
            "不是逐陈述完整标注的严格 recall。",
            "",
            "### 确定性指标明细",
            "",
            "| Profile | Citation P | Citation R | Citation F1 | Refusal Accuracy | Refusal Recall | False Refusal Rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in profiles.items():
        deterministic = item.get("deterministic", {})
        refusal = deterministic.get("refusal", {})
        lines.append(
            f"| {name} | {_percent(deterministic.get('citation_precision'))} | "
            f"{_percent(deterministic.get('citation_recall'))} | "
            f"{_percent(deterministic.get('citation_f1'))} | "
            f"{_percent(refusal.get('accuracy'))} | {_percent(refusal.get('recall'))} | "
            f"{_percent(refusal.get('false_refusal_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## 硬功能与红队",
            "",
            f"- 硬功能测试：{hard.get('passed', 0)}/{hard.get('tests', 0)} 通过；"
            f"失败 {hard.get('failures', 0)}，错误 {hard.get('errors', 0)}，跳过 {hard.get('skipped', 0)}。",
            f"- 真实 PostgreSQL/Redis：{'已运行' if hard.get('real_services') else '未运行'}。",
            f"- 越权候选数：{hard.get('unauthorized_candidate_count', '未验证')}。",
        ]
    )
    baseline = profiles.get(summary.get("baseline", ""), {})
    red_rate = baseline.get("deterministic", {}).get("red_team_attack_success_rate")
    lines.append(f"- 红队诱导成功率：{_percent(red_rate)}。")
    lines.extend(
        [
            "",
            "## 消融",
            "",
            "消融只报告同一数据上的配对方向性差异，不宣称统计显著。",
            "",
            "| Experiment | Baseline -> Candidate | Metric | Mean diff | W/T/L | Pairs |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for experiment, comparisons in summary.get("ablations", {}).items():
        for comparison in comparisons:
            lines.append(
                f"| {experiment} | {comparison.get('baseline')} -> "
                f"{comparison.get('candidate')} | {comparison.get('metric')} | "
                f"{_decimal(comparison.get('mean_difference'))} | "
                f"{comparison.get('wins', 0)}/{comparison.get('ties', 0)}/"
                f"{comparison.get('losses', 0)} | {comparison.get('sample_count', 0)} |"
            )
    threshold = summary.get("threshold_calibration", {})
    recommended = threshold.get("recommended", {})
    lines.extend(
        [
            "",
            "## 阈值校准",
            "",
            f"- 网格：`{threshold.get('grid_size', 0)}` 组；方法："
            f"`{threshold.get('method', 'not-run')}`。",
            f"- 推荐：`min_candidates={recommended.get('min_candidates', '-')}`、"
            f"`min_score={recommended.get('min_score', '-')}`；拒答 Macro-F1 "
            f"{_percent(recommended.get('macro_f1'))}，误拒率 "
            f"{_percent(recommended.get('false_refusal_rate'))}。",
            f"- 是否允许更新默认值：{'是' if threshold.get('apply_default') else '否'}。"
            f"原因：{threshold.get('reason', '未运行')}。",
            "",
            "## Benchmark 与来源",
            "",
        ]
    )
    for name in ("RAGTruth", "RGB", "CRUD-RAG"):
        item = benchmarks.get(name, {})
        detail = (
            f"，校准准确率 {_percent(item.get('accuracy'))}"
            if name == "RAGTruth" and item.get("accuracy") is not None
            else ""
        )
        lines.append(
            f"- {name}：{item.get('status', 'not-run')}，"
            f"revision `{item.get('revision', 'unknown')}`，"
            f"license `{item.get('license', 'unknown')}`，"
            f"样本 {item.get('sample_count', 0)}{detail}。"
        )
    lines.extend(
        [
            "- RAGTruth 仅用于独立裁判校准；RGB/CRUD-RAG 仅用于管线 sanity check，"
            "三者均不作为 LegalRAG 产品成绩。",
            "",
            "## 可复现性",
            "",
            f"- 依赖版本：`{json.dumps(summary.get('dependencies', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- 组件口径：`{json.dumps(summary.get('components', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- 配置哈希：`{json.dumps(summary.get('profile_configs', {}).get('config_sha256', {}), sort_keys=True)}`",
            f"- 控制变量：`{json.dumps(summary.get('profile_configs', {}).get('controlled_values', {}), ensure_ascii=False, sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    summary_path: str | Path,
    output_path: str | Path,
    *,
    require_complete: bool,
) -> None:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_report(summary, require_complete=require_complete), encoding="utf-8"
    )
