"""Full open-benchmark loading and resumable evaluation."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from statistics import fmean

from ..config.settings import Secrets
from ..core.errors import ConfigError
from ..llm.openai import OpenAIClient
from .benchmarks import BenchmarkSample, calibrate_ragtruth
from .ragas_adapter import score_benchmark_faithfulness

DEFAULT_OPEN_BENCH_ROOT = Path("artifacts/eval/benchmarks/raw/repos")


@dataclass(frozen=True)
class GenerationSample:
    sample_id: str
    dataset: str
    task: str
    prompt: str
    reference: str


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise ConfigError(f"缺少完整 benchmark 文件：{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _json(path: Path):
    if not path.exists():
        raise ConfigError(f"缺少完整 benchmark 文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_full_ragtruth(root: str | Path = DEFAULT_OPEN_BENCH_ROOT) -> list[BenchmarkSample]:
    dataset = Path(root) / "RAGTruth" / "dataset"
    sources = {row["source_id"]: row for row in _jsonl(dataset / "source_info.jsonl")}
    samples = []
    for row in _jsonl(dataset / "response.jsonl"):
        source = sources.get(row["source_id"])
        if (
            source is None
            or source.get("task_type") != "QA"
            or row.get("split") != "test"
            or row.get("quality") != "good"
        ):
            continue
        samples.append(
            BenchmarkSample(
                sample_id=str(row["id"]),
                dataset="RAGTruth",
                question=source["prompt"],
                contexts=[json.dumps(source["source_info"], ensure_ascii=False, sort_keys=True)],
                response=row["response"],
                expected_faithful=not bool(row.get("labels")),
            )
        )
    return samples


_RGB_SYSTEM = (
    "你是一个准确和可靠的人工智能助手，能够借助外部文档回答问题。外部文档可能包含噪声或"
    "事实性错误。如果文档包含正确答案，请准确回答；如果信息不足，请回答“文档信息不足，因此"
    "我无法基于提供的文档回答该问题。”；如果文档存在事实性错误，请先指出，再给出正确答案。"
)


def _rgb_prompt(question: str, documents: list[str]) -> str:
    context = "\n".join(f"[{index}] {doc}" for index, doc in enumerate(documents, start=1))
    return f"{_RGB_SYSTEM}\n\n文档：\n{context}\n\n问题：\n{question}"


def _rgb_reference(value) -> str:
    if not isinstance(value, list):
        return str(value)
    flattened = []
    for item in value:
        if isinstance(item, list):
            flattened.extend(str(value) for value in item)
        else:
            flattened.append(str(item))
    return " | ".join(flattened)


def load_full_rgb(root: str | Path = DEFAULT_OPEN_BENCH_ROOT) -> list[GenerationSample]:
    data = Path(root) / "RGB" / "data"
    samples = []
    for row in _jsonl(data / "zh_refine.json"):
        positive = list(row["positive"])
        negative = list(row["negative"])
        mixed = positive[:2] + negative[:3]
        samples.append(
            GenerationSample(
                sample_id=f"noise-60:{row['id']}",
                dataset="RGB",
                task="noise_60",
                prompt=_rgb_prompt(row["query"], mixed),
                reference=_rgb_reference(row["answer"]),
            )
        )
        samples.append(
            GenerationSample(
                sample_id=f"rejection:{row['id']}",
                dataset="RGB",
                task="rejection",
                prompt=_rgb_prompt(row["query"], negative[:5]),
                reference="文档信息不足",
            )
        )
    for row in _jsonl(data / "zh_int.json"):
        documents = [group[0] for group in row["positive"] if group][:5]
        samples.append(
            GenerationSample(
                sample_id=f"integration:{row['id']}",
                dataset="RGB",
                task="integration",
                prompt=_rgb_prompt(row["query"], documents),
                reference=_rgb_reference(row["answer"]),
            )
        )
    for row in _jsonl(data / "zh_fact.json"):
        samples.append(
            GenerationSample(
                sample_id=f"counterfactual:{row['id']}",
                dataset="RGB",
                task="counterfactual",
                prompt=_rgb_prompt(row["query"], list(row["positive_wrong"])[:5]),
                reference=_rgb_reference(row["answer"]),
            )
        )
    return samples


def _crud_context(row: dict, task: str) -> str:
    if task == "event_summary":
        return row["text"]
    if task == "continuing_writing":
        return row["summary"]
    if task == "hallu_modified":
        return row["newsRemainder"]
    return "\n\n".join(row[key] for key in ("news1", "news2", "news3") if key in row)


def _crud_prompt(row: dict, task: str, context: str) -> tuple[str, str]:
    if task == "event_summary":
        return (
            f"根据检索文档概括新闻事件“{row['event']}”。\n\n检索文档：\n{context}",
            row["summary"],
        )
    if task == "continuing_writing":
        return (
            "根据检索文档续写新闻，长度与开头大致相当，不要重复开头。"
            f"\n\n检索文档：\n{context}\n\n新闻开头：\n{row['beginning']}",
            row["continuing"],
        )
    if task == "hallu_modified":
        return (
            "根据检索文档纠正幻觉续写，不要引入无关信息。"
            f"\n\n新闻开头：\n{row['newsBeginning']}"
            f"\n\n幻觉续写：\n{row['hallucinatedContinuation']}"
            f"\n\n检索文档：\n{context}",
            row["hallucinatedMod"],
        )
    return (
        f"严格根据检索文档回答问题。\n\n问题：\n{row['questions']}\n\n检索文档：\n{context}",
        row["answers"],
    )


def load_full_crud_rag(root: str | Path = DEFAULT_OPEN_BENCH_ROOT) -> list[GenerationSample]:
    path = Path(root) / "CRUD_RAG" / "data" / "crud_split" / "split_merged.json"
    tasks = _json(path)
    samples = []
    for task, rows in tasks.items():
        for row in rows:
            context = _crud_context(row, task)
            prompt, reference = _crud_prompt(row, task, context)
            samples.append(
                GenerationSample(
                    sample_id=f"{task}:{row['ID']}",
                    dataset="CRUD-RAG",
                    task=task,
                    prompt=prompt,
                    reference=reference,
                )
            )
    return samples


def open_benchmark_plan(root: str | Path = DEFAULT_OPEN_BENCH_ROOT) -> dict:
    ragtruth = load_full_ragtruth(root)
    rgb = load_full_rgb(root)
    crud = load_full_crud_rag(root)
    return {
        "RAGTruth": {
            "samples": len(ragtruth),
            "tasks": {"faithfulness_judge": len(ragtruth)},
            "api": "judge",
            "calls_at_least": len(ragtruth),
        },
        "RGB": {
            "samples": len(rgb),
            "tasks": dict(sorted(Counter(sample.task for sample in rgb).items())),
            "api": "generator",
            "calls_at_least": len(rgb),
        },
        "CRUD-RAG": {
            "samples": len(crud),
            "tasks": dict(sorted(Counter(sample.task for sample in crud).items())),
            "api": "generator",
            "calls_at_least": len(crud),
        },
        "total_calls_at_least": len(ragtruth) + len(rgb) + len(crud),
    }


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _rouge_l_f1(prediction: str, reference: str) -> float:
    left = _normalize(prediction)
    right = _normalize(reference)
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_char == right_char
                else max(previous[index], current[-1])
            )
        previous = current
    common = previous[-1]
    precision = common / len(left)
    recall = common / len(right)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_completed(path: Path, *required_fields: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = _jsonl(path)
    return {
        row["sample_id"]: row
        for row in rows
        if not row.get("error") and all(field in row for field in required_fields)
    }


def _error_text(exc: Exception) -> str:
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _fatal_api_error(row: dict) -> bool:
    error = row.get("error", "")
    return any(marker in error for marker in ("401", "402", "Insufficient Balance"))


def _evaluate_in_batches(
    pending,
    workers: int,
    evaluate: Callable,
    observations: Path,
) -> str:
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate, sample) for sample in islice(iterator, workers)}
        while futures:
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                row = future.result()
                _append_jsonl(observations, row)
                if _fatal_api_error(row):
                    for pending_future in futures:
                        pending_future.cancel()
                    return row["error"]
                try:
                    sample = next(iterator)
                except StopIteration:
                    continue
                futures.add(executor.submit(evaluate, sample))
    return ""


def _summary(rows: list[dict], expected: int, stopped_reason: str = "") -> dict:
    successful = [row for row in rows if not row.get("error")]
    by_task = {}
    for task in sorted({row["task"] for row in successful}):
        selected = [row for row in successful if row["task"] == task]
        task_summary = {
            "samples": len(selected),
            "exact_match": fmean(row["exact_match"] for row in selected),
            "rouge_l_f1": fmean(row["rouge_l_f1"] for row in selected),
        }
        if task == "rejection":
            task_summary["rejection_rate"] = fmean(
                row["refusal_detected"] for row in selected
            )
        if task == "counterfactual":
            task_summary["error_detection_rate"] = fmean(
                row["factual_error_detected"] for row in selected
            )
        by_task[task] = task_summary
    dataset = successful[0]["dataset"] if successful else "unknown"
    return {
        "dataset": dataset,
        "evaluation_scope": (
            "provided-context generation; not LegalRAG corpus retrieval"
            if dataset in {"RGB", "CRUD-RAG"}
            else "unknown"
        ),
        "metrics_scope": "normalized answer containment and character-level ROUGE-L F1",
        "expected_samples": expected,
        "successful_samples": len(successful),
        "failed_samples": expected - len(successful),
        "observed_failures": len(rows) - len(successful),
        "stopped_reason": stopped_reason,
        "tasks": by_task,
    }


def run_generation_benchmark(
    samples: list[GenerationSample],
    secrets: Secrets,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    refresh: bool = False,
    workers: int = 1,
    complete: Callable[[str], str] | None = None,
) -> dict:
    selected = samples[:limit] if limit is not None else samples
    output = Path(output_dir)
    observations = output / "observations.jsonl"
    summary_path = output / "summary.json"
    if refresh:
        observations.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    completed = _load_completed(observations, "exact_match", "rouge_l_f1")
    if complete is None:
        client = OpenAIClient(
            api_base=secrets.llm_api_base,
            api_key=secrets.llm_api_key,
            model=secrets.llm_model,
            timeout=120.0,
        )

        def call_generator(prompt: str) -> str:
            return client.complete(prompt, temperature=0.0)

        complete = call_generator
    pending = [sample for sample in selected if sample.sample_id not in completed]

    def evaluate_sample(sample: GenerationSample) -> dict:
        try:
            response = complete(sample.prompt)
            normalized_response = _normalize(response)
            references = [item for item in sample.reference.split(" | ") if item]
            exact = float(any(_normalize(item) in normalized_response for item in references))
            row = {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "task": sample.task,
                "response": response,
                "reference": sample.reference,
                "exact_match": exact,
                "rouge_l_f1": max(_rouge_l_f1(response, item) for item in references),
                "refusal_detected": float("信息不足" in normalized_response),
                "factual_error_detected": float("事实性错误" in normalized_response),
                "error": "",
            }
        except Exception as exc:  # Keep the checkpoint usable across transient API failures.
            row = {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "task": sample.task,
                "error": _error_text(exc),
            }
        return row

    stopped_reason = _evaluate_in_batches(pending, workers, evaluate_sample, observations)
    rows = _jsonl(observations) if observations.exists() else []
    latest = {row["sample_id"]: row for row in rows}
    summary = _summary(
        [latest[sample.sample_id] for sample in selected if sample.sample_id in latest],
        len(selected),
        stopped_reason,
    )
    output.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_full_ragtruth(
    samples: list[BenchmarkSample],
    secrets: Secrets,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    refresh: bool = False,
    workers: int = 1,
    score: Callable[[list[BenchmarkSample], Secrets], list[float]] = score_benchmark_faithfulness,
) -> dict:
    selected = samples[:limit] if limit is not None else samples
    output = Path(output_dir)
    observations = output / "observations.jsonl"
    summary_path = output / "summary.json"
    if refresh:
        observations.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    completed = _load_completed(observations, "score")
    pending = [sample for sample in selected if sample.sample_id not in completed]

    def evaluate_sample(sample: BenchmarkSample) -> dict:
        try:
            value = score([sample], secrets)[0]
            row = {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "task": "faithfulness_judge",
                "score": value,
                "expected_faithful": sample.expected_faithful,
                "error": "",
            }
        except Exception as exc:
            row = {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "task": "faithfulness_judge",
                "error": _error_text(exc),
            }
        return row

    stopped_reason = _evaluate_in_batches(pending, workers, evaluate_sample, observations)
    rows = _jsonl(observations) if observations.exists() else []
    latest = {row["sample_id"]: row for row in rows}
    successful = [
        latest[sample.sample_id]
        for sample in selected
        if "score" in latest.get(sample.sample_id, {})
        and not latest[sample.sample_id].get("error")
    ]
    calibration = (
        calibrate_ragtruth(
            [
                sample
                for sample in selected
                if sample.sample_id in {row["sample_id"] for row in successful}
            ],
            [row["score"] for row in successful],
            threshold=0.8,
            min_accuracy=0.8,
        )
        if successful
        else {"status": "failed", "judge_calibrated": False}
    )
    summary = {
        "dataset": "RAGTruth",
        "evaluation_scope": "RAGAS faithfulness judge calibration on official QA test/good responses",
        "expected_samples": len(selected),
        "successful_samples": len(successful),
        "failed_samples": len(selected) - len(successful),
        "observed_failures": sum(
            bool(row.get("error")) or "score" not in row for row in latest.values()
        ),
        "stopped_reason": stopped_reason,
        **calibration,
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
