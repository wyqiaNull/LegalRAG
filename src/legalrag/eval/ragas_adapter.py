"""Lazy RAGAS integration for the three LLM-judged metrics."""

from __future__ import annotations

from statistics import fmean
from typing import Any

from pydantic import BaseModel

from ..config.settings import Secrets
from ..core.errors import ConfigError
from .benchmarks import BenchmarkSample
from .models import EvalCase, EvalObservation


class RagasResult(BaseModel):
    sample_count: int
    expected_sample_count: int
    failed_samples: int
    context_precision: float
    faithfulness: float
    answer_relevancy: float
    per_sample: list[dict[str, Any]]


def _metric_value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if row.get(key) is not None:
            return float(row[key])
    return None


def _mean(rows: list[dict[str, Any]], *keys: str) -> float:
    values = [_metric_value(row, *keys) for row in rows]
    if len(values) != len(rows):
        raise RuntimeError(f"RAGAS 指标 {keys[0]} 存在缺失值")
    if any(value is None for value in values):
        raise RuntimeError(f"RAGAS 指标 {keys[0]} 存在缺失值")
    return fmean(value for value in values if value is not None) if values else 0.0


def _validate_secrets(secrets: Secrets, *, require_embeddings: bool) -> None:
    if not all(
        [secrets.eval_llm_api_base, secrets.eval_llm_api_key, secrets.eval_llm_model]
    ):
        raise ConfigError("正式 RAGAS 评测必须配置独立的 EVAL_LLM_API_BASE/KEY/MODEL")
    if (
        secrets.eval_llm_api_base.rstrip("/") == secrets.llm_api_base.rstrip("/")
        and secrets.eval_llm_model == secrets.llm_model
    ):
        raise ConfigError("独立裁判不能与生成模型使用相同的 API endpoint 和 model")
    if require_embeddings and not all(
        [secrets.embedding_api_base, secrets.embedding_api_key, secrets.embedding_model]
    ):
        raise ConfigError("Answer Relevancy 需要配置 EMBEDDING_API_BASE/KEY/MODEL")


def _ragas_components(secrets: Secrets, *, with_embeddings: bool):
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise ConfigError('RAGAS 依赖未安装，请执行 uv pip install -e ".[dev,eval]"') from exc

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            base_url=secrets.eval_llm_api_base,
            api_key=secrets.eval_llm_api_key,
            model=secrets.eval_llm_model,
            temperature=0.0,
        ),
        bypass_n=True,
    )
    embeddings = None
    if with_embeddings:
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                base_url=secrets.embedding_api_base,
                api_key=secrets.embedding_api_key,
                model=secrets.embedding_model,
            )
        )
    return judge, embeddings


def evaluate_with_ragas(
    cases: list[EvalCase], observations: list[EvalObservation], secrets: Secrets
) -> RagasResult:
    _validate_secrets(secrets, require_embeddings=True)
    expected = {
        (case.case_id, turn_index)
        for case in cases
        for turn_index, turn in enumerate(case.turns)
        if not turn.expected_refusal
    }
    actual = {(item.case_id, item.turn_index): item for item in observations}
    failed = [
        key
        for key in expected
        if key not in actual or actual[key].error or actual[key].refused
    ]
    if failed:
        raise RuntimeError(f"RAGAS 待评分正常样本缺失、失败或被误拒：{len(failed)}")
    usable = [actual[key] for key in sorted(expected)]
    if not usable:
        raise ConfigError("没有可供 RAGAS 评分的正常回答")

    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise ConfigError('RAGAS 依赖未安装，请执行 uv pip install -e ".[dev,eval]"') from exc

    judge, embeddings = _ragas_components(secrets, with_embeddings=True)
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=item.question,
                response=item.answer,
                retrieved_contexts=[context.content for context in item.contexts],
            )
            for item in usable
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextPrecisionWithoutReference(llm=judge),
            Faithfulness(llm=judge),
            ResponseRelevancy(llm=judge, embeddings=embeddings),
        ],
        raise_exceptions=True,
    )
    rows = []
    for item, raw_row in zip(usable, result.scores, strict=True):
        row = dict(raw_row)
        row["case_id"] = item.case_id
        row["turn_index"] = item.turn_index
        rows.append(row)
    return RagasResult(
        sample_count=len(rows),
        expected_sample_count=len(expected),
        failed_samples=0,
        context_precision=_mean(
            rows,
            "llm_context_precision_without_reference",
            "context_precision_without_reference",
        ),
        faithfulness=_mean(rows, "faithfulness"),
        answer_relevancy=_mean(rows, "answer_relevancy", "response_relevancy"),
        per_sample=rows,
    )


def score_benchmark_faithfulness(
    samples: list[BenchmarkSample], secrets: Secrets
) -> list[float]:
    _validate_secrets(secrets, require_embeddings=False)
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import Faithfulness
    except ImportError as exc:
        raise ConfigError('RAGAS 依赖未安装，请执行 uv pip install -e ".[dev,eval]"') from exc
    judge, _ = _ragas_components(secrets, with_embeddings=False)
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=sample.question,
                response=sample.response,
                retrieved_contexts=sample.contexts,
            )
            for sample in samples
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(llm=judge)],
        raise_exceptions=True,
    )
    rows = [dict(row) for row in result.scores]
    scores = [_metric_value(row, "faithfulness") for row in rows]
    if len(scores) != len(samples) or any(score is None for score in scores):
        raise RuntimeError("RAGTruth faithfulness 评分存在缺失")
    return [score for score in scores if score is not None]
