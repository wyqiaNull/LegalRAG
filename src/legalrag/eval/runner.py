"""Run LegalRAG cases against isolated experiment profiles."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import BaseModel

from ..config.settings import Settings, load_settings
from ..core.errors import ConfigError
from ..core.models import DocType
from .dataset import dataset_sha256
from .metrics import evaluate_deterministic_metrics
from .models import EvalCase, EvalContext, EvalObservation, EvalRetrievalSignal
from .ragas_adapter import evaluate_with_ragas


class CorpusDocument(BaseModel):
    path: str
    doc_name: str
    version: str
    effective_date: str | None = None


class CorpusManifest(BaseModel):
    source: str
    source_version: str
    documents: list[CorpusDocument]


def load_corpus(path: str | Path) -> CorpusManifest:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return CorpusManifest.model_validate(raw)


def validate_gold_against_corpus(
    cases: list[EvalCase], corpus: CorpusManifest
) -> None:
    known = {(document.doc_name, document.version) for document in corpus.documents}
    referenced = {
        (citation.doc_name, citation.version)
        for case in cases
        for turn in case.turns
        for citation in turn.expected_citations
    }
    unknown = sorted(referenced - known)
    if unknown:
        raise ConfigError(f"gold 引用了 corpus 中不存在的文档版本：{unknown}")


def _file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cache_key(
    dataset_path: str | Path,
    config_path: str | Path,
    corpus_path: str | Path,
    settings: Settings,
) -> str:
    corpus = load_corpus(corpus_path)
    payload = {
        "dataset": dataset_sha256(dataset_path),
        "config": _file_hash(config_path),
        "corpus_manifest": _file_hash(corpus_path),
        "corpus_documents": {
            document.path: _file_hash(document.path) for document in corpus.documents
        },
        "embedding_api_base": settings.secrets.embedding_api_base,
        "embedding_model": settings.secrets.embedding_model,
        "rerank_api_base": settings.secrets.rerank_api_base,
        "rerank_model": settings.secrets.rerank_model,
        "llm_api_base": settings.secrets.llm_api_base,
        "llm_model": settings.secrets.llm_model,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_jsonl(path: Path, observations: list[EvalObservation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(item.model_dump_json() + "\n" for item in observations),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[EvalObservation]:
    return [
        EvalObservation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_metadata(
    *,
    key: str,
    profile_name: str,
    config_path: str | Path,
    dataset_path: str | Path,
    settings: Settings,
    status: str,
) -> dict:
    return {
        "cache_key": key,
        "status": status,
        "profile": profile_name,
        "config": str(config_path),
        "dataset_sha256": dataset_sha256(dataset_path),
        "models": {
            "embedding": settings.secrets.embedding_model,
            "reranker": settings.secrets.rerank_model,
            "generator": settings.secrets.llm_model,
        },
    }


def _write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepare_components(
    config_path: str | Path, corpus: CorpusManifest, workspace: Path
):
    from ..cli import Components

    settings = load_settings(str(config_path))
    settings.config.store.path = str(workspace)
    settings.config.store.metadata = "memory"
    settings.config.conversation.store = "json"
    settings.config.conversation.path = str(workspace / "conversations.json")
    components = Components(settings)
    pipeline = components.pipeline()
    for document in corpus.documents:
        pipeline.run(
            document.path,
            doc_name=document.doc_name,
            doc_type=DocType.REGULATION,
            tenant_id="__global__",
            version=document.version,
            effective_date=document.effective_date,
        )
    return components


def run_profile(
    *,
    profile_name: str,
    config_path: str | Path,
    cases: list[EvalCase],
    dataset_path: str | Path,
    corpus_path: str | Path,
    output_dir: str | Path,
    refresh: bool = False,
) -> list[EvalObservation]:
    output = Path(output_dir)
    observations_path = output / profile_name / "observations.jsonl"
    metadata_path = output / profile_name / "run.json"
    settings = load_settings(str(config_path))
    key = _cache_key(dataset_path, config_path, corpus_path, settings)
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not refresh and observations_path.exists() and metadata_path.exists():
        cached = _read_jsonl(observations_path)
        expected_count = sum(len(case.turns) for case in cases)
        if (
            metadata.get("cache_key") == key
            and metadata.get("status") == "complete"
            and len(cached) == expected_count
        ):
            return cached

    reusable = metadata.get("cache_key") == key and not refresh
    if observations_path.parent.exists() and not reusable:
        shutil.rmtree(observations_path.parent)
    observations = _read_jsonl(observations_path) if reusable and observations_path.exists() else []
    existing_by_case: dict[str, list[EvalObservation]] = {}
    for observation in observations:
        existing_by_case.setdefault(observation.case_id, []).append(observation)
    expected_turns = {case.case_id: len(case.turns) for case in cases}
    completed_cases = {
        case_id
        for case_id, items in existing_by_case.items()
        if len(items) == expected_turns.get(case_id)
        and {item.turn_index for item in items} == set(range(expected_turns[case_id]))
    }
    observations = [item for item in observations if item.case_id in completed_cases]
    _write_metadata(
        metadata_path,
        _run_metadata(
            key=key,
            profile_name=profile_name,
            config_path=config_path,
            dataset_path=dataset_path,
            settings=settings,
            status="running",
        ),
    )

    corpus = load_corpus(corpus_path)
    validate_gold_against_corpus(cases, corpus)
    with tempfile.TemporaryDirectory(prefix=f"legalrag-eval-{profile_name}-") as temp:
        components = _prepare_components(config_path, corpus, Path(temp))
        for case in cases:
            if case.case_id in completed_cases:
                continue
            case_observations = []
            session_id = f"eval:{profile_name}:{case.case_id}" if len(case.turns) > 1 else None
            for turn_index, turn in enumerate(case.turns):
                started = time.perf_counter()
                try:
                    result = components.query_orchestrator.run(
                        turn.question,
                        session_id=session_id,
                        identity=case.identity,
                    )
                    used = set(result.answer.retrieved_chunk_ids)
                    contexts = [
                        EvalContext(
                            chunk_id=candidate.chunk.chunk_id,
                            doc_name=candidate.chunk.doc_name,
                            version=candidate.chunk.version,
                            clause_no=candidate.chunk.clause_no,
                            content=candidate.chunk.content,
                            score=candidate.score,
                        )
                        for candidate in result.candidates
                        if candidate.chunk.chunk_id in used
                    ]
                    observation = EvalObservation(
                        case_id=case.case_id,
                        turn_index=turn_index,
                        profile=profile_name,
                        question=turn.question,
                        retrieval_question=result.retrieval_question,
                        retrieval_attempts=list(result.retrieval_attempts),
                        retrieval_signals=[
                            EvalRetrievalSignal(
                                question=signal.question,
                                candidate_count=signal.candidate_count,
                                top_score=signal.top_score,
                            )
                            for signal in result.retrieval_signals
                        ],
                        answer=result.answer.text,
                        refused=result.answer.refused,
                        reason=result.answer.reason,
                        route=result.answer.route,
                        contexts=contexts,
                        citations=result.answer.citations,
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )
                except Exception as exc:  # noqa: BLE001 - failures must be recorded in output
                    observation = EvalObservation(
                        case_id=case.case_id,
                        turn_index=turn_index,
                        profile=profile_name,
                        question=turn.question,
                        retrieval_question=turn.question,
                        answer="",
                        refused=False,
                        route="normal",
                        latency_ms=(time.perf_counter() - started) * 1000,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                case_observations.append(observation)
            observations.extend(case_observations)
            _write_jsonl(observations_path, observations)

    _write_jsonl(observations_path, observations)
    _write_metadata(
        metadata_path,
        _run_metadata(
            key=key,
            profile_name=profile_name,
            config_path=config_path,
            dataset_path=dataset_path,
            settings=settings,
            status="complete",
        ),
    )
    return observations


def summarize_profile(
    cases: list[EvalCase],
    observations: list[EvalObservation],
    *,
    settings: Settings,
    with_ragas: bool,
    ragas_evaluator: Callable = evaluate_with_ragas,
) -> dict:
    deterministic = evaluate_deterministic_metrics(cases, observations)
    summary = {"deterministic": deterministic.model_dump(mode="json")}
    if with_ragas:
        summary["ragas"] = ragas_evaluator(
            cases, observations, settings.secrets
        ).model_dump(mode="json")
    return summary
