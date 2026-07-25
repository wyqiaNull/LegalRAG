"""Pinned open benchmark preparation, validation and judge calibration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel, Field, model_validator

from ..core.errors import ConfigError


class BenchmarkSource(BaseModel):
    url: str
    sha256: str


class BenchmarkSpec(BaseModel):
    name: str
    purpose: str
    official_url: str
    revision: str
    license: str
    license_url: str
    sample_count: int = Field(gt=0)
    sample_ids: list[str]
    sources: list[BenchmarkSource] = Field(min_length=1)
    local_path: str
    normalized_sha256: str

    @model_validator(mode="after")
    def sample_ids_match_count(self) -> "BenchmarkSpec":
        if len(self.sample_ids) != self.sample_count:
            raise ValueError("sample_ids 数量必须等于 sample_count")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("sample_ids 不能重复")
        return self


class BenchmarkManifest(BaseModel):
    benchmarks: list[BenchmarkSpec]
    ragtruth_faithfulness_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    ragtruth_min_accuracy: float = Field(default=0.8, ge=0.0, le=1.0)


class BenchmarkSample(BaseModel):
    sample_id: str
    dataset: str
    question: str
    contexts: list[str] = Field(default_factory=list)
    response: str
    expected_faithful: bool | None = None
    expected_refusal: bool | None = None
    reference_answer: str = ""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        return BenchmarkManifest.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"benchmark 来源清单无效：{exc}") from exc


def _download(url: str) -> bytes:
    response = httpx.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    return response.content


def _jsonl(content: bytes) -> list[dict]:
    return [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]


def _as_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _extract_ragtruth(spec: BenchmarkSpec, contents: list[bytes]) -> list[BenchmarkSample]:
    sources = {item["source_id"]: item for item in _jsonl(contents[0])}
    responses = {str(item["id"]): item for item in _jsonl(contents[1])}
    samples = []
    for sample_id in spec.sample_ids:
        response = responses.get(sample_id)
        if response is None:
            raise ConfigError(f"RAGTruth 缺少固定样本：{sample_id}")
        source = sources.get(response["source_id"])
        if source is None or source.get("task_type") != "QA":
            raise ConfigError(f"RAGTruth 样本不是可校准的 QA：{sample_id}")
        if response.get("split") != "test" or response.get("quality") != "good":
            raise ConfigError(f"RAGTruth 样本不属于 test/good：{sample_id}")
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                dataset=spec.name,
                question=source["prompt"],
                contexts=[_as_text(source["source_info"])],
                response=response["response"],
                expected_faithful=not bool(response.get("labels")),
            )
        )
    return samples


def _extract_rgb(spec: BenchmarkSpec, contents: list[bytes]) -> list[BenchmarkSample]:
    rows = {str(item["id"]): item for item in _jsonl(contents[0])}
    samples = []
    for sample_id in spec.sample_ids:
        row = rows.get(sample_id)
        if row is None or not row.get("negative"):
            raise ConfigError(f"RGB 缺少固定负例样本：{sample_id}")
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                dataset=spec.name,
                question=row["query"],
                contexts=row["negative"][:5],
                response="检索材料不足以回答该问题。",
                expected_refusal=True,
                reference_answer=_as_text(row.get("answer", [])),
            )
        )
    return samples


def _extract_crud(spec: BenchmarkSpec, contents: list[bytes]) -> list[BenchmarkSample]:
    rows = json.loads(contents[0].decode("utf-8"))
    samples = []
    for sample_id in spec.sample_ids:
        record_id, raw_index = sample_id.rsplit(":", 1)
        row = rows.get(record_id)
        index = int(raw_index)
        if row is None or index >= len(row.get("question", [])) or index >= len(row.get("answers", [])):
            raise ConfigError(f"CRUD-RAG 缺少固定问答样本：{sample_id}")
        answer = row["answers"][index]
        if not answer or answer == "无法推断":
            raise ConfigError(f"CRUD-RAG 固定样本没有有效答案：{sample_id}")
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                dataset=spec.name,
                question=row["question"][index],
                contexts=list(row.get("key_info", [])),
                response=answer,
                reference_answer=answer,
            )
        )
    return samples


_EXTRACTORS = {
    "RAGTruth": _extract_ragtruth,
    "RGB": _extract_rgb,
    "CRUD-RAG": _extract_crud,
}


def _serialize_samples(samples: list[BenchmarkSample]) -> bytes:
    return "".join(sample.model_dump_json() + "\n" for sample in samples).encode("utf-8")


def prepare_benchmarks(
    manifest_path: str | Path,
    *,
    fetch: Callable[[str], bytes] = _download,
) -> dict[str, dict]:
    manifest = load_benchmark_manifest(manifest_path)
    result = {}
    for spec in manifest.benchmarks:
        extractor = _EXTRACTORS.get(spec.name)
        if extractor is None:
            raise ConfigError(f"不支持的 benchmark：{spec.name}")
        contents = []
        source_hashes = {}
        for source in spec.sources:
            content = fetch(source.url)
            actual = _sha256(content)
            if actual != source.sha256:
                raise ConfigError(
                    f"{spec.name} 官方源哈希不匹配：url={source.url} "
                    f"expected={source.sha256} actual={actual}"
                )
            contents.append(content)
            source_hashes[source.url] = actual
        samples = extractor(spec, contents)
        payload = _serialize_samples(samples)
        normalized_hash = _sha256(payload)
        if normalized_hash != spec.normalized_sha256:
            raise ConfigError(
                f"{spec.name} 固定抽样哈希不匹配："
                f"expected={spec.normalized_sha256} actual={normalized_hash}"
            )
        target = Path(spec.local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        result[spec.name] = {
            "status": "prepared",
            "revision": spec.revision,
            "license": spec.license,
            "sample_count": len(samples),
            "sample_ids": spec.sample_ids,
            "normalized_sha256": normalized_hash,
            "source_sha256": source_hashes,
        }
    return result


def load_benchmark_samples(spec: BenchmarkSpec) -> list[BenchmarkSample]:
    target = Path(spec.local_path)
    if not target.exists():
        raise ConfigError(f"benchmark 尚未准备：{spec.name} ({target})")
    content = target.read_bytes()
    actual_hash = _sha256(content)
    if actual_hash != spec.normalized_sha256:
        raise ConfigError(
            f"benchmark 文件哈希不匹配：{spec.name} "
            f"expected={spec.normalized_sha256} actual={actual_hash}"
        )
    samples = [BenchmarkSample.model_validate(item) for item in _jsonl(content)]
    ids = [sample.sample_id for sample in samples]
    if ids != spec.sample_ids or len(samples) != spec.sample_count:
        raise ConfigError(f"benchmark 固定样本清单不匹配：{spec.name}")
    return samples


def validate_benchmarks(manifest_path: str | Path) -> dict[str, dict]:
    manifest = load_benchmark_manifest(manifest_path)
    result = {}
    for spec in manifest.benchmarks:
        samples = load_benchmark_samples(spec)
        result[spec.name] = {
            "status": "valid",
            "purpose": spec.purpose,
            "revision": spec.revision,
            "license": spec.license,
            "sample_count": len(samples),
            "normalized_sha256": spec.normalized_sha256,
        }
    return result


def calibrate_ragtruth(
    samples: list[BenchmarkSample],
    scores: list[float],
    *,
    threshold: float,
    min_accuracy: float,
) -> dict:
    if len(samples) != len(scores) or not samples:
        raise ValueError("RAGTruth 样本与评分必须非空且一一对应")
    tp = tn = fp = fn = 0
    for sample, score in zip(samples, scores, strict=True):
        if sample.expected_faithful is None:
            raise ValueError(f"RAGTruth 样本缺少人工标签：{sample.sample_id}")
        predicted = score >= threshold
        if sample.expected_faithful and predicted:
            tp += 1
        elif sample.expected_faithful:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    accuracy = (tp + tn) / len(samples)
    return {
        "status": "passed" if accuracy >= min_accuracy else "failed",
        "sample_count": len(samples),
        "faithfulness_threshold": threshold,
        "minimum_accuracy": min_accuracy,
        "accuracy": accuracy,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "judge_calibrated": accuracy >= min_accuracy,
    }


def benchmark_sanity(samples: list[BenchmarkSample], *, expected_dataset: str) -> dict:
    if not samples or any(sample.dataset != expected_dataset for sample in samples):
        raise ConfigError(f"{expected_dataset} sanity check 样本无效")
    if expected_dataset == "RGB" and any(sample.expected_refusal is not True for sample in samples):
        raise ConfigError("RGB sanity check 必须全部为负例拒答样本")
    if expected_dataset == "CRUD-RAG" and any(not sample.reference_answer for sample in samples):
        raise ConfigError("CRUD-RAG sanity check 必须包含参考答案")
    return {"status": "passed", "sample_count": len(samples), "scope": "pipeline_sanity_only"}
