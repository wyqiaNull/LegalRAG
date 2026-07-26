"""Milvus dense + BM25 向量存储实现。"""

from __future__ import annotations

import json
from typing import Any

from ..core import registry
from ..core.errors import ConfigError, StorageError
from ..core.interfaces import Candidate, Filter, VectorStore
from ..core.models import CandidateSource, Chunk

_SCALAR_FIELDS = {
    "chunk_id",
    "doc_id",
    "doc_name",
    "doc_type",
    "tenant_id",
    "department",
    "confidentiality",
    "version",
    "effective_date",
    "is_current",
    "superseded_by",
    "clause_no",
    "page",
    "content_hash",
}
_ARRAY_FIELDS = {"allowed_roles"}
_OUTPUT_FIELDS = [
    "chunk_id",
    "doc_id",
    "doc_name",
    "doc_type",
    "tenant_id",
    "department",
    "allowed_roles",
    "confidentiality",
    "version",
    "effective_date",
    "is_current",
    "superseded_by",
    "clause_no",
    "page",
    "content",
    "content_hash",
]


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise ConfigError(f"Milvus 过滤值类型不受支持：{type(value).__name__}")


def _list_literal(values: list[Any]) -> str:
    return "[" + ", ".join(_literal(value) for value in values) + "]"


def compile_filter(filters: Filter | None) -> str:
    """把内部过滤结构编译为白名单 Milvus 表达式。"""
    if not filters:
        return ""
    expressions: list[str] = []
    for field, wanted in filters.items():
        if field == "$or":
            if not isinstance(wanted, list) or not wanted:
                return 'chunk_id == "__deny_all__"'
            branches = [compile_filter(branch) for branch in wanted if isinstance(branch, dict)]
            if len(branches) != len(wanted) or any(not branch for branch in branches):
                return 'chunk_id == "__deny_all__"'
            expressions.append("(" + " or ".join(f"({branch})" for branch in branches) + ")")
            continue
        if field not in _SCALAR_FIELDS | _ARRAY_FIELDS:
            raise ConfigError(f"Milvus 过滤字段不受支持：{field!r}")
        values = wanted if isinstance(wanted, list) else [wanted]
        if not values:
            return 'chunk_id == "__deny_all__"'
        serialized = _list_literal(values)
        if field in _ARRAY_FIELDS:
            expressions.append(f"array_contains_any({field}, {serialized})")
        elif isinstance(wanted, list):
            expressions.append(f"{field} in {serialized}")
        else:
            expressions.append(f"{field} == {_literal(wanted)}")
    return " and ".join(expressions)


def _entity_for(chunk: Chunk, vector: list[float]) -> dict[str, Any]:
    data = chunk.model_dump(mode="json")
    data["dense"] = vector
    return data


def _candidate(hit: dict[str, Any], source: CandidateSource) -> Candidate:
    entity = dict(hit.get("entity") or {})
    entity.setdefault("chunk_id", str(hit.get("id", "")))
    return Candidate(
        chunk=Chunk(**entity),
        score=float(hit.get("distance", 0.0)),
        source=source,
    )


class MilvusVectorStore(VectorStore):
    def __init__(
        self,
        uri: str,
        token: str = "",
        collection_name: str = "legalrag_chunks",
        dim: int = 1024,
        client: Any | None = None,
    ) -> None:
        if not uri:
            raise ConfigError("Milvus 存储必须配置 MILVUS_URI")
        if dim < 1:
            raise ConfigError("Milvus dense 向量维度必须大于 0")
        self.collection_name = collection_name
        self.dim = dim
        if client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise ConfigError(
                    'Milvus 依赖未安装，请执行 uv pip install -e ".[service]"'
                ) from exc
            kwargs = {"uri": uri}
            if token:
                kwargs["token"] = token
            client = MilvusClient(**kwargs)
        self.client = client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            if self.client.has_collection(self.collection_name):
                self.client.load_collection(self.collection_name)
                return
            from pymilvus import DataType, Function, FunctionType, MilvusClient

            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
            schema.add_field("doc_name", DataType.VARCHAR, max_length=512)
            schema.add_field("doc_type", DataType.VARCHAR, max_length=64)
            schema.add_field("tenant_id", DataType.VARCHAR, max_length=256)
            schema.add_field("department", DataType.VARCHAR, max_length=256)
            schema.add_field(
                "allowed_roles",
                DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=64,
                max_length=128,
            )
            schema.add_field("confidentiality", DataType.VARCHAR, max_length=32)
            schema.add_field("version", DataType.VARCHAR, max_length=128)
            schema.add_field(
                "effective_date", DataType.VARCHAR, max_length=32, nullable=True
            )
            schema.add_field("is_current", DataType.BOOL)
            schema.add_field(
                "superseded_by", DataType.VARCHAR, max_length=128, nullable=True
            )
            schema.add_field("clause_no", DataType.VARCHAR, max_length=128)
            schema.add_field("page", DataType.INT64, nullable=True)
            schema.add_field(
                "content",
                DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params={"tokenizer": "jieba"},
            )
            schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
            schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(
                Function(
                    name="content_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["content"],
                    output_field_names=["sparse"],
                )
            )
            indexes = self.client.prepare_index_params()
            indexes.add_index(
                field_name="dense",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 128},
            )
            indexes.add_index(
                field_name="sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=indexes,
                consistency_level="Strong",
            )
            self.client.load_collection(self.collection_name)
        except ConfigError:
            raise
        except Exception as exc:
            raise StorageError(f"初始化 Milvus collection 失败：{exc}") from exc

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk 与向量数量必须一致")
        if not chunks:
            return
        if any(len(vector) != self.dim for vector in vectors):
            raise StorageError("Milvus dense 向量维度与配置不一致")
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                data=[_entity_for(chunk, vector) for chunk, vector in zip(chunks, vectors)],
            )
        except Exception as exc:
            raise StorageError(f"写入 Milvus 失败：{exc}") from exc

    def get_vectors(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        if not chunk_ids:
            return {}
        try:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter=f"chunk_id in {_list_literal(chunk_ids)}",
                output_fields=["chunk_id", "dense"],
                limit=len(chunk_ids),
            )
            return {row["chunk_id"]: list(row["dense"]) for row in rows}
        except Exception as exc:
            raise StorageError(f"读取 Milvus 向量失败：{exc}") from exc

    def search(
        self,
        vector: list[float],
        filters: Filter | None,
        top_n: int,
    ) -> list[Candidate]:
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[vector],
                anns_field="dense",
                filter=compile_filter(filters),
                limit=top_n,
                output_fields=_OUTPUT_FIELDS,
                search_params={"metric_type": "COSINE", "params": {"ef": max(64, top_n)}},
            )
            return [_candidate(hit, CandidateSource.DENSE) for hit in results[0]]
        except Exception as exc:
            raise StorageError(f"Milvus dense 检索失败：{exc}") from exc

    def search_sparse(
        self,
        query: str,
        filters: Filter | None,
        top_n: int,
    ) -> list[Candidate]:
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse",
                filter=compile_filter(filters),
                limit=top_n,
                output_fields=_OUTPUT_FIELDS,
                search_params={"metric_type": "BM25", "params": {}},
            )
            return [_candidate(hit, CandidateSource.SPARSE) for hit in results[0]]
        except Exception as exc:
            raise StorageError(f"Milvus BM25 检索失败：{exc}") from exc

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        try:
            self.client.delete(
                collection_name=self.collection_name,
                ids=chunk_ids,
            )
        except Exception as exc:
            raise StorageError(f"删除 Milvus 文档向量失败：{exc}") from exc

    def ping(self) -> bool:
        try:
            return self.collection_name in self.client.list_collections()
        except Exception:
            return False


registry.register("vector_store", "milvus", MilvusVectorStore)
