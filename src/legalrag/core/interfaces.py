"""抽象接口（SPEC §4.2）—— 所有签名 MVP 定稿，实现按版本后补。

预留扩展点（哲学四）：
- ``Retriever.search`` 恒带 ``filters``（MVP 传空 dict）—— v0.2 加权限过滤时调用方零改动；
- ``Embedder.embed`` 恒返 ``EmbedResult(dense, sparse)``（MVP sparse=None）—— v0.1 启用稀疏时下游零改动。

能力层实现均通过 ``core.registry`` 注入，互不直接 import（哲学五）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .models import Answer, Candidate, Chunk, Identity, Query, Route

# 元数据过滤条件：字段名 -> 约束值。MVP 传空 dict，v0.2 起塞 tenant/密级/is_current 等。
Filter = dict[str, Any]


@dataclass
class RawDoc:
    """加载后的原始文档：按页保存文本，供分块器切分并回填页码。

    ``pages`` 下标 0 对应第 1 页；无分页概念的格式（txt/docx）整篇作为单页。
    """

    doc_name: str
    pages: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.pages)


@dataclass
class EmbedResult:
    """向量化结果。dense 恒有；sparse 预留，MVP 返 None，v0.1 启用。"""

    dense: list[list[float]]
    sparse: list[dict[int, float]] | None = None


# ============ 能力层抽象（MVP 实现）============


class Loader(ABC):
    """文档加载器：path -> RawDoc（纯文本 + 页码）。"""

    @abstractmethod
    def load(self, path: str) -> RawDoc: ...


class Chunker(ABC):
    """分块器：原始文档 + 基础元数据 -> 一组 Chunk。"""

    @abstractmethod
    def split(self, raw: RawDoc, meta: dict[str, Any]) -> list[Chunk]: ...


class Embedder(ABC):
    """向量化：一批文本 -> EmbedResult(dense, sparse|None)。"""

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbedResult: ...


class VectorStore(ABC):
    """向量库：标量过滤 + ANN 检索一体（接口对齐 Milvus，MVP 用内存实现）。"""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    @abstractmethod
    def search(
        self, vector: list[float], filters: Filter | None, top_n: int
    ) -> list[Candidate]:
        """在 ``filters`` 圈定的子集内做 ANN。filters=None/空 表示全库（MVP）。"""
        ...


class MetadataStore(ABC):
    """元数据库：保存 chunk 元数据、提供 ACL 策略（ACL 方法 v0.2 才有实体）。"""

    @abstractmethod
    def save(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def get_acl(self, role: str) -> Any:
        """角色 -> ACL 策略。MVP 空实现（返回 None），v0.2 接 acl_policies。"""
        ...


class Retriever(ABC):
    """检索器：query -> 候选集。filters 参数恒在，MVP 传空。"""

    @abstractmethod
    def search(
        self, query: Query, filters: Filter | None, top_k: int
    ) -> list[Candidate]: ...


class Reranker(ABC):
    """精排器：对候选集重排序并截断到 top_k。MVP 用 Noop 占位。"""

    @abstractmethod
    def rerank(
        self, query: Query, candidates: list[Candidate], top_k: int
    ) -> list[Candidate]: ...


class Generator(ABC):
    """生成器：query + 上下文候选 -> Answer。"""

    @abstractmethod
    def generate(self, query: Query, contexts: list[Candidate]) -> Answer: ...


class LLMClient(ABC):
    """LLM 客户端：prompt -> 文本。opts 透传温度/最大 token 等。"""

    @abstractmethod
    def complete(self, prompt: str, **opts: Any) -> str: ...


# ============ 未来版本抽象（签名先定，实现后补）============


class PermissionFilter(ABC):  # v0.2
    """身份 -> 权限过滤条件（预过滤，绝不后过滤）。"""

    @abstractmethod
    def build(self, identity: Identity) -> Filter: ...


class VersionFilter(ABC):  # v0.2
    """版本过滤条件（默认只返现行版）。"""

    @abstractmethod
    def build(self, only_current: bool = True) -> Filter: ...


class Router(ABC):  # v0.3
    """检索前路由（寒暄/正常/超范围/需澄清/合同上传）。"""

    @abstractmethod
    def route(self, query: Query) -> Route: ...


class Reflector(ABC):  # v0.3
    """检索后反思纠正（须设重试上限防死循环）。"""

    @abstractmethod
    def should_retry(self, candidates: list[Candidate]) -> bool: ...

    @abstractmethod
    def rewrite(self, query: Query) -> Query: ...
