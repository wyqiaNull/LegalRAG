"""配置加载：pydantic-settings 合并 env + config/*.yaml。

- 密钥类（API base/key/model）从 ``.env`` / 环境变量读 —— 见 ``Secrets``；
- 组件选择与参数从 yaml 读 —— 见 ``AppConfig``，消融只改 yaml。
两者由 ``load_settings()`` 一次性组装为 ``Settings``。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..core.errors import ConfigError
from ..core.models import Confidentiality, DocType


class Secrets(BaseSettings):
    """从 .env / 环境变量读取的密钥与端点（不入库）。"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "bge-m3"

    rerank_api_base: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_model: str = "qwen2.5-7b-instruct"

    postgres_dsn: str = ""

    legalrag_config: str = "config/default.yaml"


# —— 以下模型镜像 config/default.yaml 的各段（extra=allow 容纳各实现的私有参数）——


class IngestCfg(BaseModel):
    model_config = {"extra": "allow"}
    loader: str = "auto"
    chunker: str = "fixed_len"
    chunk_size: int = 512
    chunk_overlap: int = 64


class EmbeddingCfg(BaseModel):
    model_config = {"extra": "allow"}
    impl: str = "bge_m3_api"
    dim: int = 1024
    batch_size: int = 32


class StoreCfg(BaseModel):
    model_config = {"extra": "allow"}
    vector: str = "memory"
    metadata: str = "memory"
    path: str = "data/processed"


class RetrievalCfg(BaseModel):
    model_config = {"extra": "allow"}
    impl: str = "vector_only"
    top_n: int = 50
    top_k: int = 5
    rrf_k: int = 60


class RerankCfg(BaseModel):
    model_config = {"extra": "allow"}
    impl: str = "noop"
    timeout: float = 30.0


class GenerationCfg(BaseModel):
    model_config = {"extra": "allow"}
    impl: str = "default"
    max_context_chunks: int = 5


class LLMCfg(BaseModel):
    model_config = {"extra": "allow"}
    impl: str = "qwen"


class ConversationCfg(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool = True
    resolver: str = "llm"
    history_turns: int = 4
    path: str = ""


class AclPolicyCfg(BaseModel):
    role: str
    allowed_confidentiality: list[Confidentiality]
    allowed_doc_types: list[DocType] = Field(default_factory=lambda: list(DocType))
    tenant_scope: str = "own"


def _default_acl_policies() -> list[AclPolicyCfg]:
    return [
        AclPolicyCfg(
            role="legal_staff",
            allowed_confidentiality=list(Confidentiality),
        ),
        AclPolicyCfg(
            role="business_user",
            allowed_confidentiality=[
                Confidentiality.PUBLIC,
                Confidentiality.INTERNAL,
            ],
        ),
        AclPolicyCfg(
            role="compliance_auditor",
            allowed_confidentiality=list(Confidentiality),
        ),
        AclPolicyCfg(
            role="external_client",
            allowed_confidentiality=[Confidentiality.PUBLIC],
        ),
    ]


class GovernanceCfg(BaseModel):
    model_config = {"extra": "allow"}
    permissions_enabled: bool = False
    permission_filter: str = "acl"
    versions_enabled: bool = False
    version_filter: str = "current"
    shared_tenant_id: str = "__global__"
    shared_doc_types: list[DocType] = Field(
        default_factory=lambda: [DocType.REGULATION]
    )
    acl_policies: list[AclPolicyCfg] = Field(default_factory=_default_acl_policies)

    @field_validator("shared_tenant_id")
    @classmethod
    def validate_shared_tenant_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized == "default":
            raise ValueError("shared_tenant_id 必须非空且不能为 default")
        return normalized

    @field_validator("shared_doc_types")
    @classmethod
    def deduplicate_shared_doc_types(cls, value: list[DocType]) -> list[DocType]:
        return list(dict.fromkeys(value))


class AppConfig(BaseModel):
    """一份 yaml 配置的强类型视图。"""

    ingest: IngestCfg = Field(default_factory=IngestCfg)
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    store: StoreCfg = Field(default_factory=StoreCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    rerank: RerankCfg = Field(default_factory=RerankCfg)
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    conversation: ConversationCfg = Field(default_factory=ConversationCfg)
    governance: GovernanceCfg = Field(default_factory=GovernanceCfg)


class Settings(BaseModel):
    """运行期完整配置 = 密钥 + yaml 组件配置。"""

    secrets: Secrets
    config: AppConfig


def load_settings(config_path: str | None = None) -> Settings:
    """加载密钥与 yaml 配置。config_path 为空时用 .env 里的 LEGALRAG_CONFIG。"""
    secrets = Secrets()
    path = Path(config_path or secrets.legalrag_config)
    if not path.exists():
        raise ConfigError(f"配置文件不存在：{path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings(secrets=secrets, config=AppConfig(**raw))
