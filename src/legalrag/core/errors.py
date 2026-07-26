"""领域异常层级。

所有 LegalRAG 主动抛出的异常都继承 LegalRAGError，便于接口层统一捕获与降级。
"""

from __future__ import annotations


class LegalRAGError(Exception):
    """所有领域异常的基类。"""


class ConfigError(LegalRAGError):
    """配置缺失或非法（如 yaml 指定了未注册的实现名、缺 API key）。"""


class RegistryError(LegalRAGError):
    """组件注册表相关错误（重复注册 / 请求未注册的实现）。"""


class IngestError(LegalRAGError):
    """摄取阶段错误（无法加载文件、不支持的格式等）。"""


class StorageError(LegalRAGError):
    """持久化失败（数据库连接、建表或事务写入失败）。"""


class RetrievalError(LegalRAGError):
    """检索阶段错误（向量库不可用、维度不匹配等）。"""


class GenerationError(LegalRAGError):
    """生成阶段错误（LLM 调用失败等）。"""


class ContractReviewError(LegalRAGError):
    """合同审查输入、模型输出或编排过程无效。"""
