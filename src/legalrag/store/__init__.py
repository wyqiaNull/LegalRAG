"""存储层（SPEC §3）。导入即注册内存实现。"""

from __future__ import annotations

from . import metadata_store, vector_store  # noqa: F401 —— 触发注册副作用
