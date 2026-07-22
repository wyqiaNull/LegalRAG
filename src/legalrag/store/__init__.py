"""存储层。导入即注册全部存储实现。"""

from __future__ import annotations

from . import metadata_store, postgres, vector_store  # noqa: F401 —— 触发注册副作用
