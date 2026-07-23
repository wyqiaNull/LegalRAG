"""LLM 能力层。导入即注册 openai（生产）/ echo（离线）。"""

from __future__ import annotations

from . import echo, openai  # noqa: F401 —— 触发注册副作用
