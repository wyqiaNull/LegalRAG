"""LLM 能力层（SPEC §3）。导入即注册 qwen（生产）/ echo（离线）。"""

from __future__ import annotations

from . import echo, qwen  # noqa: F401 —— 触发注册副作用
