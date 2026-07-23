"""离线占位 LLM（仅供无 API key 时跑通链路 / 冒烟测试，非真实生成）。

不产出真实法律意见，只回显收到的上下文条数，用于验证 ingest→检索→生成 的管线接通。
生产路径请用 openai。
"""

from __future__ import annotations

from typing import Any

from ..core import registry
from ..core.interfaces import LLMClient


class EchoLLM(LLMClient):
    def complete(self, prompt: str, **opts: Any) -> str:
        return "【离线占位回答】未配置真实 LLM，仅回显：已根据检索到的上下文生成占位答案。"


registry.register("llm", "echo", EchoLLM)
