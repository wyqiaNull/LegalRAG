"""OpenAI 兼容 chat/completions 客户端。"""

from __future__ import annotations

from typing import Any

import httpx

from ..core import registry
from ..core.errors import ConfigError, GenerationError
from ..core.interfaces import LLMClient
from ..observability import UsageCollector


class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model: str = "qwen2.5-7b-instruct",
        timeout: float = 60.0,
        usage_collector: UsageCollector | None = None,
    ) -> None:
        if not api_base:
            raise ConfigError("openai 需配置 LLM_API_BASE")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.usage_collector = usage_collector

    def complete(self, prompt: str, **opts: Any) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": opts.get("temperature", 0.2),
        }
        try:
            resp = httpx.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            if self.usage_collector is not None:
                self.usage_collector.record(body)
            return body["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError) as e:
            raise GenerationError(f"LLM API 调用失败：{e}") from e


registry.register("llm", "openai", OpenAIClient)
