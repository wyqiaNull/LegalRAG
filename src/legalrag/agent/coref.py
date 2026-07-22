"""多轮追问的独立问题改写。"""

from __future__ import annotations

from ..core import registry
from ..core.interfaces import LLMClient

_PROMPT = """你负责把多轮对话中的当前追问改写成可独立检索的问题。
只使用必要的历史问题补全指代，不回答问题，不添加历史中没有的信息。
如果当前问题已经独立完整，原样返回。
只输出一行改写后的问题，不要解释，不要加标签。

最近的独立问题：
{history}

当前追问：{question}
"""


class LLMCoreferenceResolver:
    def __init__(self, llm: LLMClient, history_turns: int = 4) -> None:
        self.llm = llm
        self.history_turns = history_turns

    def resolve(self, question: str, history: list[str]) -> str:
        if not history:
            return question
        recent = history[-self.history_turns :]
        rendered = "\n".join(
            f"{index}. {item}" for index, item in enumerate(recent, start=1)
        )
        rewritten = self.llm.complete(
            _PROMPT.format(history=rendered, question=question), temperature=0.0
        ).strip()
        if rewritten.startswith("```") and rewritten.endswith("```"):
            rewritten = rewritten[3:-3].strip()
        for prefix in ("改写后的问题：", "独立问题：", "问题："):
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix) :].strip()
        return rewritten or question


registry.register("coreference", "llm", LLMCoreferenceResolver)
