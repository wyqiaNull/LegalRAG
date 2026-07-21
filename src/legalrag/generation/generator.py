"""生成器（SPEC §6，FR-4 MVP）—— context 拼接 → LLM → Answer。

MVP 不做引用装配与一致性校验（citations 留空），这些到 v0.2 补齐。
"""

from __future__ import annotations

from pathlib import Path

from ..core import registry
from ..core.interfaces import Candidate, Generator, LLMClient
from ..core.models import Answer, Query, Route

_PROMPT_FILE = Path(__file__).parent / "prompts" / "answer.txt"


def _format_context(contexts: list[Candidate]) -> str:
    blocks = []
    for i, cand in enumerate(contexts, start=1):
        c = cand.chunk
        tag = f"[{i}] 《{c.doc_name}》"
        if c.clause_no:
            tag += f" {c.clause_no}"
        if c.page is not None:
            tag += f"（第{c.page}页）"
        blocks.append(f"{tag}\n{c.content}")
    return "\n\n".join(blocks) if blocks else "（无检索结果）"


class DefaultGenerator(Generator):
    def __init__(self, llm: LLMClient, max_context_chunks: int = 5) -> None:
        self.llm = llm
        self.max_context_chunks = max_context_chunks
        self.template = _PROMPT_FILE.read_text(encoding="utf-8")

    def generate(self, query: Query, contexts: list[Candidate]) -> Answer:
        used = contexts[: self.max_context_chunks]
        prompt = self.template.format(
            context=_format_context(used), question=query.text
        )
        text = self.llm.complete(prompt)
        return Answer(
            text=text,
            citations=[],  # v0.2 引用装配
            refused=False,
            route=Route.NORMAL,
            retrieved_chunk_ids=[c.chunk.chunk_id for c in used],
        )


registry.register("generator", "default", DefaultGenerator)
