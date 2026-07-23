"""生成器 —— context 拼接 → LLM → 引用装配 → Answer。"""

from __future__ import annotations

from pathlib import Path

from ..core import registry
from ..core.interfaces import Candidate, Generator, LLMClient
from ..core.models import Answer, Query, Route
from .citation import assemble_citations, historical_versions, validate_citations

_PROMPT_FILE = Path(__file__).parent / "prompts" / "answer.txt"


def _format_context(contexts: list[Candidate]) -> str:
    blocks = []
    for i, cand in enumerate(contexts, start=1):
        c = cand.chunk
        tag = f"[{i}] 《{c.doc_name}》"
        if c.clause_no:
            tag += f" {c.clause_no}"
        details = []
        if c.version:
            details.append(f"版本：{c.version}")
        details.append(f"第{c.page}页" if c.page is not None else "页码未知")
        if not c.is_current:
            details.append("历史版本")
        tag += f"（{'；'.join(details)}）"
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
        versions = historical_versions(used)
        if versions:
            text = f"【历史版本：{'、'.join(versions)}】\n{text}"
        citations = validate_citations(assemble_citations(used), used)
        return Answer(
            text=text,
            citations=citations,
            refused=False,
            route=Route.NORMAL,
            retrieved_chunk_ids=[c.chunk.chunk_id for c in used],
        )


registry.register("generator", "default", DefaultGenerator)
