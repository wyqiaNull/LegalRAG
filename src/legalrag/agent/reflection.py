"""基于检索质量门槛的 query 反思与改写。"""

from __future__ import annotations

import re

from ..core import registry
from ..core.interfaces import LLMClient, Reflector
from ..core.models import Candidate, Query

_PROMPT = """你负责改写企业法律知识库中召回质量不足的检索问题。
保留用户原意，不回答问题，不添加原问题中没有的事实。
优先补充规范法律术语、同义表达或明确的检索对象，使问题更容易命中相关法规、制度或法律意见。
只输出一行改写后的问题，不要解释，不要加标签。

原检索问题：
{question}
"""

_FENCED_OUTPUT = re.compile(r"^```(?:text)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)
_PREFIXES = ("改写后的问题：", "改写后的问题:", "检索问题：", "检索问题:", "问题：", "问题:")


def _parse_rewrite(output: str, original: str) -> str:
    normalized = output.strip()
    fenced = _FENCED_OUTPUT.fullmatch(normalized)
    if fenced is not None:
        normalized = fenced.group(1).strip()
    for prefix in _PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    first_line = next((line.strip() for line in normalized.splitlines() if line.strip()), "")
    return first_line.strip("'\"") or original


class ScoreThresholdReflector(Reflector):
    """用候选数量和最高重排分判断是否需要纠正检索。"""

    def __init__(
        self,
        llm: LLMClient,
        min_candidates: int = 1,
        min_score: float = 0.5,
    ) -> None:
        if min_candidates < 1:
            raise ValueError("min_candidates 必须至少为 1")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score 必须在 0 到 1 之间")
        self.llm = llm
        self.min_candidates = min_candidates
        self.min_score = min_score

    def should_retry(self, candidates: list[Candidate]) -> bool:
        if len(candidates) < self.min_candidates:
            return True
        return max(candidate.score for candidate in candidates) < self.min_score

    def rewrite(self, query: Query) -> Query:
        output = self.llm.complete(
            _PROMPT.format(question=query.text), temperature=0.0
        )
        rewritten = _parse_rewrite(output, query.text)
        return query.model_copy(update={"text": rewritten})


registry.register("reflector", "score_threshold", ScoreThresholdReflector)
