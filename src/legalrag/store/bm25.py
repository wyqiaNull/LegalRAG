"""小规模语料使用的内存 BM25 实现。"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[0-9a-zA-Z]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """以中文单字、连续字母数字及相邻二元组构造词面特征。"""
    tokens = _TOKEN_RE.findall(text.lower())
    return tokens + [tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)]


def bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not documents:
        return []
    query_terms = set(tokenize(query))
    if not query_terms:
        return [0.0] * len(documents)

    tokenized = [tokenize(document) for document in documents]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / len(lengths) or 1.0
    frequencies = [Counter(tokens) for tokens in tokenized]
    document_frequency = {
        term: sum(1 for frequency in frequencies if term in frequency)
        for term in query_terms
    }
    total = len(documents)
    scores: list[float] = []
    for frequency, length in zip(frequencies, lengths):
        score = 0.0
        for term in query_terms:
            term_frequency = frequency.get(term, 0)
            if not term_frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            denominator = term_frequency + k1 * (
                1.0 - b + b * length / average_length
            )
            score += inverse_frequency * term_frequency * (k1 + 1.0) / denominator
        scores.append(score)
    return scores
