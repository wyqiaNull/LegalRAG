"""条款级分块器。

以法条标题为边界，款、项及其他续行保留在所属法条内。标题、修订说明和目录
作为前言块保留；正文中的章、节标题归入其后的法条。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ...core import registry
from ...core.interfaces import Chunker, RawDoc
from ...core.models import Chunk
from .. import metadata

_NUMERALS = "一二三四五六七八九十百千万零〇两0-9"
ARTICLE_RE = re.compile(
    rf"^\s*(第[{_NUMERALS}]+条(?:之[{_NUMERALS}]+)?)(?=[\s　]|$)",
    re.MULTILINE,
)
_STRUCTURE_RE = re.compile(
    rf"^\s*第[{_NUMERALS}]+(?:编|章|节)(?=[\s　]|$)"
)
_PAGE_SEPARATOR = "\n"


@dataclass(frozen=True)
class _Line:
    text: str
    start: int


def article_headings(text: str) -> list[str]:
    """返回正文中所有位于行首的法条编号。"""
    return [match.group(1) for match in ARTICLE_RE.finditer(text)]


class ClauseChunker(Chunker):
    def _page_ranges(self, pages: list[str]) -> list[tuple[int, int, int]]:
        ranges: list[tuple[int, int, int]] = []
        cursor = 0
        for page_no, page_text in enumerate(pages, start=1):
            if page_no > 1:
                cursor += len(_PAGE_SEPARATOR)
            start = cursor
            cursor += len(page_text)
            ranges.append((page_no, start, cursor))
        return ranges

    def _page_of(self, offset: int, ranges: list[tuple[int, int, int]]) -> int | None:
        for page_no, start, end in ranges:
            if start <= offset < end:
                return page_no
        return ranges[-1][0] if ranges else None

    def split(self, raw: RawDoc, meta: dict[str, Any]) -> list[Chunk]:
        text = raw.text
        if not text.strip():
            return []

        lines: list[_Line] = []
        cursor = 0
        for line_text in text.splitlines(keepends=True):
            lines.append(_Line(line_text, cursor))
            cursor += len(line_text)
        if cursor < len(text):
            lines.append(_Line(text[cursor:], cursor))

        ranges = self._page_ranges(raw.pages)
        chunks: list[Chunk] = []
        preamble: list[_Line] = []
        pending_structure: list[_Line] = []
        current: list[_Line] = []
        clause_no = ""
        body_started = False

        def emit(parts: list[_Line], number: str = "") -> None:
            if not parts:
                return
            joined = "".join(part.text for part in parts)
            content = joined.strip()
            if not content:
                return
            leading = len(joined) - len(joined.lstrip())
            start = parts[0].start + leading
            chunks.append(
                metadata.assemble_chunk(
                    content,
                    meta,
                    seq=len(chunks),
                    page=self._page_of(start, ranges),
                    clause_no=number,
                )
            )

        for line in lines:
            stripped = line.text.strip()
            article_match = ARTICLE_RE.match(stripped)
            if article_match:
                emit(current, clause_no)
                current = []
                clause_no = ""
                if not body_started:
                    emit(preamble)
                    preamble = []
                    body_started = True
                current.extend(pending_structure)
                pending_structure = []
                clause_no = article_match.group(1)
                current.append(line)
                continue

            if body_started and _STRUCTURE_RE.match(stripped):
                emit(current, clause_no)
                current = []
                clause_no = ""
                pending_structure.append(line)
                continue

            if not body_started:
                preamble.append(line)
            elif pending_structure:
                pending_structure.append(line)
            else:
                current.append(line)

        emit(current, clause_no)
        emit(pending_structure)
        emit(preamble)
        return chunks


registry.register("chunker", "clause", ClauseChunker)
