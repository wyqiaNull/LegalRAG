"""固定长度 + overlap 分块器（SPEC §6，FR-2 MVP）。

按字符窗口切分整篇文本，为每个 chunk 回填其起始位置所在的页码。
条款级语义分块见 v0.1 的 clause.py。
"""

from __future__ import annotations

from typing import Any

from ...core import registry
from ...core.errors import ConfigError
from ...core.interfaces import Chunker, RawDoc
from ...core.models import Chunk
from .. import metadata

_SEP = "\n"  # 与 RawDoc.text 的页间连接符一致，保证偏移量对齐


class FixedLenChunker(Chunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ConfigError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _page_ranges(self, pages: list[str]) -> list[tuple[int, int, int]]:
        """返回 [(page_no, start, end)]，偏移量与 SEP.join(pages) 对齐。"""
        ranges: list[tuple[int, int, int]] = []
        cursor = 0
        for i, ptext in enumerate(pages, start=1):
            if i > 1:
                cursor += len(_SEP)
            start = cursor
            cursor += len(ptext)
            ranges.append((i, start, cursor))
        return ranges

    def _page_of(self, offset: int, ranges: list[tuple[int, int, int]]) -> int | None:
        for page_no, start, end in ranges:
            if start <= offset < end:
                return page_no
        return ranges[-1][0] if ranges else None

    def split(self, raw: RawDoc, meta: dict[str, Any]) -> list[Chunk]:
        text = raw.text
        ranges = self._page_ranges(raw.pages)
        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        seq = 0
        for start in range(0, max(len(text), 1), step):
            piece = text[start : start + self.chunk_size]
            if not piece.strip():
                continue
            chunks.append(
                metadata.assemble_chunk(
                    piece,
                    meta,
                    seq=seq,
                    page=self._page_of(start, ranges),
                )
            )
            seq += 1
            if start + self.chunk_size >= len(text):
                break
        return chunks


registry.register("chunker", "fixed_len", FixedLenChunker)
