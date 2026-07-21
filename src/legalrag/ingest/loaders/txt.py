"""TXT 加载器：整篇作为单页（无分页概念）。"""

from __future__ import annotations

from pathlib import Path

from ...core import registry
from ...core.interfaces import Loader, RawDoc


class TxtLoader(Loader):
    def load(self, path: str) -> RawDoc:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="ignore")
        return RawDoc(doc_name=p.stem, pages=[text])


registry.register("loader", "txt", TxtLoader)
