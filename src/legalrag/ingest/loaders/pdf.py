"""PDF 加载器（PyMuPDF/fitz）：按页保存文本，供回填页码。"""

from __future__ import annotations

from pathlib import Path

from ...core import registry
from ...core.errors import IngestError
from ...core.interfaces import Loader, RawDoc


class PdfLoader(Loader):
    def load(self, path: str) -> RawDoc:
        try:
            import fitz  # PyMuPDF
        except ImportError as e:  # pragma: no cover
            raise IngestError("加载 PDF 需安装 pymupdf") from e
        p = Path(path)
        pages: list[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                pages.append(page.get_text("text"))
        return RawDoc(doc_name=p.stem, pages=pages)


registry.register("loader", "pdf", PdfLoader)
