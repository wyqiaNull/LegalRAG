"""DOCX 加载器（python-docx）：无分页概念，整篇作为单页。"""

from __future__ import annotations

from pathlib import Path

from ...core import registry
from ...core.errors import IngestError
from ...core.interfaces import Loader, RawDoc


class DocxLoader(Loader):
    def load(self, path: str) -> RawDoc:
        try:
            import docx  # python-docx
        except ImportError as e:  # pragma: no cover
            raise IngestError("加载 DOCX 需安装 python-docx") from e
        p = Path(path)
        doc = docx.Document(path)
        text = "\n".join(para.text for para in doc.paragraphs)
        return RawDoc(doc_name=p.stem, pages=[text])


registry.register("loader", "docx", DocxLoader)
