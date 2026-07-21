"""加载器集合 + auto 分派。

导入本包即注册全部具体加载器；AutoLoader 按扩展名委派。
"""

from __future__ import annotations

from pathlib import Path

from ...core import registry
from ...core.errors import IngestError
from ...core.interfaces import Loader, RawDoc

from . import docx, pdf, txt  # noqa: F401 —— 触发注册副作用

_SUFFIX_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "txt",
}


class AutoLoader(Loader):
    """按文件扩展名分派到具体加载器。"""

    def load(self, path: str) -> RawDoc:
        suffix = Path(path).suffix.lower()
        name = _SUFFIX_MAP.get(suffix)
        if name is None:
            raise IngestError(f"不支持的文件类型：{suffix}（支持 {sorted(_SUFFIX_MAP)}）")
        return registry.build("loader", name).load(path)


registry.register("loader", "auto", AutoLoader)
