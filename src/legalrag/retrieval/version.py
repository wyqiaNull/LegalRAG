"""检索版本过滤中间件。"""

from __future__ import annotations

from ..core import registry
from ..core.interfaces import Filter, VersionFilter


class CurrentVersionFilter(VersionFilter):
    def build(self, only_current: bool = True) -> Filter:
        return {"is_current": True} if only_current else {}


registry.register("version_filter", "current", CurrentVersionFilter)
