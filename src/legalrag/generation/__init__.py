"""生成能力层（SPEC §3）。导入即注册默认生成器。"""

from __future__ import annotations

from . import generator  # noqa: F401 —— 触发注册副作用
