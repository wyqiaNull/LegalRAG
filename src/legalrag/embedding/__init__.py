"""Embedding 能力层（SPEC §3）。

导入本包即注册全部实现：bge_m3_api（生产）/ local_hash（离线测试）。
"""

from __future__ import annotations

from . import bge_m3_api, local_hash  # noqa: F401 —— 触发注册副作用
