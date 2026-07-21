"""组件注册表 / 工厂 —— 配置驱动消融。

能力层实现在各自模块用 ``register(kind, name, cls)`` 登记；编排层只用
``build(kind, name, **kwargs)`` 按配置里的实现名拿实例，不 import 具体类。
换组件 = 改 config 里的名字，主流程零改动。
"""

from __future__ import annotations

from typing import Any

from .errors import RegistryError

# kind（chunker/retriever/...） -> { name -> 实现类 }
_REGISTRY: dict[str, dict[str, type]] = {}


def register(kind: str, name: str, cls: type) -> None:
    """登记一个实现。重复登记同名视为错误，避免消融时静默覆盖。"""
    impls = _REGISTRY.setdefault(kind, {})
    if name in impls and impls[name] is not cls:
        raise RegistryError(f"重复注册组件：kind={kind!r} name={name!r}")
    impls[name] = cls


def get(kind: str, name: str) -> type:
    """取实现类；未注册则报错并列出可选项，便于排查配置笔误。"""
    impls = _REGISTRY.get(kind, {})
    if name not in impls:
        available = sorted(impls) or ["<空>"]
        raise RegistryError(
            f"未注册的组件：kind={kind!r} name={name!r}；可选：{available}"
        )
    return impls[name]


def build(kind: str, name: str, **kwargs: Any) -> Any:
    """按 (kind, name) 实例化组件，kwargs 透传构造参数（来自 config）。"""
    return get(kind, name)(**kwargs)


def registered(kind: str) -> list[str]:
    """列出某类已注册的实现名（供 CLI/调试）。"""
    return sorted(_REGISTRY.get(kind, {}))
