"""请求级模型调用用量采集。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class UsageSummary:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_cost_usd: Decimal | None
    complete: bool


@dataclass
class UsageCollector:
    input_cost_per_million: Decimal = Decimal("0")
    output_cost_per_million: Decimal = Decimal("0")
    _usages: list[dict[str, int]] = field(default_factory=list, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def record(self, payload: Any) -> None:
        normalized: dict[str, int] = {}
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict):
            aliases = {
                "prompt_tokens": ("prompt_tokens", "input_tokens"),
                "completion_tokens": ("completion_tokens", "output_tokens"),
                "total_tokens": ("total_tokens",),
            }
            for target, keys in aliases.items():
                value = next(
                    (usage.get(key) for key in keys if usage.get(key) is not None),
                    None,
                )
                if isinstance(value, int) and value >= 0:
                    normalized[target] = value
        with self._lock:
            self._usages.append(normalized)

    def summary(self) -> UsageSummary:
        with self._lock:
            usages = list(self._usages)
        if not usages:
            return UsageSummary(None, None, None, None, False)

        prompt_complete = all("prompt_tokens" in item for item in usages)
        completion_complete = all("completion_tokens" in item for item in usages)
        total_complete = all("total_tokens" in item for item in usages)
        prompt = sum(item.get("prompt_tokens", 0) for item in usages)
        completion = sum(item.get("completion_tokens", 0) for item in usages)
        total = sum(item.get("total_tokens", 0) for item in usages)
        if not total_complete and prompt_complete and completion_complete:
            total = prompt + completion
            total_complete = True

        cost = None
        if prompt_complete and completion_complete:
            cost = (
                Decimal(prompt) * self.input_cost_per_million
                + Decimal(completion) * self.output_cost_per_million
            ) / Decimal(1_000_000)
        return UsageSummary(
            prompt if prompt_complete else None,
            completion if completion_complete else None,
            total if total_complete else None,
            cost,
            prompt_complete and completion_complete and total_complete,
        )
