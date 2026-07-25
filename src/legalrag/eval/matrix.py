"""Controlled-variable validation for ablation profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import yaml
from pydantic import BaseModel, Field

from ..config.settings import AppConfig
from ..core.errors import ConfigError


class MatrixProfile(BaseModel):
    name: str
    config: str


class MatrixExperiment(BaseModel):
    name: str
    profiles: list[str] = Field(min_length=2)
    allowed_differences: list[str]


class EvalMatrix(BaseModel):
    profiles: list[MatrixProfile] = Field(min_length=2)
    experiments: list[MatrixExperiment] = Field(min_length=1)


def load_matrix(path: str | Path) -> EvalMatrix:
    target = Path(path)
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    try:
        matrix = EvalMatrix.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"消融矩阵无效：{exc}") from exc
    names = [profile.name for profile in matrix.profiles]
    if len(names) != len(set(names)):
        raise ConfigError("消融矩阵 profile 名称不能重复")
    known = set(names)
    for experiment in matrix.experiments:
        unknown = set(experiment.profiles) - known
        if unknown:
            raise ConfigError(f"实验 {experiment.name} 引用了未知 profile：{sorted(unknown)}")
    return matrix


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten(item, path))
        else:
            flattened[path] = item
    return flattened


def _load_app_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)


def validate_matrix(path: str | Path) -> EvalMatrix:
    target = Path(path)
    matrix = load_matrix(target)
    configs: dict[str, dict[str, Any]] = {}
    for profile in matrix.profiles:
        config_path = Path(profile.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        if not config_path.exists():
            raise ConfigError(f"评测 profile 不存在：{config_path}")
        configs[profile.name] = _flatten(
            _load_app_config(config_path).model_dump(mode="json")
        )

    for experiment in matrix.experiments:
        baseline_name = experiment.profiles[0]
        baseline = configs[baseline_name]
        allowed = set(experiment.allowed_differences)
        for profile_name in experiment.profiles[1:]:
            candidate = configs[profile_name]
            differing = {
                key
                for key in baseline.keys() | candidate.keys()
                if baseline.get(key) != candidate.get(key)
            }
            unexpected = differing - allowed
            if unexpected:
                raise ConfigError(
                    f"实验 {experiment.name} 的 {baseline_name}/{profile_name} "
                    f"存在非控制变量差异：{sorted(unexpected)}"
                )
    return matrix


def matrix_evidence(path: str | Path) -> dict:
    target = Path(path)
    matrix = validate_matrix(target)
    profile_values = {}
    profile_hashes = {}
    for profile in matrix.profiles:
        config_path = Path(profile.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        profile_values[profile.name] = _flatten(
            _load_app_config(config_path).model_dump(mode="json")
        )
        profile_hashes[profile.name] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    experiments = {}
    for experiment in matrix.experiments:
        experiments[experiment.name] = {
            profile: {
                key: profile_values[profile].get(key)
                for key in experiment.allowed_differences
            }
            for profile in experiment.profiles
        }
    return {"config_sha256": profile_hashes, "controlled_values": experiments}
