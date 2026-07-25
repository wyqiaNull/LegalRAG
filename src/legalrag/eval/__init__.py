"""LegalRAG evaluation harness."""

from .dataset import dataset_sha256, load_cases, validate_review
from .metrics import evaluate_deterministic_metrics
from .models import EvalCase, EvalObservation, EvalTurn, GoldCitation

__all__ = [
    "EvalCase",
    "EvalObservation",
    "EvalTurn",
    "GoldCitation",
    "dataset_sha256",
    "evaluate_deterministic_metrics",
    "load_cases",
    "validate_review",
]
