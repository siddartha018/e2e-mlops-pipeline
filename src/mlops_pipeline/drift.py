"""Drift detector: PSI, Wasserstein, embedding cosine.

Three signals:
- PSI (Population Stability Index) for categorical / discretized numeric.
- Wasserstein distance for continuous numeric.
- Cosine distance between current-batch centroid and training centroid for
  text/embeddings (uses the vector store's stored embeddings).

Verdict: 'drift' if any monitored signal crosses its threshold. Report per-signal
so the user can see which feature moved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import wasserstein_distance


class DriftSignal(BaseModel):
    name: str
    kind: str
    value: float
    threshold: float
    drifted: bool
    detail: dict[str, Any] = {}


class DriftReport(BaseModel):
    verdict: str
    signals: list[DriftSignal]
    baseline_id: str | None = None
    current_id: str | None = None

    @property
    def drifted(self) -> bool:
        return self.verdict == "drift"


@dataclass
class DriftThresholds:
    psi: float = 0.2
    wasserstein: float = 0.15
    cosine: float = 0.15


class DriftDetector:
    def __init__(self, thresholds: DriftThresholds | None = None) -> None:
        self.t = thresholds or DriftThresholds()

    def check_numeric(self, name: str, baseline: np.ndarray, current: np.ndarray) -> DriftSignal:
        w = float(wasserstein_distance(baseline, current))
        b_std = float(np.std(baseline)) or 1.0
        normed = w / b_std
        return DriftSignal(
            name=name,
            kind="wasserstein",
            value=normed,
            threshold=self.t.wasserstein,
            drifted=normed > self.t.wasserstein,
            detail={"raw_wasserstein": w, "baseline_std": b_std},
        )

    def check_categorical(
        self, name: str, baseline: pd.Series, current: pd.Series
    ) -> DriftSignal:
        b_counts = baseline.value_counts(normalize=True)
        c_counts = current.value_counts(normalize=True)
        cats = set(b_counts.index) | set(c_counts.index)
        eps = 1e-4
        psi = 0.0
        for cat in cats:
            b = float(b_counts.get(cat, eps))
            c = float(c_counts.get(cat, eps))
            b = max(b, eps)
            c = max(c, eps)
            psi += (c - b) * float(np.log(c / b))
        return DriftSignal(
            name=name,
            kind="psi",
            value=psi,
            threshold=self.t.psi,
            drifted=psi > self.t.psi,
        )

    def check_embedding_centroid(
        self, name: str, baseline_centroid: np.ndarray, current_centroid: np.ndarray
    ) -> DriftSignal:
        def _norm(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        b = _norm(baseline_centroid.astype(np.float64))
        c = _norm(current_centroid.astype(np.float64))
        cosine = float(1.0 - np.dot(b, c))
        return DriftSignal(
            name=name,
            kind="cosine",
            value=cosine,
            threshold=self.t.cosine,
            drifted=cosine > self.t.cosine,
        )

    def check(
        self,
        signals: list[DriftSignal],
        baseline_id: str | None = None,
        current_id: str | None = None,
    ) -> DriftReport:
        verdict = "drift" if any(s.drifted for s in signals) else "stable"
        return DriftReport(
            verdict=verdict, signals=signals, baseline_id=baseline_id, current_id=current_id
        )
