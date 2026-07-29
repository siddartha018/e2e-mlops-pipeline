from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_pipeline.drift import DriftDetector, DriftThresholds


def test_no_drift_numeric() -> None:
    det = DriftDetector(DriftThresholds(wasserstein=0.15))
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 1000)
    current = rng.normal(0, 1, 1000)
    signal = det.check_numeric("feature", baseline, current)
    assert not signal.drifted


def test_drift_numeric_shift() -> None:
    det = DriftDetector(DriftThresholds(wasserstein=0.15))
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 1000)
    current = rng.normal(2, 1, 1000)  # big mean shift
    signal = det.check_numeric("feature", baseline, current)
    assert signal.drifted
    assert signal.value > 0.15


def test_no_drift_categorical() -> None:
    det = DriftDetector(DriftThresholds(psi=0.2))
    baseline = pd.Series([0, 1] * 500)
    current = pd.Series([0, 1] * 500)
    signal = det.check_categorical("label", baseline, current)
    assert not signal.drifted
    assert signal.value < 1e-3


def test_drift_categorical_flip() -> None:
    det = DriftDetector(DriftThresholds(psi=0.2))
    baseline = pd.Series([0] * 800 + [1] * 200)
    current = pd.Series([0] * 200 + [1] * 800)  # inverted
    signal = det.check_categorical("label", baseline, current)
    assert signal.drifted


def test_no_drift_embedding() -> None:
    det = DriftDetector(DriftThresholds(cosine=0.15))
    v = np.array([1.0, 0.0, 0.0])
    signal = det.check_embedding_centroid("embedding", v, v)
    assert not signal.drifted
    assert signal.value < 1e-6


def test_drift_embedding_rotation() -> None:
    det = DriftDetector(DriftThresholds(cosine=0.15))
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])  # orthogonal -> cosine = 1
    signal = det.check_embedding_centroid("embedding", b, c)
    assert signal.drifted
    assert 0.99 < signal.value <= 1.0001


def test_overall_verdict() -> None:
    det = DriftDetector()
    rng = np.random.default_rng(0)
    s1 = det.check_numeric("x", rng.normal(0, 1, 500), rng.normal(0, 1, 500))
    s2 = det.check_embedding_centroid(
        "e", np.array([1.0, 0]), np.array([0.0, 1.0])
    )
    report = det.check([s1, s2])
    assert report.verdict == "drift"  # one drifted signal => drift
