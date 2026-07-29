from __future__ import annotations

from mlops_pipeline.config import PromoteConfig
from mlops_pipeline.promote import evaluate_promotion
from mlops_pipeline.registry import Registry, Stage


def test_first_production_auto_passes(registry: Registry) -> None:
    v = registry.register("clf", "r1", "h1")
    out = evaluate_promotion(
        registry=registry,
        model_name="clf",
        challenger_version=v,
        evaluate_fn=lambda mv: 0.8,
        cfg=PromoteConfig(metric="accuracy", higher_is_better=True, min_improvement=0.0),
    )
    assert out.passed
    prod = registry.get_production("clf")
    assert prod and prod.version == v


def test_challenger_beats_champion(registry: Registry) -> None:
    v1 = registry.register("clf", "r1", "h1")
    registry.promote("clf", v1, Stage.STAGING)
    registry.promote("clf", v1, Stage.PRODUCTION)
    v2 = registry.register("clf", "r2", "h2")

    scores = {"h1": 0.80, "h2": 0.85}
    out = evaluate_promotion(
        registry=registry,
        model_name="clf",
        challenger_version=v2,
        evaluate_fn=lambda mv: scores[mv.blob_hash],
        cfg=PromoteConfig(metric="accuracy", higher_is_better=True, min_improvement=0.01),
    )
    assert out.passed
    prod = registry.get_production("clf")
    assert prod and prod.version == v2


def test_challenger_below_threshold_rejected(registry: Registry) -> None:
    v1 = registry.register("clf", "r1", "h1")
    registry.promote("clf", v1, Stage.STAGING)
    registry.promote("clf", v1, Stage.PRODUCTION)
    v2 = registry.register("clf", "r2", "h2")

    scores = {"h1": 0.80, "h2": 0.805}
    out = evaluate_promotion(
        registry=registry,
        model_name="clf",
        challenger_version=v2,
        evaluate_fn=lambda mv: scores[mv.blob_hash],
        cfg=PromoteConfig(metric="accuracy", higher_is_better=True, min_improvement=0.02),
    )
    assert not out.passed
    prod = registry.get_production("clf")
    assert prod and prod.version == v1  # champion still reigns


def test_lower_is_better(registry: Registry) -> None:
    v1 = registry.register("clf", "r1", "h1")
    registry.promote("clf", v1, Stage.STAGING)
    registry.promote("clf", v1, Stage.PRODUCTION)
    v2 = registry.register("clf", "r2", "h2")

    scores = {"h1": 0.50, "h2": 0.30}  # lower loss = better
    out = evaluate_promotion(
        registry=registry,
        model_name="clf",
        challenger_version=v2,
        evaluate_fn=lambda mv: scores[mv.blob_hash],
        cfg=PromoteConfig(metric="loss", higher_is_better=False, min_improvement=0.05),
    )
    assert out.passed
