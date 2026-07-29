"""Champion vs challenger promotion gate.

Given a challenger model version, evaluate both champion (current production)
and challenger on the frozen holdout, and require challenger to meet the
promotion criterion. On pass, promote challenger to production; on fail, keep
champion.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import PromoteConfig
from .registry import ModelVersion, Registry, Stage


@dataclass
class PromotionOutcome:
    passed: bool
    champion_metric: float | None
    challenger_metric: float
    delta: float
    reason: str


def evaluate_promotion(
    registry: Registry,
    model_name: str,
    challenger_version: int,
    evaluate_fn: Callable[[ModelVersion], float],
    cfg: PromoteConfig,
) -> PromotionOutcome:
    """
    evaluate_fn: takes a ModelVersion, returns the primary metric on the holdout.
    Caller is responsible for loading the model artefact and running eval.
    """
    challenger = registry.get(model_name, challenger_version)
    challenger_metric = evaluate_fn(challenger)

    def _to_production(version: int) -> None:
        current = registry.get(model_name, version).stage
        if current == Stage.NONE:
            registry.promote(model_name, version, Stage.STAGING)
        registry.promote(model_name, version, Stage.PRODUCTION)

    champion = registry.get_production(model_name)
    if champion is None:
        _to_production(challenger_version)
        return PromotionOutcome(
            passed=True,
            champion_metric=None,
            challenger_metric=challenger_metric,
            delta=0.0,
            reason="no incumbent — challenger becomes first production version",
        )

    champion_metric = evaluate_fn(champion)
    if cfg.higher_is_better:
        delta = challenger_metric - champion_metric
        passed = delta >= cfg.min_improvement
    else:
        delta = champion_metric - challenger_metric
        passed = delta >= cfg.min_improvement

    if passed:
        _to_production(challenger_version)
        reason = (
            f"challenger {cfg.metric}={challenger_metric:.4f} "
            f"beats champion {champion_metric:.4f} by {delta:+.4f} "
            f"(threshold {cfg.min_improvement:+.4f})"
        )
    else:
        reason = (
            f"challenger {cfg.metric}={challenger_metric:.4f} "
            f"vs champion {champion_metric:.4f} delta {delta:+.4f} "
            f"< threshold {cfg.min_improvement:+.4f} — rejected"
        )

    return PromotionOutcome(
        passed=passed,
        champion_metric=champion_metric,
        challenger_metric=challenger_metric,
        delta=delta,
        reason=reason,
    )
