from __future__ import annotations

import pytest

from mlops_pipeline.registry import Registry, RegistryError, Stage


def test_register_and_promote_first_production(registry: Registry) -> None:
    v1 = registry.register("clf", run_id="r1", blob_hash="h1")
    assert v1 == 1
    registry.promote("clf", v1, Stage.STAGING)
    registry.promote("clf", v1, Stage.PRODUCTION)
    prod = registry.get_production("clf")
    assert prod is not None
    assert prod.version == v1


def test_promote_supersedes_previous_production(registry: Registry) -> None:
    v1 = registry.register("clf", run_id="r1", blob_hash="h1")
    registry.promote("clf", v1, Stage.STAGING)
    registry.promote("clf", v1, Stage.PRODUCTION)
    v2 = registry.register("clf", run_id="r2", blob_hash="h2")
    registry.promote("clf", v2, Stage.STAGING)
    registry.promote("clf", v2, Stage.PRODUCTION)
    prod = registry.get_production("clf")
    assert prod is not None
    assert prod.version == v2
    # v1 should now be archived (invariant: single production per name)
    v1_row = registry.get("clf", v1)
    assert v1_row.stage == Stage.ARCHIVED


def test_invalid_transition(registry: Registry) -> None:
    v1 = registry.register("clf", run_id="r1", blob_hash="h1")
    # none -> production is not allowed (must go via staging)
    with pytest.raises(RegistryError):
        registry.promote("clf", v1, Stage.PRODUCTION)


def test_promote_unknown_version(registry: Registry) -> None:
    with pytest.raises(RegistryError):
        registry.promote("clf", 999, Stage.STAGING)


def test_version_autoincrement(registry: Registry) -> None:
    assert registry.register("clf", "r1", "h1") == 1
    assert registry.register("clf", "r2", "h2") == 2
    assert registry.register("clf", "r3", "h3") == 3
    # separate model name = independent version stream
    assert registry.register("other", "r4", "h4") == 1
