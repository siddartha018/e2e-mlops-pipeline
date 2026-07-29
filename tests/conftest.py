from __future__ import annotations

from pathlib import Path

import pytest

from mlops_pipeline.core import ArtefactStore, FakeClock, RealRng
from mlops_pipeline.dataset_store import DatasetStore
from mlops_pipeline.registry import Registry
from mlops_pipeline.tracker import Tracker


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(start=1000.0)


@pytest.fixture
def rng() -> RealRng:
    return RealRng(seed=42)


@pytest.fixture
def artefacts(tmp_path: Path) -> ArtefactStore:
    return ArtefactStore(root=tmp_path / "store")


@pytest.fixture
def tracker(tmp_path: Path, artefacts: ArtefactStore, clock: FakeClock) -> Tracker:
    return Tracker(db_path=tmp_path / "tracker.sqlite", artefacts=artefacts, clock=clock)


@pytest.fixture
def registry(tmp_path: Path, clock: FakeClock) -> Registry:
    return Registry(db_path=tmp_path / "registry.sqlite", clock=clock)


@pytest.fixture
def dataset_store(tmp_path: Path, artefacts: ArtefactStore) -> DatasetStore:
    return DatasetStore(artefacts=artefacts, manifests_dir=tmp_path / "manifests")
