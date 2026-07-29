"""Wire everything together based on config. One place to construct the app."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .clients import (
    FaissLocalStore,
    HttpP2InferenceClient,
    HttpP4Client,
    HttpP5Client,
    InferenceClient,
    LocalInferenceClient,
    LocalRunner,
    TaskQueueClient,
    VectorStoreClient,
)
from .config import Config
from .core import ArtefactStore, Clock, RealClock, RealRng, Rng
from .dataset_store import DatasetStore
from .registry import Registry
from .tracker import Tracker


@dataclass
class Context:
    cfg: Config
    clock: Clock
    rng: Rng
    artefacts: ArtefactStore
    tracker: Tracker
    registry: Registry
    dataset_store: DatasetStore
    queue: TaskQueueClient
    vector_store: VectorStoreClient
    inference: InferenceClient


def build_context(cfg: Config, clock: Clock | None = None, rng: Rng | None = None) -> Context:
    clock = clock or RealClock()
    rng = rng or RealRng(seed=cfg.train.seed)
    root = Path(cfg.paths.root)
    root.mkdir(parents=True, exist_ok=True)
    artefacts = ArtefactStore(root=root)
    tracker = Tracker(db_path=cfg.paths.tracker_db, artefacts=artefacts, clock=clock)
    registry = Registry(db_path=cfg.paths.registry_db, clock=clock)
    dataset_store = DatasetStore(artefacts=artefacts, manifests_dir=cfg.paths.manifests)

    queue: TaskQueueClient
    if cfg.clients.task_queue == "local":
        queue = LocalRunner(clock=clock)
    else:
        assert cfg.clients.task_queue_url
        queue = HttpP5Client(base_url=cfg.clients.task_queue_url, clock=clock)

    vector_store: VectorStoreClient
    if cfg.clients.vector_store == "faiss_local":
        vector_store = FaissLocalStore(dim=cfg.encoder.dim)
    else:
        assert cfg.clients.vector_store_url
        vector_store = HttpP4Client(base_url=cfg.clients.vector_store_url, dim=cfg.encoder.dim)

    inference: InferenceClient
    if cfg.clients.inference == "local":
        inference = LocalInferenceClient()
    else:
        assert cfg.clients.inference_url
        inference = HttpP2InferenceClient(base_url=cfg.clients.inference_url)

    return Context(
        cfg=cfg,
        clock=clock,
        rng=rng,
        artefacts=artefacts,
        tracker=tracker,
        registry=registry,
        dataset_store=dataset_store,
        queue=queue,
        vector_store=vector_store,
        inference=inference,
    )
