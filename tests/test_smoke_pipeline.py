"""End-to-end smoke test: DAG runs, tracker records the run, registry receives a version.

Uses a fake training step so CI doesn't have to load HF transformers. The
real training path is exercised by tests/test_training_slow.py (marked slow).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mlops_pipeline.clients.task_queue import LocalRunner
from mlops_pipeline.core import ArtefactStore, FakeClock
from mlops_pipeline.dataset_store import DatasetStore, Expectation
from mlops_pipeline.pipeline import Pipeline
from mlops_pipeline.registry import Registry, Stage
from mlops_pipeline.tracker import Tracker


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence": [f"sample {i}" for i in range(20)],
            "label": [i % 2 for i in range(20)],
        }
    )


def test_full_smoke_pipeline(tmp_path: Path) -> None:
    clock = FakeClock(start=1000.0)
    artefacts = ArtefactStore(root=tmp_path / "store")
    tracker = Tracker(db_path=tmp_path / "tracker.sqlite", artefacts=artefacts, clock=clock)
    registry = Registry(db_path=tmp_path / "registry.sqlite", clock=clock)
    dataset_store = DatasetStore(artefacts=artefacts, manifests_dir=tmp_path / "manifests")
    queue = LocalRunner(clock=clock)

    df = _tiny_df()
    shared: dict[str, Any] = {}

    pipe = Pipeline("smoke", queue)

    @pipe.step("ingest")
    def _ingest(config: dict[str, Any], **_: Any) -> dict[str, str]:
        h = dataset_store.ingest(
            name="tiny",
            files={"train": df.head(15), "test": df.tail(5)},
            expectations=[
                Expectation(kind="not_null", column="sentence", args={"max_null_rate": 0.0}),
                Expectation(kind="value_in_set", column="label", args={"values": [0, 1]}),
            ],
            created_at=clock.now(),
        )
        shared["manifest"] = h
        return {"manifest_hash": h}

    @pipe.step("train", depends_on=["ingest"])
    def _train(config: dict[str, Any], **_: Any) -> dict[str, Any]:
        run = tracker.start_run("smoke", config=config)
        run.log_param("dataset", shared["manifest"])
        run.log_metric("train/loss", 0.5)
        blob = run.log_artefact("model.bin", b"fake-model-bytes")
        shared["run_id"] = run.id
        shared["blob"] = blob
        run.finish("success")
        return {"run_id": run.id, "blob": blob}

    @pipe.step("evaluate", depends_on=["train"])
    def _eval(config: dict[str, Any], **_: Any) -> dict[str, float]:
        metrics = {"accuracy": 0.75, "f1_macro": 0.72}
        for k, v in metrics.items():
            tracker._log_metric(shared["run_id"], f"eval/{k}", v, step=0)
        shared["metrics"] = metrics
        return metrics

    @pipe.step("register", depends_on=["evaluate"])
    def _register(config: dict[str, Any], **_: Any) -> dict[str, Any]:
        v = registry.register("classifier", run_id=shared["run_id"], blob_hash=shared["blob"])
        return {"version": v, "metrics": shared["metrics"]}

    results = pipe.run_and_wait(config={"epochs": 1})
    assert results["register"]["version"] == 1

    # State assertions
    runs = tracker.list_runs()
    assert len(runs) == 1 and runs[0].status == "success"
    metrics = tracker.get_metrics(runs[0].id)
    assert metrics["eval/accuracy"][0][1] == 0.75

    models = registry.list()
    assert len(models) == 1
    registry.promote("classifier", 1, Stage.STAGING)
    registry.promote("classifier", 1, Stage.PRODUCTION)
    assert registry.get_production("classifier").version == 1  # type: ignore[union-attr]
