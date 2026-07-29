"""Training pipeline: ingest -> validate -> train -> eval -> register.

Each step is a plain function; the pipeline wires them as a DAG. Deps are
passed via the `_deps` kwarg (see LocalRunner.run_one). The pipeline object
threads the config through every step.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ..clients import TaskQueueClient
from ..config import Config
from ..core import ArtefactStore, Clock
from ..dataset_store import DatasetStore, Expectation
from ..models import evaluate_classifier, train_classifier
from ..pipeline import Pipeline
from ..registry import Registry
from ..tracker import Tracker


def build_training_pipeline(
    cfg: Config,
    queue: TaskQueueClient,
    tracker: Tracker,
    registry: Registry,
    dataset_store: DatasetStore,
    artefacts: ArtefactStore,
    clock: Clock,
) -> tuple[Pipeline, dict[str, Any]]:
    """Returns (pipeline, shared_state). shared_state is mutated across steps."""
    pipe = Pipeline(name="train_classifier", queue=queue)
    shared: dict[str, Any] = {}

    @pipe.step("ingest")
    def _ingest(config: dict[str, Any], **_: Any) -> dict[str, Any]:
        train_df, val_df, test_df = _load_dataset(cfg)
        manifest_hash = dataset_store.ingest(
            name=cfg.data.name,
            files={"train": train_df, "val": val_df, "test": test_df},
            expectations=_default_expectations(cfg),
            created_at=clock.now(),
        )
        shared["train_df"] = train_df
        shared["val_df"] = val_df
        shared["test_df"] = test_df
        shared["manifest_hash"] = manifest_hash
        return {"manifest_hash": manifest_hash, "rows": {
            "train": len(train_df), "val": len(val_df), "test": len(test_df)
        }}

    @pipe.step("validate", depends_on=["ingest"])
    def _validate(config: dict[str, Any], **_: Any) -> dict[str, Any]:
        # Data validation happens in ingest (dataset_store.ingest raises on failure).
        # This step is a placeholder for extra invariants (e.g., no train/test overlap).
        train = set(shared["train_df"][cfg.data.text_column].astype(str).tolist())
        test = set(shared["test_df"][cfg.data.text_column].astype(str).tolist())
        overlap = len(train & test)
        return {"train_test_overlap": overlap}

    @pipe.step("train", depends_on=["validate"])
    def _train(config: dict[str, Any], **_: Any) -> dict[str, Any]:
        run = tracker.start_run("train_classifier", config=cfg.dump())
        run.log_param("manifest_hash", shared["manifest_hash"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model"
            result = train_classifier(
                shared["train_df"],
                shared["val_df"],
                text_column=cfg.data.text_column,
                label_column=cfg.data.label_column,
                output_dir=out,
                cfg=cfg.train,
            )
            for k, v in result.metrics.items():
                run.log_metric(f"train/{k}", v)
            model_blob = _archive_model_dir(out, artefacts)
        shared["train_run_id"] = run.id
        shared["model_blob_hash"] = model_blob
        run.log_param("model_blob_hash", model_blob)
        run.finish("success")
        return {"run_id": run.id, "model_blob_hash": model_blob, "metrics": result.metrics}

    @pipe.step("evaluate", depends_on=["train"])
    def _evaluate(config: dict[str, Any], **_: Any) -> dict[str, Any]:
        model_dir = _extract_model_dir(shared["model_blob_hash"], artefacts)
        eval_res = evaluate_classifier(
            model_dir=model_dir,
            test_df=shared["test_df"],
            text_column=cfg.data.text_column,
            label_column=cfg.data.label_column,
            batch_size=cfg.train.batch_size,
            max_length=cfg.train.max_length,
        )
        run = shared.get("train_run_id")
        assert isinstance(run, str)
        for k, v in eval_res.metrics.items():
            tracker._log_metric(run, f"eval/{k}", v, step=0)
        shutil.rmtree(model_dir, ignore_errors=True)
        shared["eval_metrics"] = eval_res.metrics
        return {"metrics": eval_res.metrics, "n": eval_res.n, "confusion": eval_res.confusion}

    @pipe.step("register", depends_on=["evaluate"])
    def _register(config: dict[str, Any], **_: Any) -> dict[str, Any]:
        version = registry.register(
            name="classifier",
            run_id=shared["train_run_id"],
            blob_hash=shared["model_blob_hash"],
        )
        return {
            "model_name": "classifier",
            "version": version,
            "metrics": shared["eval_metrics"],
        }

    return pipe, shared


def _load_dataset(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load SST-2 (or configured source). CSV path used in tests."""
    if cfg.data.dataset_source == "csv":
        raise NotImplementedError("csv source is exercised only in tests via fixtures")
    from datasets import load_dataset
    ds = load_dataset(cfg.data.hf_name)
    train_split = ds["train"] if "train" in ds else ds
    val_split = ds.get("validation") or ds.get("test") or ds["train"]
    train_df = train_split.to_pandas().head(cfg.data.train_size)
    val_df = val_split.to_pandas().head(cfg.data.val_size)
    test_df = val_split.to_pandas().tail(cfg.data.test_size)
    for df in (train_df, val_df, test_df):
        assert cfg.data.text_column in df.columns, f"missing text col {cfg.data.text_column}"
        assert cfg.data.label_column in df.columns, f"missing label col {cfg.data.label_column}"
    return train_df, val_df, test_df


def _default_expectations(cfg: Config) -> list[Expectation]:
    return [
        Expectation(kind="row_count_between", args={"min": 10, "max": 10_000_000}),
        Expectation(kind="not_null", column=cfg.data.text_column, args={"max_null_rate": 0.0}),
        Expectation(kind="not_null", column=cfg.data.label_column, args={"max_null_rate": 0.0}),
        Expectation(
            kind="value_in_set",
            column=cfg.data.label_column,
            args={"values": list(range(cfg.train.num_labels))},
        ),
        Expectation(
            kind="class_balance",
            column=cfg.data.label_column,
            args={"min_fraction": 0.05},
        ),
    ]


def _archive_model_dir(model_dir: Path, artefacts: ArtefactStore) -> str:
    """Tar the model dir; store the tar in the artefact store; return its hash."""
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(model_dir, arcname="model")
    return artefacts.put(buf.getvalue())


def _extract_model_dir(blob_hash: str, artefacts: ArtefactStore) -> Path:
    import io
    import tarfile
    data = artefacts.get(blob_hash)
    out = Path(tempfile.mkdtemp(prefix="mlops-model-"))
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(out)
    return out / "model"
