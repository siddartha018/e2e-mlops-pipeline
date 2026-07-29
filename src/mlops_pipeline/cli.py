"""CLI: `mlops-pipeline …`.

Commands mirror the pipeline surface: run, runs, models, promote, drift, serve, demo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import Config
from .context import build_context
from .drift import DriftDetector, DriftThresholds
from .pipelines import build_training_pipeline
from .promote import evaluate_promotion
from .registry import Stage

app = typer.Typer(help="End-to-end MLOps pipeline CLI.", no_args_is_help=True)
console = Console()


def _load_config(config: Optional[Path]) -> Config:
    return Config.load(config)


@app.command()
def run(
    pipeline: str = typer.Argument("train", help="pipeline name: train"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config"),
) -> None:
    """Run a pipeline end-to-end."""
    cfg = _load_config(config)
    ctx = build_context(cfg)
    if pipeline == "train":
        pipe, _ = build_training_pipeline(
            cfg, ctx.queue, ctx.tracker, ctx.registry, ctx.dataset_store, ctx.artefacts, ctx.clock
        )
        results = pipe.run_and_wait(config=cfg.dump())
        console.print_json(json.dumps(results, default=str))
    else:
        raise typer.BadParameter(f"unknown pipeline: {pipeline}")


@app.command("runs")
def runs_cmd(
    action: str = typer.Argument("list"),
    run_id: Optional[str] = typer.Argument(None),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """List, show, or diff runs."""
    cfg = _load_config(config)
    ctx = build_context(cfg)
    if action == "list":
        t = Table("id", "name", "status", "started", "config_hash")
        for r in ctx.tracker.list_runs():
            t.add_row(r.id[:8], r.name, r.status, f"{r.started_at:.0f}", (r.config_hash or "")[:12])
        console.print(t)
    elif action == "show":
        if not run_id:
            raise typer.BadParameter("run_id required")
        r = ctx.tracker.get_run(run_id)
        console.print({"run": r.__dict__})
        console.print({"params": ctx.tracker.get_params(run_id)})
        console.print({"metrics": ctx.tracker.get_metrics(run_id)})
        console.print({"artefacts": ctx.tracker.get_artefacts(run_id)})
    else:
        raise typer.BadParameter(f"unknown runs action: {action}")


@app.command("models")
def models_cmd(
    action: str = typer.Argument("list"),
    name: Optional[str] = typer.Argument(None),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """List, show, register model versions."""
    cfg = _load_config(config)
    ctx = build_context(cfg)
    if action == "list":
        t = Table("name", "version", "stage", "run_id", "blob_hash")
        for m in ctx.registry.list(name):
            t.add_row(m.name, str(m.version), m.stage, m.run_id[:8], m.blob_hash[:12])
        console.print(t)
    else:
        raise typer.BadParameter(f"unknown models action: {action}")


@app.command()
def promote(
    name: str = typer.Argument(...),
    version: int = typer.Argument(...),
    stage: str = typer.Option("production", "--stage"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Promote a model version to a stage. For 'production', runs the champion-vs-challenger gate."""
    cfg = _load_config(config)
    ctx = build_context(cfg)
    target = Stage(stage)
    if target != Stage.PRODUCTION:
        ctx.registry.promote(name, version, target)
        console.print(f"[green]{name} v{version} -> {target}[/green]")
        return

    from .pipelines.training import _extract_model_dir
    from .models import evaluate_classifier
    # We need the frozen holdout for eval. Assume the latest manifest for cfg.data.name.
    versions = ctx.dataset_store.list_versions(cfg.data.name)
    if not versions:
        raise typer.BadParameter("no dataset version available for eval")
    latest = versions[-1]
    view = ctx.dataset_store.load(cfg.data.name, latest)
    test_df = view.frames["test"]

    def _eval(mv: Any) -> float:  # noqa: ANN401
        model_dir = _extract_model_dir(mv.blob_hash, ctx.artefacts)
        try:
            res = evaluate_classifier(
                model_dir=model_dir,
                test_df=test_df,
                text_column=cfg.data.text_column,
                label_column=cfg.data.label_column,
                batch_size=cfg.train.batch_size,
                max_length=cfg.train.max_length,
            )
            return res.metrics[cfg.promote.metric]
        finally:
            import shutil
            shutil.rmtree(model_dir, ignore_errors=True)

    outcome = evaluate_promotion(
        registry=ctx.registry,
        model_name=name,
        challenger_version=version,
        evaluate_fn=_eval,
        cfg=cfg.promote,
    )
    color = "green" if outcome.passed else "red"
    console.print(f"[{color}]{outcome.reason}[/{color}]")


@app.command()
def drift(
    baseline_manifest: str = typer.Argument(...),
    current_manifest: str = typer.Argument(...),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run drift detection between two dataset versions using the encoder + vector store."""
    cfg = _load_config(config)
    ctx = build_context(cfg)
    from .models import Encoder

    baseline_view = ctx.dataset_store.load(cfg.data.name, baseline_manifest)
    current_view = ctx.dataset_store.load(cfg.data.name, current_manifest)

    enc = Encoder(model_name=cfg.encoder.model_name)
    b_texts = baseline_view.frames["train"][cfg.data.text_column].astype(str).tolist()
    c_texts = current_view.frames["train"][cfg.data.text_column].astype(str).tolist()
    b_embs = enc.encode(b_texts, batch_size=cfg.encoder.batch_size)
    c_embs = enc.encode(c_texts, batch_size=cfg.encoder.batch_size)
    b_centroid = b_embs.mean(axis=0)
    c_centroid = c_embs.mean(axis=0)

    det = DriftDetector(
        DriftThresholds(
            psi=cfg.drift.psi_threshold,
            wasserstein=cfg.drift.wasserstein_threshold,
            cosine=cfg.drift.cosine_threshold,
        )
    )
    signals = [
        det.check_embedding_centroid("embedding_centroid", b_centroid, c_centroid),
        det.check_categorical(
            "label_dist",
            baseline_view.frames["train"][cfg.data.label_column],
            current_view.frames["train"][cfg.data.label_column],
        ),
    ]
    report = det.check(signals, baseline_id=baseline_manifest, current_id=current_manifest)
    console.print(report.model_dump_json(indent=2))


@app.command()
def demo(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Run the flagship demo end-to-end (ingest -> train -> eval -> register -> drift check)."""
    cfg = _load_config(config)
    ctx = build_context(cfg)
    pipe, _ = build_training_pipeline(
        cfg, ctx.queue, ctx.tracker, ctx.registry, ctx.dataset_store, ctx.artefacts, ctx.clock
    )
    results = pipe.run_and_wait(config=cfg.dump())
    console.print("[bold green]Pipeline complete[/bold green]")
    console.print_json(json.dumps(results, default=str))


if __name__ == "__main__":
    app()
