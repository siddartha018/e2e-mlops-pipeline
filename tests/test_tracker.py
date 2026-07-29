from __future__ import annotations

from mlops_pipeline.tracker import Tracker


def test_start_finish_run(tracker: Tracker) -> None:
    run = tracker.start_run("smoke", config={"lr": 0.01, "epochs": 2})
    run.log_metric("loss", 0.5, step=0)
    run.log_metric("loss", 0.3, step=1)
    run.log_param("optim", "adamw")
    run.log_artefact("readme", b"hello")
    run.finish("success")

    r = tracker.get_run(run.id)
    assert r.status == "success"
    assert r.finished_at is not None
    assert r.config_hash is not None

    params = tracker.get_params(run.id)
    assert params["lr"] == 0.01
    assert params["epochs"] == 2
    assert params["optim"] == "adamw"

    metrics = tracker.get_metrics(run.id)
    assert metrics["loss"] == [(0, 0.5), (1, 0.3)]

    artefacts = tracker.get_artefacts(run.id)
    assert "readme" in artefacts


def test_list_runs_ordering(tracker: Tracker) -> None:
    r1 = tracker.start_run("a")
    r1.finish()
    tracker.clock.advance(10)  # type: ignore[attr-defined]
    r2 = tracker.start_run("a")
    r2.finish()
    runs = tracker.list_runs(name="a")
    assert [r.id for r in runs] == [r2.id, r1.id]
