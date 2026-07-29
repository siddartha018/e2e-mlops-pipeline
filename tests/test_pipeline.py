from __future__ import annotations

from typing import Any

import pytest

from mlops_pipeline.clients.task_queue import JobStatus, LocalRunner
from mlops_pipeline.core import FakeClock
from mlops_pipeline.pipeline import Pipeline


def test_simple_linear_dag() -> None:
    clock = FakeClock()
    runner = LocalRunner(clock)
    pipe = Pipeline("t", runner)

    calls: list[str] = []

    @pipe.step("a")
    def a(config: dict[str, Any], **_: Any) -> int:
        calls.append("a")
        return 1

    @pipe.step("b", depends_on=["a"])
    def b(config: dict[str, Any], _deps: dict[str, Any], **_: Any) -> int:
        calls.append("b")
        return _deps["a-dep"] if "a-dep" in _deps else 2

    @pipe.step("c", depends_on=["b"])
    def c(config: dict[str, Any], **_: Any) -> str:
        calls.append("c")
        return "done"

    results = pipe.run_and_wait(config={})
    assert calls == ["a", "b", "c"]
    assert results["c"] == "done"


def test_fanout_fanin() -> None:
    clock = FakeClock()
    runner = LocalRunner(clock)
    pipe = Pipeline("t", runner)

    @pipe.step("root")
    def root(config: dict[str, Any], **_: Any) -> int:
        return 10

    @pipe.step("left", depends_on=["root"])
    def left(config: dict[str, Any], _deps: dict[str, Any]) -> int:
        return _deps[list(_deps.keys())[0]] * 2

    @pipe.step("right", depends_on=["root"])
    def right(config: dict[str, Any], _deps: dict[str, Any]) -> int:
        return _deps[list(_deps.keys())[0]] + 5

    @pipe.step("join", depends_on=["left", "right"])
    def join(config: dict[str, Any], _deps: dict[str, Any]) -> int:
        return sum(_deps.values())

    results = pipe.run_and_wait(config={})
    assert results["join"] == 20 + 15


def test_retry_then_success() -> None:
    clock = FakeClock()
    runner = LocalRunner(clock)
    pipe = Pipeline("t", runner)

    attempts = {"n": 0}

    @pipe.step("flaky", max_retries=3)
    def flaky(config: dict[str, Any], **_: Any) -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    result = pipe.run_and_wait(config={})
    assert result["flaky"] == "ok"
    assert attempts["n"] == 3


def test_retry_exhausted_kills_downstream() -> None:
    clock = FakeClock()
    runner = LocalRunner(clock)
    pipe = Pipeline("t", runner)

    @pipe.step("bad", max_retries=1)
    def bad(config: dict[str, Any], **_: Any) -> str:
        raise RuntimeError("nope")

    @pipe.step("downstream", depends_on=["bad"])
    def downstream(config: dict[str, Any], **_: Any) -> str:
        return "never"

    with pytest.raises(RuntimeError, match="did not succeed"):
        pipe.run_and_wait(config={})


def test_idempotency() -> None:
    clock = FakeClock()
    runner = LocalRunner(clock)
    pipe = Pipeline("t", runner)

    calls = {"n": 0}

    @pipe.step("a")
    def a(config: dict[str, Any], **_: Any) -> int:
        calls["n"] += 1
        return 42

    ids_1 = pipe.run(config={"x": 1})
    ids_2 = pipe.run(config={"x": 1})
    # Same config -> same idempotency key -> same job id -> not re-executed.
    assert ids_1 == ids_2

    # Different config -> different key -> new job.
    ids_3 = pipe.run(config={"x": 2})
    assert ids_3 != ids_1


def test_cycle_detected() -> None:
    clock = FakeClock()
    runner = LocalRunner(clock)
    pipe = Pipeline("t", runner)

    @pipe.step("a", depends_on=["b"])
    def a(config: dict[str, Any], **_: Any) -> int:
        return 1

    @pipe.step("b", depends_on=["a"])
    def b(config: dict[str, Any], **_: Any) -> int:
        return 2

    with pytest.raises(RuntimeError, match="cycle"):
        pipe.run(config={})
