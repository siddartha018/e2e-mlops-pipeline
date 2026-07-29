"""Task queue client: interface + LocalRunner (topo DAG executor) + HttpP5 stub.

LocalRunner runs today. HttpP5Client is a thin stub that will connect to the
from-scratch P5 distributed task queue when it lands.

Contract:
- submit(spec, depends_on) -> job_id
- wait(job_id, timeout) -> JobStatus with result payload or error
- cancel(job_id) -> bool
- Idempotency: submitting the same idempotency_key twice returns the first job_id.
- Retries: max_retries with exponential backoff, dead-letter on exhaustion.
"""
from __future__ import annotations

import logging
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

import httpx

from ..core import Clock

log = logging.getLogger(__name__)


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DEAD = "dead"
    CANCELLED = "cancelled"


@dataclass
class JobSpec:
    """A submittable unit of work. `fn_name` maps to a registered callable."""
    fn_name: str
    args: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    idempotency_key: str | None = None
    timeout_s: float = 3600.0


@dataclass
class JobRecord:
    id: str
    spec: JobSpec
    depends_on: list[str] = field(default_factory=list)
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: str | None = None
    attempts: int = 0
    submitted_at: float = 0.0
    finished_at: float | None = None


class TaskQueueClient(ABC):
    @abstractmethod
    def submit(self, spec: JobSpec, depends_on: list[str] | None = None) -> str: ...

    @abstractmethod
    def wait(self, job_id: str, timeout_s: float = 3600.0) -> JobRecord: ...

    @abstractmethod
    def status(self, job_id: str) -> JobStatus: ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool: ...

    @abstractmethod
    def register(self, fn_name: str, fn: Callable[..., Any]) -> None: ...


class LocalRunner(TaskQueueClient):
    """In-process topological DAG executor.

    Not concurrent — runs jobs in dependency order on the caller's thread.
    Good enough for the pipeline smoke test and for demoing the pipeline
    layer without depending on P5.
    """

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self._registry: dict[str, Callable[..., Any]] = {}
        self._jobs: dict[str, JobRecord] = {}
        self._by_idem: dict[str, str] = {}

    def register(self, fn_name: str, fn: Callable[..., Any]) -> None:
        self._registry[fn_name] = fn

    def submit(self, spec: JobSpec, depends_on: list[str] | None = None) -> str:
        if spec.idempotency_key and spec.idempotency_key in self._by_idem:
            return self._by_idem[spec.idempotency_key]
        job_id = uuid.uuid4().hex
        rec = JobRecord(
            id=job_id,
            spec=spec,
            depends_on=list(depends_on or []),
            submitted_at=self.clock.now(),
        )
        self._jobs[job_id] = rec
        if spec.idempotency_key:
            self._by_idem[spec.idempotency_key] = job_id
        return job_id

    def wait(self, job_id: str, timeout_s: float = 3600.0) -> JobRecord:
        self._run_until_complete(job_id)
        return self._jobs[job_id]

    def status(self, job_id: str) -> JobStatus:
        return self._jobs[job_id].status

    def cancel(self, job_id: str) -> bool:
        rec = self._jobs.get(job_id)
        if rec is None or rec.status in (JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.DEAD):
            return False
        rec.status = JobStatus.CANCELLED
        return True

    def _run_until_complete(self, target: str) -> None:
        order = self._topo_order(target)
        for jid in order:
            rec = self._jobs[jid]
            if rec.status in (JobStatus.SUCCESS, JobStatus.DEAD, JobStatus.CANCELLED):
                continue
            self._run_one(rec)

    def _topo_order(self, target: str) -> list[str]:
        indeg: dict[str, int] = defaultdict(int)
        graph: dict[str, list[str]] = defaultdict(list)
        seen: set[str] = set()
        stack = [target]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            rec = self._jobs[n]
            for dep in rec.depends_on:
                graph[dep].append(n)
                indeg[n] += 1
                stack.append(dep)
        q: deque[str] = deque(n for n in seen if indeg[n] == 0)
        order: list[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for m in graph[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    q.append(m)
        if len(order) != len(seen):
            raise RuntimeError("cycle detected in job DAG")
        return order

    def _run_one(self, rec: JobRecord) -> None:
        for dep_id in rec.depends_on:
            dep = self._jobs[dep_id]
            if dep.status != JobStatus.SUCCESS:
                rec.status = JobStatus.DEAD
                rec.error = f"upstream {dep_id} did not succeed ({dep.status})"
                rec.finished_at = self.clock.now()
                return

        fn = self._registry.get(rec.spec.fn_name)
        if fn is None:
            rec.status = JobStatus.DEAD
            rec.error = f"unknown fn: {rec.spec.fn_name}"
            rec.finished_at = self.clock.now()
            return

        dep_results = {dep: self._jobs[dep].result for dep in rec.depends_on}
        args = {**rec.spec.args, "_deps": dep_results}

        backoff = 1.0
        while rec.attempts <= rec.spec.max_retries:
            rec.attempts += 1
            rec.status = JobStatus.RUNNING
            try:
                rec.result = fn(**args)
                rec.status = JobStatus.SUCCESS
                rec.error = None
                rec.finished_at = self.clock.now()
                return
            except Exception as e:
                rec.error = f"{e}\n{traceback.format_exc()}"
                log.warning("job %s attempt %d failed: %s", rec.id, rec.attempts, e)
                if rec.attempts > rec.spec.max_retries:
                    break
                self.clock.sleep(backoff)
                backoff *= 2
        rec.status = JobStatus.DEAD
        rec.finished_at = self.clock.now()


class HttpP5Client(TaskQueueClient):
    """Stub HTTP client for the from-scratch P5 task queue.

    Assumes P5 exposes:
      POST /jobs            -> {job_id}
      GET  /jobs/{id}       -> {status, result, error, attempts}
      POST /jobs/{id}/cancel

    Not usable until P5 is up.
    """

    def __init__(self, base_url: str, clock: Clock) -> None:
        self.base_url = base_url.rstrip("/")
        self.clock = clock
        self._registry: dict[str, Callable[..., Any]] = {}
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def register(self, fn_name: str, fn: Callable[..., Any]) -> None:
        self._registry[fn_name] = fn

    def submit(self, spec: JobSpec, depends_on: list[str] | None = None) -> str:
        r = self._client.post(
            "/jobs",
            json={
                "fn_name": spec.fn_name,
                "args": spec.args,
                "max_retries": spec.max_retries,
                "idempotency_key": spec.idempotency_key,
                "timeout_s": spec.timeout_s,
                "depends_on": depends_on or [],
            },
        )
        r.raise_for_status()
        return str(r.json()["job_id"])

    def wait(self, job_id: str, timeout_s: float = 3600.0) -> JobRecord:
        deadline = self.clock.now() + timeout_s
        while self.clock.now() < deadline:
            rec = self._get(job_id)
            if rec.status in (JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.DEAD, JobStatus.CANCELLED):
                return rec
            self.clock.sleep(1.0)
        raise TimeoutError(f"job {job_id} did not complete in {timeout_s}s")

    def status(self, job_id: str) -> JobStatus:
        return self._get(job_id).status

    def cancel(self, job_id: str) -> bool:
        r = self._client.post(f"/jobs/{job_id}/cancel")
        return r.status_code == 200

    def _get(self, job_id: str) -> JobRecord:
        r = self._client.get(f"/jobs/{job_id}")
        r.raise_for_status()
        body = r.json()
        return JobRecord(
            id=job_id,
            spec=JobSpec(fn_name=body.get("fn_name", "?")),
            status=JobStatus(body["status"]),
            result=body.get("result"),
            error=body.get("error"),
            attempts=body.get("attempts", 0),
        )
