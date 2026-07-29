"""Pipeline DAG layer over TaskQueueClient.

Steps are decorated Python functions collected into a Pipeline. On run,
we compile the DAG into a sequence of TaskQueueClient.submit(...) calls
with depends_on edges, then wait on the terminal node.

Step caching: each step's idempotency_key = sha256(fn_name, config_hash,
sorted upstream job idempotency_keys). A rerun with identical inputs
returns the same job_id from the queue and skips recomputation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .clients.task_queue import JobSpec, TaskQueueClient


@dataclass
class Step:
    name: str
    fn: Callable[..., Any]
    depends_on: list[str] = field(default_factory=list)
    max_retries: int = 3


class Pipeline:
    def __init__(self, name: str, queue: TaskQueueClient) -> None:
        self.name = name
        self.queue = queue
        self.steps: dict[str, Step] = {}

    def step(
        self,
        name: str,
        depends_on: list[str] | None = None,
        max_retries: int = 3,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.steps[name] = Step(
                name=name, fn=fn, depends_on=list(depends_on or []), max_retries=max_retries
            )
            self.queue.register(f"{self.name}.{name}", fn)
            return fn
        return wrapper

    def run(self, config: dict[str, Any] | None = None) -> dict[str, str]:
        """Submit all steps, honoring dependency edges. Returns {step_name: job_id}."""
        config = config or {}
        cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

        order = self._topo_order()
        job_ids: dict[str, str] = {}
        for step_name in order:
            step = self.steps[step_name]
            dep_ids = [job_ids[d] for d in step.depends_on]
            idem = _idem_key(self.name, step_name, cfg_hash, dep_ids)
            spec = JobSpec(
                fn_name=f"{self.name}.{step_name}",
                args={"config": config},
                max_retries=step.max_retries,
                idempotency_key=idem,
            )
            jid = self.queue.submit(spec, depends_on=dep_ids)
            job_ids[step_name] = jid
        return job_ids

    def run_and_wait(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run + wait on terminal nodes. Returns {step_name: result}."""
        job_ids = self.run(config)
        terminals = self._terminals()
        results: dict[str, Any] = {}
        for term in terminals:
            rec = self.queue.wait(job_ids[term])
            results[term] = rec.result
            if rec.status != "success":
                raise RuntimeError(
                    f"pipeline {self.name} step {term} did not succeed: {rec.status} / {rec.error}"
                )
        return results

    def _topo_order(self) -> list[str]:
        from collections import defaultdict, deque
        indeg: dict[str, int] = defaultdict(int)
        graph: dict[str, list[str]] = defaultdict(list)
        for name, step in self.steps.items():
            for d in step.depends_on:
                if d not in self.steps:
                    raise ValueError(f"step {name} depends on unknown step {d}")
                graph[d].append(name)
                indeg[name] += 1
        q: deque[str] = deque([n for n in self.steps if indeg[n] == 0])
        order: list[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for m in graph[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    q.append(m)
        if len(order) != len(self.steps):
            raise RuntimeError(f"cycle in pipeline {self.name}")
        return order

    def _terminals(self) -> list[str]:
        has_children = set()
        for step in self.steps.values():
            for d in step.depends_on:
                has_children.add(d)
        return [n for n in self.steps if n not in has_children]


def _idem_key(pipeline_name: str, step_name: str, cfg_hash: str, dep_ids: list[str]) -> str:
    h = hashlib.sha256()
    h.update(pipeline_name.encode())
    h.update(step_name.encode())
    h.update(cfg_hash.encode())
    for d in sorted(dep_ids):
        h.update(d.encode())
    return h.hexdigest()
