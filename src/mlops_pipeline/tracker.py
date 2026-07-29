"""Experiment tracker: SQLite-backed run log with artefact linkage.

Schema is intentionally minimal:
- runs(id, name, status, started_at, finished_at, git_sha, config_hash)
- params(run_id, key, value)
- metrics(run_id, key, value, step)
- artefacts(run_id, name, blob_hash)

Every metric/param/artefact is scoped to a run. Artefact bytes live in
ArtefactStore; the tracker only stores the hash + a logical name.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .core import ArtefactStore, Clock, content_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    git_sha TEXT,
    config_hash TEXT
);
CREATE TABLE IF NOT EXISTS params (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER NOT NULL DEFAULT 0,
    ts REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
CREATE TABLE IF NOT EXISTS artefacts (
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    blob_hash TEXT NOT NULL,
    PRIMARY KEY (run_id, name),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


@dataclass
class Run:
    id: str
    name: str
    status: str
    started_at: float
    finished_at: float | None
    git_sha: str | None
    config_hash: str | None


class RunHandle:
    def __init__(self, tracker: "Tracker", run_id: str) -> None:
        self._tracker = tracker
        self.id = run_id

    def log_param(self, key: str, value: Any) -> None:
        self._tracker._log_param(self.id, key, value)

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        self._tracker._log_metric(self.id, key, value, step)

    def log_artefact(self, name: str, data: bytes) -> str:
        return self._tracker._log_artefact(self.id, name, data)

    def log_artefact_file(self, name: str, path: Path) -> str:
        return self._tracker._log_artefact_file(self.id, name, path)

    def finish(self, status: str = "success") -> None:
        self._tracker._finish(self.id, status)


class Tracker:
    def __init__(self, db_path: Path, artefacts: ArtefactStore, clock: Clock) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artefacts = artefacts
        self.clock = clock
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def start_run(self, name: str, config: dict[str, Any] | None = None) -> RunHandle:
        run_id = uuid.uuid4().hex
        cfg_hash = None
        if config is not None:
            cfg_json = json.dumps(config, sort_keys=True)
            cfg_hash = content_hash(cfg_json)
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (id, name, status, started_at, git_sha, config_hash) "
                "VALUES (?, ?, 'running', ?, ?, ?)",
                (run_id, name, self.clock.now(), _git_sha(), cfg_hash),
            )
            if config is not None:
                for k, v in _flatten(config).items():
                    c.execute(
                        "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
                        (run_id, k, json.dumps(v)),
                    )
        return RunHandle(self, run_id)

    def _log_param(self, run_id: str, key: str, value: Any) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
                (run_id, key, json.dumps(value)),
            )

    def _log_metric(self, run_id: str, key: str, value: float, step: int) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO metrics (run_id, key, value, step, ts) VALUES (?, ?, ?, ?, ?)",
                (run_id, key, value, step, self.clock.now()),
            )

    def _log_artefact(self, run_id: str, name: str, data: bytes) -> str:
        digest = self.artefacts.put(data)
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO artefacts (run_id, name, blob_hash) VALUES (?, ?, ?)",
                (run_id, name, digest),
            )
        return digest

    def _log_artefact_file(self, run_id: str, name: str, path: Path) -> str:
        digest = self.artefacts.put_file(path)
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO artefacts (run_id, name, blob_hash) VALUES (?, ?, ?)",
                (run_id, name, digest),
            )
        return digest

    def _finish(self, run_id: str, status: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
                (status, self.clock.now(), run_id),
            )

    def get_run(self, run_id: str) -> Run:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run {run_id} not found")
        return Run(*row)

    def list_runs(self, name: str | None = None, limit: int = 50) -> list[Run]:
        with self._conn() as c:
            if name:
                rows = c.execute(
                    "SELECT * FROM runs WHERE name = ? ORDER BY started_at DESC LIMIT ?",
                    (name, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [Run(*r) for r in rows]

    def get_metrics(self, run_id: str) -> dict[str, list[tuple[int, float]]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT key, step, value FROM metrics WHERE run_id = ? ORDER BY key, step",
                (run_id,),
            ).fetchall()
        out: dict[str, list[tuple[int, float]]] = {}
        for key, step, value in rows:
            out.setdefault(key, []).append((step, value))
        return out

    def get_params(self, run_id: str) -> dict[str, Any]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT key, value FROM params WHERE run_id = ?", (run_id,)
            ).fetchall()
        return {k: json.loads(v) for k, v in rows}

    def get_artefacts(self, run_id: str) -> dict[str, str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, blob_hash FROM artefacts WHERE run_id = ?", (run_id,)
            ).fetchall()
        return dict(rows)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
