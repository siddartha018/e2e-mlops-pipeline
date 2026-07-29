"""Model registry: promotion state machine over the tracker/artefact store.

State machine: none -> staging -> production -> archived.
Invariant: at most one 'production' row per model name.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterator

from .core import Clock

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    blob_hash TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'none',
    promoted_at REAL,
    created_at REAL NOT NULL,
    PRIMARY KEY (name, version)
);
CREATE INDEX IF NOT EXISTS idx_models_name_stage ON models(name, stage);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_production
    ON models(name) WHERE stage = 'production';
"""


class Stage(StrEnum):
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


VALID_TRANSITIONS = {
    Stage.NONE: {Stage.STAGING, Stage.ARCHIVED},
    Stage.STAGING: {Stage.PRODUCTION, Stage.ARCHIVED, Stage.NONE},
    Stage.PRODUCTION: {Stage.ARCHIVED, Stage.STAGING},
    Stage.ARCHIVED: set(),
}


@dataclass
class ModelVersion:
    name: str
    version: int
    run_id: str
    blob_hash: str
    stage: Stage
    promoted_at: float | None
    created_at: float


class RegistryError(Exception):
    pass


class Registry:
    def __init__(self, db_path: Path, clock: Clock) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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

    def register(self, name: str, run_id: str, blob_hash: str) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(version), 0) FROM models WHERE name = ?", (name,)
            ).fetchone()
            next_version = int(row[0]) + 1
            c.execute(
                "INSERT INTO models (name, version, run_id, blob_hash, stage, created_at) "
                "VALUES (?, ?, ?, ?, 'none', ?)",
                (name, next_version, run_id, blob_hash, self.clock.now()),
            )
        return next_version

    def promote(self, name: str, version: int, target: Stage) -> None:
        with self._conn() as c:
            row = c.execute(
                "SELECT stage FROM models WHERE name = ? AND version = ?", (name, version)
            ).fetchone()
            if row is None:
                raise RegistryError(f"model {name} v{version} not found")
            current = Stage(row[0])
            if target not in VALID_TRANSITIONS[current]:
                raise RegistryError(
                    f"invalid transition {current} -> {target} for {name} v{version}"
                )
            if target == Stage.PRODUCTION:
                c.execute(
                    "UPDATE models SET stage = 'archived' WHERE name = ? AND stage = 'production'",
                    (name,),
                )
            c.execute(
                "UPDATE models SET stage = ?, promoted_at = ? WHERE name = ? AND version = ?",
                (target.value, self.clock.now(), name, version),
            )

    def get(self, name: str, version: int) -> ModelVersion:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM models WHERE name = ? AND version = ?", (name, version)
            ).fetchone()
        if row is None:
            raise RegistryError(f"model {name} v{version} not found")
        return _row_to_mv(row)

    def get_production(self, name: str) -> ModelVersion | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM models WHERE name = ? AND stage = 'production'", (name,)
            ).fetchone()
        return _row_to_mv(row) if row else None

    def list(self, name: str | None = None) -> list[ModelVersion]:
        with self._conn() as c:
            if name:
                rows = c.execute(
                    "SELECT * FROM models WHERE name = ? ORDER BY version DESC", (name,)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM models ORDER BY name, version DESC"
                ).fetchall()
        return [_row_to_mv(r) for r in rows]


def _row_to_mv(row: tuple) -> ModelVersion:
    name, version, run_id, blob_hash, stage, promoted_at, created_at = row
    return ModelVersion(
        name=name,
        version=version,
        run_id=run_id,
        blob_hash=blob_hash,
        stage=Stage(stage),
        promoted_at=promoted_at,
        created_at=created_at,
    )
