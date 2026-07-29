"""Content-addressed dataset store with manifest-based versioning.

A dataset version is a manifest: {name, files: [(logical_name, blob_hash, size, schema)]}.
Manifest itself is content-addressed. Manifest hash IS the dataset version.

Also does data validation on ingest (schema, null rate, class balance).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from .core import ArtefactStore


class Expectation(BaseModel):
    """Steal Great Expectations vocabulary for a small subset."""
    kind: str
    column: str | None = None
    args: dict[str, Any] = {}

    def check(self, df: pd.DataFrame) -> tuple[bool, str]:
        if self.kind == "columns_match":
            expected = set(self.args["columns"])
            actual = set(df.columns)
            if expected != actual:
                return False, f"columns mismatch: expected {expected}, got {actual}"
            return True, "ok"
        if self.kind == "not_null":
            col = self.column
            assert col is not None
            max_rate = self.args.get("max_null_rate", 0.0)
            rate = df[col].isna().mean() if col in df.columns else 1.0
            if rate > max_rate:
                return False, f"{col} null rate {rate:.4f} > {max_rate}"
            return True, "ok"
        if self.kind == "value_in_set":
            col = self.column
            assert col is not None
            allowed = set(self.args["values"])
            bad = set(df[col].dropna().unique()) - allowed
            if bad:
                return False, f"{col} has unexpected values: {bad}"
            return True, "ok"
        if self.kind == "value_between":
            col = self.column
            assert col is not None
            lo, hi = self.args["min"], self.args["max"]
            s = df[col].dropna()
            if len(s) and (s.min() < lo or s.max() > hi):
                return False, f"{col} out of [{lo}, {hi}]: min={s.min()}, max={s.max()}"
            return True, "ok"
        if self.kind == "row_count_between":
            lo, hi = self.args["min"], self.args["max"]
            n = len(df)
            if not (lo <= n <= hi):
                return False, f"row count {n} not in [{lo}, {hi}]"
            return True, "ok"
        if self.kind == "class_balance":
            col = self.column
            assert col is not None
            min_frac = self.args.get("min_fraction", 0.0)
            counts = df[col].value_counts(normalize=True)
            worst = counts.min() if len(counts) else 0.0
            if worst < min_frac:
                return False, f"{col} class balance worst={worst:.4f} < {min_frac}"
            return True, "ok"
        return False, f"unknown expectation kind: {self.kind}"


class ValidationReport(BaseModel):
    passed: bool
    checks: list[dict[str, Any]] = []

    @property
    def summary(self) -> str:
        return f"{sum(1 for c in self.checks if c['ok'])}/{len(self.checks)} checks passed"


class DatasetSchema(BaseModel):
    columns: list[str]
    dtypes: dict[str, str] = {}


class Manifest(BaseModel):
    model_config = {"protected_namespaces": ()}

    name: str
    files: list[dict[str, Any]] = []
    dataset_schema: DatasetSchema | None = None
    row_count: int = 0
    created_at: float = 0.0
    parent: str | None = None

    @property
    def manifest_hash(self) -> str:
        import hashlib
        payload = self.model_dump_json().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass
class DatasetView:
    name: str
    manifest_hash: str
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)


class DataValidationError(Exception):
    pass


class DatasetStore:
    def __init__(self, artefacts: ArtefactStore, manifests_dir: Path) -> None:
        self.artefacts = artefacts
        self.manifests_dir = Path(manifests_dir)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        name: str,
        files: dict[str, pd.DataFrame],
        expectations: list[Expectation] | None = None,
        created_at: float = 0.0,
        parent: str | None = None,
    ) -> str:
        expectations = expectations or []
        results: list[dict[str, Any]] = []
        total_rows = 0
        for logical_name, df in files.items():
            for exp in expectations:
                ok, msg = exp.check(df)
                results.append({"file": logical_name, "kind": exp.kind, "ok": ok, "msg": msg})
                if not ok:
                    raise DataValidationError(f"validation failed on {logical_name}: {msg}")
            total_rows += len(df)

        first_df = next(iter(files.values()))
        ds_schema = DatasetSchema(
            columns=list(first_df.columns),
            dtypes={c: str(first_df[c].dtype) for c in first_df.columns},
        )

        entries: list[dict[str, Any]] = []
        for logical_name, df in files.items():
            payload = df.to_parquet(index=False)
            blob_hash = self.artefacts.put(payload)
            entries.append(
                {
                    "logical_name": logical_name,
                    "blob_hash": blob_hash,
                    "size": len(payload),
                    "rows": len(df),
                }
            )

        manifest = Manifest(
            name=name,
            files=entries,
            dataset_schema=ds_schema,
            row_count=total_rows,
            created_at=created_at,
            parent=parent,
        )
        manifest_hash = manifest.manifest_hash
        (self.manifests_dir / f"{name}-{manifest_hash[:12]}.json").write_text(
            manifest.model_dump_json(indent=2)
        )
        return manifest_hash

    def load(self, name: str, manifest_hash: str) -> DatasetView:
        manifest_path = self._find_manifest(name, manifest_hash)
        manifest = Manifest.model_validate_json(manifest_path.read_text())
        frames: dict[str, pd.DataFrame] = {}
        import io
        for entry in manifest.files:
            data = self.artefacts.get(entry["blob_hash"])
            frames[entry["logical_name"]] = pd.read_parquet(io.BytesIO(data))
        return DatasetView(name=name, manifest_hash=manifest_hash, frames=frames)

    def list_versions(self, name: str) -> list[str]:
        out: list[str] = []
        for p in self.manifests_dir.glob(f"{name}-*.json"):
            manifest = Manifest.model_validate_json(p.read_text())
            out.append(manifest.manifest_hash)
        return out

    def _find_manifest(self, name: str, manifest_hash: str) -> Path:
        prefix = manifest_hash[:12]
        expected = self.manifests_dir / f"{name}-{prefix}.json"
        if expected.exists():
            return expected
        for p in self.manifests_dir.glob(f"{name}-*.json"):
            manifest = Manifest.model_validate_json(p.read_text())
            if manifest.manifest_hash == manifest_hash:
                return p
        raise KeyError(f"manifest {name}/{manifest_hash} not found")
