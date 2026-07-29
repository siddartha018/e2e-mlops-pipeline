"""Inference client: interface + local HF impl + HTTP P2 stub."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class Prediction:
    label: str
    score: float
    all_scores: dict[str, float]


class InferenceClient(ABC):
    @abstractmethod
    def load_model(self, name: str, version: int, blob_path: Path) -> None: ...

    @abstractmethod
    def unload(self, name: str, version: int) -> None: ...

    @abstractmethod
    def predict(self, name: str, version: int | None, inputs: list[str]) -> list[Prediction]: ...

    @abstractmethod
    def list_loaded(self) -> list[tuple[str, int]]: ...


class LocalInferenceClient(InferenceClient):
    """In-process transformers pipeline.

    Lazy-imports torch/transformers so tests that don't need real inference
    don't pay the import cost.
    """

    def __init__(self) -> None:
        self._loaded: dict[tuple[str, int], Any] = {}
        self._active: dict[str, int] = {}

    def load_model(self, name: str, version: int, blob_path: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        model = AutoModelForSequenceClassification.from_pretrained(str(blob_path))
        tok = AutoTokenizer.from_pretrained(str(blob_path))
        clf = pipeline("text-classification", model=model, tokenizer=tok, top_k=None)
        self._loaded[(name, version)] = clf
        self._active[name] = version

    def unload(self, name: str, version: int) -> None:
        self._loaded.pop((name, version), None)
        if self._active.get(name) == version:
            self._active.pop(name, None)

    def predict(self, name: str, version: int | None, inputs: list[str]) -> list[Prediction]:
        v = version if version is not None else self._active.get(name)
        if v is None:
            raise KeyError(f"no active version for model {name}")
        clf = self._loaded.get((name, v))
        if clf is None:
            raise KeyError(f"model {name} v{v} not loaded")
        results = clf(inputs)
        out: list[Prediction] = []
        for r in results:
            all_scores = {item["label"]: float(item["score"]) for item in r}
            top = max(all_scores.items(), key=lambda kv: kv[1])
            out.append(Prediction(label=top[0], score=top[1], all_scores=all_scores))
        return out

    def list_loaded(self) -> list[tuple[str, int]]:
        return list(self._loaded.keys())


class HttpP2InferenceClient(InferenceClient):
    """Stub HTTP client for the from-scratch P2 CPU inference server.

    Assumes P2 exposes:
      POST /models/{name}/versions/{v}/load   {blob_path or blob_hash}
      POST /models/{name}/versions/{v}/unload
      POST /predict                           {model, version, inputs}
      GET  /models                            -> [{name, version}]
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=60.0)

    def load_model(self, name: str, version: int, blob_path: Path) -> None:
        r = self._client.post(
            f"/models/{name}/versions/{version}/load",
            json={"blob_path": str(blob_path)},
        )
        r.raise_for_status()

    def unload(self, name: str, version: int) -> None:
        r = self._client.post(f"/models/{name}/versions/{version}/unload")
        r.raise_for_status()

    def predict(self, name: str, version: int | None, inputs: list[str]) -> list[Prediction]:
        r = self._client.post(
            "/predict", json={"model": name, "version": version, "inputs": inputs}
        )
        r.raise_for_status()
        return [Prediction(**p) for p in r.json()["predictions"]]

    def list_loaded(self) -> list[tuple[str, int]]:
        r = self._client.get("/models")
        r.raise_for_status()
        return [(m["name"], m["version"]) for m in r.json()]
