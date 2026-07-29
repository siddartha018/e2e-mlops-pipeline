"""Typed configuration via Pydantic. Loaded from YAML, overridable via CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    name: str = "sst2"
    dataset_source: Literal["huggingface", "csv"] = "huggingface"
    hf_name: str = "sst2"
    text_column: str = "sentence"
    label_column: str = "label"
    train_size: int = 5000
    val_size: int = 500
    test_size: int = 500
    random_seed: int = 42


class TrainConfig(BaseModel):
    model_name: str = "prajjwal1/bert-tiny"
    max_length: int = 128
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    num_labels: int = 2
    seed: int = 42


class EncoderConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384
    batch_size: int = 32


class DriftConfig(BaseModel):
    psi_threshold: float = 0.2
    wasserstein_threshold: float = 0.15
    cosine_threshold: float = 0.15
    window_size: int = 200


class PromoteConfig(BaseModel):
    metric: str = "accuracy"
    higher_is_better: bool = True
    min_improvement: float = 0.0


class PathsConfig(BaseModel):
    root: Path = Path(".mlops")
    manifests: Path = Path(".mlops/manifests")
    tracker_db: Path = Path(".mlops/tracker.sqlite")
    registry_db: Path = Path(".mlops/registry.sqlite")


class ClientsConfig(BaseModel):
    task_queue: Literal["local", "http_p5"] = "local"
    task_queue_url: str | None = None
    vector_store: Literal["faiss_local", "http_p4"] = "faiss_local"
    vector_store_url: str | None = None
    inference: Literal["local", "http_p2"] = "local"
    inference_url: str | None = None


class Config(BaseModel):
    project: str = "mlops-pipeline"
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    promote: PromoteConfig = Field(default_factory=PromoteConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    clients: ClientsConfig = Field(default_factory=ClientsConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        if path is None:
            return cls()
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
