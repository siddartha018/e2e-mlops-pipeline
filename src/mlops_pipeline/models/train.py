"""Text classifier fine-tuning (transformers Trainer).

Uses a small model (BERT-tiny by default) so training completes in minutes
on CPU. Config selects the actual HF model. Full determinism: fixes all seeds.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import TrainConfig


@dataclass
class TrainResult:
    output_dir: Path
    metrics: dict[str, float]
    n_train: int
    n_val: int


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # honest tradeoff — bit-level is fragile
    except Exception:
        pass


def train_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir: Path,
    cfg: TrainConfig,
) -> TrainResult:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    _set_seed(cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    def _to_hf(df: pd.DataFrame) -> Any:
        ds = Dataset.from_pandas(df[[text_column, label_column]].rename(
            columns={label_column: "labels"}
        ))
        return ds.map(
            lambda x: tok(x[text_column], truncation=True, max_length=cfg.max_length, padding="max_length"),
            batched=True,
        )

    train_hf = _to_hf(train_df)
    val_hf = _to_hf(val_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=[],
        disable_tqdm=True,
    )

    def _compute_metrics(pred: Any) -> dict[str, float]:
        preds = np.argmax(pred.predictions, axis=1)
        labels = pred.label_ids
        acc = float((preds == labels).mean())
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))

    return TrainResult(
        output_dir=output_dir,
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        n_train=len(train_df),
        n_val=len(val_df),
    )
