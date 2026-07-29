"""Frozen-holdout evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EvalResult:
    n: int
    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)


def evaluate_classifier(
    model_dir: Path,
    test_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    batch_size: int = 32,
    max_length: int = 128,
) -> EvalResult:
    from datasets import Dataset
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    ds = Dataset.from_pandas(
        test_df[[text_column, label_column]].rename(columns={label_column: "labels"})
    ).map(
        lambda x: tok(x[text_column], truncation=True, max_length=max_length, padding="max_length"),
        batched=True,
    )
    args = TrainingArguments(
        output_dir="/tmp/eval-noop",
        per_device_eval_batch_size=batch_size,
        report_to=[],
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=args, tokenizer=tok)
    pred_out = trainer.predict(ds)
    preds = np.argmax(pred_out.predictions, axis=1)
    labels = pred_out.label_ids

    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    prec = float(precision_score(labels, preds, average="macro", zero_division=0))
    rec = float(recall_score(labels, preds, average="macro", zero_division=0))
    cm = confusion_matrix(labels, preds).tolist()

    per_class: dict[str, dict[str, float]] = {}
    for cls in sorted(set(labels.tolist())):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        per_class[str(cls)] = {
            "n": int(mask.sum()),
            "accuracy": float((preds[mask] == labels[mask]).mean()),
        }

    return EvalResult(
        n=len(labels),
        metrics={"accuracy": acc, "f1_macro": f1, "precision_macro": prec, "recall_macro": rec},
        per_class=per_class,
        confusion=cm,
    )
