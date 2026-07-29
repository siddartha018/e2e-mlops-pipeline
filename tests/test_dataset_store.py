from __future__ import annotations

import pandas as pd
import pytest

from mlops_pipeline.dataset_store import DatasetStore, DataValidationError, Expectation


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence": ["good movie", "bad film", "meh", "loved it", "terrible"],
            "label": [1, 0, 0, 1, 0],
        }
    )


def test_ingest_and_load(dataset_store: DatasetStore) -> None:
    df = _sample_df()
    manifest_hash = dataset_store.ingest(name="sst2", files={"train": df})
    assert len(manifest_hash) == 64
    view = dataset_store.load("sst2", manifest_hash)
    assert set(view.frames.keys()) == {"train"}
    pd.testing.assert_frame_equal(view.frames["train"], df)


def test_ingest_deterministic_hash(dataset_store: DatasetStore) -> None:
    df = _sample_df()
    h1 = dataset_store.ingest(name="sst2", files={"train": df}, created_at=100.0)
    h2 = dataset_store.ingest(name="sst2", files={"train": df}, created_at=100.0)
    assert h1 == h2


def test_expectation_columns_match(dataset_store: DatasetStore) -> None:
    df = _sample_df()
    ok_exp = [Expectation(kind="columns_match", args={"columns": ["sentence", "label"]})]
    dataset_store.ingest(name="sst2", files={"train": df}, expectations=ok_exp)

    bad_exp = [Expectation(kind="columns_match", args={"columns": ["sentence"]})]
    with pytest.raises(DataValidationError):
        dataset_store.ingest(name="sst2-bad", files={"train": df}, expectations=bad_exp)


def test_expectation_class_balance(dataset_store: DatasetStore) -> None:
    skewed = pd.DataFrame({"sentence": ["x"] * 10, "label": [0] * 9 + [1]})
    exp = [Expectation(kind="class_balance", column="label", args={"min_fraction": 0.2})]
    with pytest.raises(DataValidationError):
        dataset_store.ingest(name="skewed", files={"train": skewed}, expectations=exp)


def test_expectation_value_in_set(dataset_store: DatasetStore) -> None:
    df = pd.DataFrame({"sentence": ["a", "b"], "label": [0, 3]})
    exp = [Expectation(kind="value_in_set", column="label", args={"values": [0, 1]})]
    with pytest.raises(DataValidationError):
        dataset_store.ingest(name="bad", files={"train": df}, expectations=exp)
