from __future__ import annotations

import numpy as np
import pytest

from mlops_pipeline.clients.vector_store import FaissLocalStore

pytest.importorskip("faiss")


def test_faiss_upsert_and_query() -> None:
    store = FaissLocalStore(dim=4)
    v1 = np.array([1.0, 0, 0, 0], dtype=np.float32)
    v2 = np.array([0.0, 1, 0, 0], dtype=np.float32)
    v3 = np.array([0.0, 0, 1, 0], dtype=np.float32)
    store.batch_upsert(["a", "b", "c"], np.stack([v1, v2, v3]), [{"c": 0}, {"c": 1}, {"c": 0}])

    hits = store.query(v1, k=2)
    assert hits[0].id == "a"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_faiss_centroid_filter() -> None:
    store = FaissLocalStore(dim=2)
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    store.batch_upsert(["a", "b"], np.stack([a, b]), [{"cls": 0}, {"cls": 1}])
    c0 = store.centroid({"cls": 0})
    assert np.allclose(c0, a, atol=1e-4)
    c1 = store.centroid({"cls": 1})
    assert np.allclose(c1, b, atol=1e-4)


def test_faiss_size() -> None:
    store = FaissLocalStore(dim=2)
    assert store.size() == 0
    store.upsert("a", np.array([1.0, 0.0], dtype=np.float32))
    assert store.size() == 1
