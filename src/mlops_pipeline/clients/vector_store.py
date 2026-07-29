"""Vector store client: interface + FAISS local + HTTP P4 stub.

Used for two things:
1. Retrieval-style storage of embeddings.
2. Rolling window of production input embeddings for drift monitoring.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np


@dataclass
class Hit:
    id: str
    score: float
    metadata: dict[str, Any]


class VectorStoreClient(ABC):
    @abstractmethod
    def upsert(self, id: str, embedding: np.ndarray, metadata: dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    def batch_upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None: ...

    @abstractmethod
    def query(self, embedding: np.ndarray, k: int = 10) -> list[Hit]: ...

    @abstractmethod
    def centroid(self, filter: dict[str, Any] | None = None) -> np.ndarray: ...

    @abstractmethod
    def all_embeddings(self, filter: dict[str, Any] | None = None) -> np.ndarray: ...

    @abstractmethod
    def size(self) -> int: ...


class FaissLocalStore(VectorStoreClient):
    """In-process FAISS-backed vector store.

    Uses IndexFlatIP (inner product) so cosine similarity works with
    L2-normalized vectors — normalize before upsert.
    """

    def __init__(self, dim: int, persist_dir: Path | None = None) -> None:
        import faiss
        self.dim = dim
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._persist_dir = persist_dir

    def upsert(self, id: str, embedding: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        self.batch_upsert([id], embedding.reshape(1, -1), [metadata or {}])

    def batch_upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        arr = np.ascontiguousarray(embeddings, dtype=np.float32)
        assert arr.shape[1] == self.dim
        self._index.add(arr)
        self._ids.extend(ids)
        if metadatas is None:
            metadatas = [{} for _ in ids]
        self._metas.extend(metadatas)

    def query(self, embedding: np.ndarray, k: int = 10) -> list[Hit]:
        q = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
        scores, idxs = self._index.search(q, k)
        out: list[Hit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append(Hit(id=self._ids[idx], score=float(score), metadata=self._metas[idx]))
        return out

    def centroid(self, filter: dict[str, Any] | None = None) -> np.ndarray:
        embs = self.all_embeddings(filter)
        if len(embs) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        c = embs.mean(axis=0)
        norm = np.linalg.norm(c)
        return (c / norm).astype(np.float32) if norm > 0 else c.astype(np.float32)

    def all_embeddings(self, filter: dict[str, Any] | None = None) -> np.ndarray:
        n = self._index.ntotal
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        all_embs = self._index.reconstruct_n(0, n)
        if not filter:
            return all_embs
        keep = [
            i for i, m in enumerate(self._metas)
            if all(m.get(k) == v for k, v in filter.items())
        ]
        return all_embs[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)

    def size(self) -> int:
        return int(self._index.ntotal)


class HttpP4Client(VectorStoreClient):
    """Stub HTTP client for the from-scratch P4 vector database.

    Assumes P4 exposes:
      POST /vectors        {id, embedding, metadata}
      POST /vectors/batch  {ids, embeddings, metadatas}
      POST /query          {embedding, k} -> {hits: [...]}
    """

    def __init__(self, base_url: str, dim: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.dim = dim
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def upsert(self, id: str, embedding: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        r = self._client.post(
            "/vectors",
            json={"id": id, "embedding": embedding.tolist(), "metadata": metadata or {}},
        )
        r.raise_for_status()

    def batch_upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        r = self._client.post(
            "/vectors/batch",
            json={
                "ids": ids,
                "embeddings": embeddings.tolist(),
                "metadatas": metadatas or [{} for _ in ids],
            },
        )
        r.raise_for_status()

    def query(self, embedding: np.ndarray, k: int = 10) -> list[Hit]:
        r = self._client.post("/query", json={"embedding": embedding.tolist(), "k": k})
        r.raise_for_status()
        return [Hit(**h) for h in r.json()["hits"]]

    def centroid(self, filter: dict[str, Any] | None = None) -> np.ndarray:
        r = self._client.post("/centroid", json={"filter": filter or {}})
        r.raise_for_status()
        return np.asarray(r.json()["centroid"], dtype=np.float32)

    def all_embeddings(self, filter: dict[str, Any] | None = None) -> np.ndarray:
        r = self._client.post("/embeddings/all", json={"filter": filter or {}})
        r.raise_for_status()
        return np.asarray(r.json()["embeddings"], dtype=np.float32)

    def size(self) -> int:
        r = self._client.get("/size")
        r.raise_for_status()
        return int(r.json()["size"])
