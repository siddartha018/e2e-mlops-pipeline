"""Sentence-transformer encoder wrapper.

Used for:
- Populating the vector store with training-set embeddings (drift baseline)
- Embedding production inputs at prediction time
"""
from __future__ import annotations

from typing import Any

import numpy as np


class Encoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None

    def _lazy_load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        model = self._lazy_load()
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vecs.astype(np.float32))

    @property
    def dim(self) -> int:
        model = self._lazy_load()
        return int(model.get_sentence_embedding_dimension())
