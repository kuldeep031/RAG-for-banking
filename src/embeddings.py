from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODELS


_MODEL_CACHE: dict[str, SentenceTransformer] = {}

def clear_cache() -> None:
    _MODEL_CACHE.clear()
    import gc
    gc.collect()

@dataclass
class EmbeddingModel:
    model_key: str

    def __post_init__(self) -> None:
        if self.model_key not in EMBEDDING_MODELS:
            raise KeyError(f"Unknown embedding model key: {self.model_key}")
        self.model_name = EMBEDDING_MODELS[self.model_key]
        if self.model_name not in _MODEL_CACHE:
            # Force offline reuse after the first download so the project keeps working
            # without needing live network access during retrieval or evaluation.
            _MODEL_CACHE[self.model_name] = SentenceTransformer(
                self.model_name,
                local_files_only=True,
            )
        self.model = _MODEL_CACHE[self.model_name]

    def _prepare_query(self, text: str) -> str:
        if self.model_key == "e5_small_v2":
            return f"query: {text}"
        return text

    def _prepare_passage(self, text: str) -> str:
        if self.model_key == "e5_small_v2":
            return f"passage: {text}"
        return text

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prepared = [self._prepare_query(text) for text in texts]
        return self.model.encode(
            prepared,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        prepared = [self._prepare_passage(text) for text in texts]
        return self.model.encode(
            prepared,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
