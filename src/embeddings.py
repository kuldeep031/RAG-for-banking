from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_embedding_spec
from src.ollama_client import OllamaClient


_MODEL_CACHE: dict[str, SentenceTransformer] = {}

def clear_cache() -> None:
    _MODEL_CACHE.clear()
    import gc
    gc.collect()

@dataclass
class EmbeddingModel:
    model_key: str

    def __post_init__(self) -> None:
        self.spec = get_embedding_spec(self.model_key)
        self.backend = str(self.spec.get("backend", "sentence_transformers"))
        self.model_name = str(self.spec.get("model_name", ""))
        self.query_prefix = str(self.spec.get("query_prefix", ""))
        self.passage_prefix = str(self.spec.get("passage_prefix", ""))

        if self.backend == "sentence_transformers":
            if self.model_name not in _MODEL_CACHE:
                # Force offline reuse after the first download so the project keeps working
                # without needing live network access during retrieval or evaluation.
                _MODEL_CACHE[self.model_name] = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                )
            self.model = _MODEL_CACHE[self.model_name]
        elif self.backend == "ollama":
            self.model = OllamaClient(self.model_name)
        else:
            raise ValueError(
                f"Unsupported embedding backend '{self.backend}' for {self.model_key}."
            )

    def _prepare_query(self, text: str) -> str:
        return f"{self.query_prefix}{text}" if self.query_prefix else text

    def _prepare_passage(self, text: str) -> str:
        return f"{self.passage_prefix}{text}" if self.passage_prefix else text

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (vectors / norms).astype("float32")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prepared = [self._prepare_query(text) for text in texts]
        if self.backend == "sentence_transformers":
            return self.model.encode(
                prepared,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")

        vectors = self.model.embed_many(prepared)
        return self._normalize(vectors)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        prepared = [self._prepare_passage(text) for text in texts]
        if self.backend == "sentence_transformers":
            return self.model.encode(
                prepared,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")

        vectors = self.model.embed_many(prepared)
        return self._normalize(vectors)
