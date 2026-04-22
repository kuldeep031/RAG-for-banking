from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.config import INDEX_DIR


class FaissStore:
    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        self.store_dir = INDEX_DIR / model_key
        self.index_path = self.store_dir / "index.faiss"
        self.metadata_path = self.store_dir / "metadata.csv"
        self.index: faiss.Index | None = None
        self.metadata: pd.DataFrame | None = None

    def build(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        self.index = index
        self.metadata = metadata.reset_index(drop=True)

    def save(self) -> None:
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index and metadata must be built before saving.")
        faiss.write_index(self.index, str(self.index_path))
        self.metadata.to_csv(self.metadata_path, index=False)

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_csv(self.metadata_path)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.index is None or self.metadata is None:
            self.load()

        assert self.index is not None
        assert self.metadata is not None

        scores, indices = self.index.search(query_embedding, top_k)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.metadata.iloc[int(idx)].to_dict()
            row["score"] = float(score)
            results.append(row)
        return results
