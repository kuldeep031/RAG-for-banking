from dataclasses import dataclass

import pandas as pd

from src.embeddings import EmbeddingModel
from src.faiss_store import FaissStore


@dataclass
class RetrievalResult:
    query: str
    model_key: str
    rows: list[dict]


class RetrieverAgent:
    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        self.embedding_model = EmbeddingModel(model_key)
        self.store = FaissStore(model_key)

    def build_index_from_chunks(self, chunks_df: pd.DataFrame) -> None:
        embeddings = self.embedding_model.encode_passages(
            chunks_df["chunk_text"].fillna("").tolist()
        )
        self.store.build(embeddings, chunks_df)
        self.store.save()

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        query_embedding = self.embedding_model.encode_queries([query])
        rows = self.store.search(query_embedding, top_k=top_k)
        return RetrievalResult(query=query, model_key=self.model_key, rows=rows)
