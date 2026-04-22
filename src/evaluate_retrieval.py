from collections import defaultdict

import pandas as pd


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for chunk_id in retrieved[:k] if chunk_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    import math

    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        rel = 1.0 if chunk_id in relevant else 0.0
        dcg += rel / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg else 0.0


def summarize_retrieval_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    grouped = defaultdict(list)
    for _, row in results_df.iterrows():
        relevant = set(str(row["relevant_chunk_ids"]).split("|")) if row["relevant_chunk_ids"] else set()
        retrieved = str(row["retrieved_chunk_ids"]).split("|") if row["retrieved_chunk_ids"] else []
        grouped["embedding_model"].append(row["embedding_model"])
        grouped["precision_at_3"].append(precision_at_k(retrieved, relevant, 3))
        grouped["recall_at_3"].append(recall_at_k(retrieved, relevant, 3))
        grouped["recall_at_5"].append(recall_at_k(retrieved, relevant, 5))
        grouped["mrr"].append(reciprocal_rank(retrieved, relevant))
        grouped["ndcg_at_5"].append(ndcg_at_k(retrieved, relevant, 5))

    expanded = pd.DataFrame(grouped)
    return expanded.groupby("embedding_model", as_index=False).mean(numeric_only=True)
