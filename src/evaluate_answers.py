from dataclasses import dataclass

import pandas as pd

from src.embeddings import EmbeddingModel


def _split_pipe_ids(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [item for item in value.split("|") if item]


def _citation_hit_rate(retrieved_ids: list[str], relevant_ids: list[str]) -> int:
    return int(bool(set(retrieved_ids) & set(relevant_ids)))


@dataclass
class AnswerMetricComputer:
    similarity_model_key: str = "all_minilm_l6_v2"

    def __post_init__(self) -> None:
        self.embedding_model = EmbeddingModel(self.similarity_model_key)

    def semantic_similarity(self, answers: list[str], references: list[str]) -> list[float]:
        answer_embeddings = self.embedding_model.encode_passages(answers)
        reference_embeddings = self.embedding_model.encode_passages(references)
        similarities = (answer_embeddings * reference_embeddings).sum(axis=1)
        return [float(value) for value in similarities]


def compute_answer_metrics(answer_df: pd.DataFrame) -> pd.DataFrame:
    if answer_df.empty:
        return pd.DataFrame()

    working_df = answer_df.copy()
    working_df["label_match"] = (
        working_df["risk_label"].fillna("")
        == working_df["expected_risk_label"].fillna("")
    ).astype(int)
    working_df["evidence_sufficient"] = (
        working_df["evidence_status"].fillna("").str.lower() == "sufficient"
    ).astype(int)
    working_df["retrieved_id_list"] = working_df["retrieved_chunk_ids"].apply(_split_pipe_ids)
    working_df["relevant_id_list"] = working_df["relevant_chunk_ids"].apply(_split_pipe_ids)
    working_df["citation_hit"] = [
        _citation_hit_rate(retrieved_ids, relevant_ids)
        for retrieved_ids, relevant_ids in zip(
            working_df["retrieved_id_list"],
            working_df["relevant_id_list"],
        )
    ]

    metric_computer = AnswerMetricComputer()
    working_df["answer_similarity"] = metric_computer.semantic_similarity(
        working_df["response"].fillna("").astype(str).tolist(),
        working_df["reference"].fillna("").astype(str).tolist(),
    )

    summary_df = (
        working_df.groupby("embedding_model", as_index=False)
        .agg(
            label_accuracy=("label_match", "mean"),
            sufficient_evidence_rate=("evidence_sufficient", "mean"),
            citation_hit_rate=("citation_hit", "mean"),
            avg_answer_similarity=("answer_similarity", "mean"),
            avg_total_seconds=("total_seconds", "mean"),
        )
    )
    return summary_df
