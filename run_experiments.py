import argparse
import json

import pandas as pd

from src.config import (
    DEFAULT_EXPERIMENT_EMBEDDING_KEYS,
    DEFAULT_OLLAMA_MODEL,
    EVAL_DIR,
    INDEX_DIR,
    MODEL_SPECS,
    RESULTS_DIR,
    RetrievalConfig,
)
from src.ollama_client import OllamaClient
from src.evaluate_answers import compute_answer_metrics
from src.evaluate_retrieval import summarize_retrieval_metrics
from src.rag_pipeline import SimpleBankingRiskRAG


def _resolve_embedding_keys(models_arg: str | None) -> list[str]:
    if not models_arg:
        return list(DEFAULT_EXPERIMENT_EMBEDDING_KEYS)
    return [item.strip() for item in models_arg.split(",") if item.strip()]


def _index_disk_mb(model_key: str) -> float:
    store_dir = INDEX_DIR / model_key
    total_bytes = 0
    for filename in ["index.faiss", "metadata.csv"]:
        path = store_dir / filename
        if path.exists():
            total_bytes += path.stat().st_size
    return round(total_bytes / (1024 * 1024), 3)


def _validate_required_indexes(model_keys: list[str]) -> None:
    missing: list[str] = []
    for model_key in model_keys:
        index_path = INDEX_DIR / model_key / "index.faiss"
        metadata_path = INDEX_DIR / model_key / "metadata.csv"
        if not index_path.exists() or not metadata_path.exists():
            missing.append(model_key)
    if missing:
        raise FileNotFoundError(
            "Missing FAISS indexes for: "
            + ", ".join(missing)
            + ". Build them first with `python -m src.build_indexes --models ...`."
        )


def run(
    model_keys: list[str] | None = None,
    run_ragas: bool = False,
    ragas_provider: str = "ollama",
    ragas_model: str = "llama3.2",
) -> str:
    selected_model_keys = model_keys or list(DEFAULT_EXPERIMENT_EMBEDDING_KEYS)
    _validate_required_indexes(selected_model_keys)
    OllamaClient(DEFAULT_OLLAMA_MODEL).ensure_model_available()

    questions_path = EVAL_DIR / "questions.csv"
    relevance_path = EVAL_DIR / "gold_relevance.csv"

    questions_df = pd.read_csv(questions_path)
    relevance_df = pd.read_csv(relevance_path)
    gold_map = (
        relevance_df.groupby("question_id")["relevant_chunk_id"]
        .apply(list)
        .to_dict()
    )

    answer_rows: list[dict] = []
    retrieval_rows: list[dict] = []

    from src.embeddings import clear_cache

    for embedding_key in selected_model_keys:
        pipeline = SimpleBankingRiskRAG(RetrievalConfig(embedding_key=embedding_key))
        for _, row in questions_df.iterrows():
            output = pipeline.run(str(row["question"]))
            retrieved_chunk_ids = [
                str(chunk.get("chunk_id"))
                for chunk in output.retrieved_rows
                if chunk.get("chunk_id")
            ]
            retrieved_contexts = [
                str(chunk.get("chunk_text", "")) for chunk in output.retrieved_rows
            ]
            relevant_chunk_ids = gold_map.get(row["question_id"], [])

            answer_rows.append(
                {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "question_type": row.get("question_type", ""),
                    "embedding_model": embedding_key,
                    "response": output.decision.get("answer", ""),
                    "risk_label": output.decision.get("risk_label", "Unknown"),
                    "expected_risk_label": row.get("expected_risk_label", ""),
                    "retrieved_chunk_ids": "|".join(retrieved_chunk_ids),
                    "relevant_chunk_ids": "|".join(relevant_chunk_ids),
                    "retrieved_contexts": json.dumps(retrieved_contexts, ensure_ascii=True),
                    "reference": row.get("reference_answer", ""),
                    "evidence_status": output.decision.get("evidence_status", ""),
                    "total_seconds": output.timings.get("total_seconds", 0.0),
                }
            )
            retrieval_rows.append(
                {
                    "question_id": row["question_id"],
                    "embedding_model": embedding_key,
                    "retrieved_chunk_ids": "|".join(retrieved_chunk_ids),
                    "relevant_chunk_ids": "|".join(relevant_chunk_ids),
                    "retrieval_seconds": output.timings.get("retrieval_seconds", 0.0),
                }
            )

        clear_cache()

    answer_output_path = RESULTS_DIR / "answers" / "experiment_outputs.csv"
    retrieval_output_path = RESULTS_DIR / "retrieval" / "per_question_retrieval.csv"
    retrieval_summary_path = RESULTS_DIR / "retrieval" / "model_summary.csv"
    answer_summary_path = RESULTS_DIR / "answers" / "model_summary.csv"
    ragas_input_path = RESULTS_DIR / "ragas" / "ragas_input.csv"

    answers_df = pd.DataFrame(answer_rows)
    retrieval_df = pd.DataFrame(retrieval_rows)

    answer_output_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_output_path.parent.mkdir(parents=True, exist_ok=True)
    ragas_input_path.parent.mkdir(parents=True, exist_ok=True)

    answers_df.to_csv(answer_output_path, index=False)
    retrieval_df.to_csv(retrieval_output_path, index=False)

    retrieval_summary_df = summarize_retrieval_metrics(retrieval_df)
    if not retrieval_df.empty:
        latency_df = (
            retrieval_df.groupby("embedding_model", as_index=False)["retrieval_seconds"]
            .mean()
            .rename(columns={"retrieval_seconds": "avg_retrieval_seconds"})
        )
        retrieval_summary_df = retrieval_summary_df.merge(
            latency_df,
            on="embedding_model",
            how="left",
        )
    retrieval_summary_df["params_millions"] = retrieval_summary_df["embedding_model"].map(
        lambda key: MODEL_SPECS.get(key, {}).get("params_millions", 0.0)
    )
    retrieval_summary_df["memory_mb_approx"] = retrieval_summary_df["embedding_model"].map(
        lambda key: MODEL_SPECS.get(key, {}).get("memory_mb_approx", 0.0)
    )
    retrieval_summary_df["index_disk_mb"] = retrieval_summary_df["embedding_model"].map(
        _index_disk_mb
    )
    retrieval_summary_df.to_csv(retrieval_summary_path, index=False)

    ragas_df = answers_df[
        ["question_id", "embedding_model", "question", "response", "retrieved_contexts", "reference"]
    ].copy()
    ragas_df.to_csv(ragas_input_path, index=False)

    answer_summary_df = compute_answer_metrics(answers_df)

    if run_ragas:
        from src.evaluate_ragas import run_ragas_evaluation

        print("Running RAGAS evaluation...")
        try:
            ragas_eval_df = ragas_df.copy()
            ragas_eval_df["retrieved_contexts"] = ragas_eval_df["retrieved_contexts"].apply(
                lambda value: json.loads(value) if isinstance(value, str) else value
            )
            ragas_scores_df = run_ragas_evaluation(
                df=ragas_eval_df,
                evaluator_model=ragas_model,
                provider=ragas_provider,
                metric_set="core",
            )
            ragas_scores_df["embedding_model"] = ragas_eval_df["embedding_model"].values
            ragas_scores_df.to_csv(RESULTS_DIR / "ragas" / "ragas_scores.csv", index=False)
            ragas_agg_df = (
                ragas_scores_df.groupby("embedding_model")[["context_precision", "faithfulness"]]
                .mean()
                .reset_index()
            )
            answer_summary_df = answer_summary_df.merge(
                ragas_agg_df,
                on="embedding_model",
                how="left",
            )
            print("RAGAS evaluation completed successfully.")
        except Exception as exc:
            print(f"RAGAS evaluation failed: {exc}")

    answer_summary_df.to_csv(answer_summary_path, index=False)
    return str(answer_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run local embedding experiments for the banking-risk RAG study."
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_EXPERIMENT_EMBEDDING_KEYS),
        help="Comma-separated embedding keys to evaluate.",
    )
    parser.add_argument(
        "--run-ragas",
        action="store_true",
        help="Opt in to RAGAS evaluation. Keep this disabled for a fully local-only run.",
    )
    parser.add_argument(
        "--ragas-provider",
        choices=["ollama", "grok", "google"],
        default="ollama",
        help="Provider to use if --run-ragas is enabled.",
    )
    parser.add_argument(
        "--ragas-model",
        default="llama3.2",
        help="Model to use if --run-ragas is enabled.",
    )
    args = parser.parse_args()

    print(
        run(
            model_keys=_resolve_embedding_keys(args.models),
            run_ragas=args.run_ragas,
            ragas_provider=args.ragas_provider,
            ragas_model=args.ragas_model,
        )
    )
