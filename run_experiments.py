import json

import pandas as pd

from src.config import EVAL_DIR, RESULTS_DIR, RetrievalConfig
from src.evaluate_answers import compute_answer_metrics
from src.evaluate_retrieval import summarize_retrieval_metrics
from src.rag_pipeline import SimpleBankingRiskRAG


def run() -> str:
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
    for embedding_key in ["all_minilm_l6_v2", "e5_small_v2", "bge_small_en_v1_5"]:
        pipeline = SimpleBankingRiskRAG(RetrievalConfig(embedding_key=embedding_key))
        for _, row in questions_df.iterrows():
            output = pipeline.run(str(row["question"]))
            retrieved_chunk_ids = [
                str(chunk.get("chunk_id")) for chunk in output.retrieved_rows if chunk.get("chunk_id")
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
        # Clear out the model footprint from RAM
        clear_cache()

    answer_output_path = RESULTS_DIR / "answers" / "experiment_outputs.csv"
    retrieval_output_path = RESULTS_DIR / "retrieval" / "per_question_retrieval.csv"
    ragas_input_path = RESULTS_DIR / "ragas" / "ragas_input.csv"
    retrieval_summary_path = RESULTS_DIR / "retrieval" / "model_summary.csv"
    answer_summary_path = RESULTS_DIR / "answers" / "model_summary.csv"

    answers_df = pd.DataFrame(answer_rows)
    retrieval_df = pd.DataFrame(retrieval_rows)

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
    retrieval_summary_df.to_csv(retrieval_summary_path, index=False)

    ragas_df = answers_df[
        ["question_id", "embedding_model", "question", "response", "retrieved_contexts", "reference"]
    ].copy()
    ragas_df.to_csv(ragas_input_path, index=False)

    answer_summary_df = compute_answer_metrics(answers_df)

    from src.evaluate_ragas import run_ragas_evaluation

    print("Running RAGAS Evaluation... This may take a while since it calls the LLM Judge.")
    try:
        # Ragas requires actual Python lists, not JSON strings
        ragas_df["retrieved_contexts"] = ragas_df["retrieved_contexts"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        
        ragas_scores_df = run_ragas_evaluation(
            df=ragas_df, 
            evaluator_model="grok-beta", 
            provider="grok", 
            metric_set="core"
        )
        
        # Add the embedding model explicitly for aggregation
        ragas_scores_df["embedding_model"] = ragas_df["embedding_model"].values
        ragas_scores_df.to_csv(RESULTS_DIR / "ragas" / "ragas_scores.csv", index=False)
        
        ragas_agg_df = ragas_scores_df.groupby("embedding_model")[["context_precision", "faithfulness"]].mean().reset_index()
        answer_summary_df = answer_summary_df.merge(ragas_agg_df, on="embedding_model", how="left")
        
        print("RAGAS Evaluation completed successfully.")
    except Exception as e:
        print(f"RAGAS evaluation failed (perhaps API key missing?): {e}")

    answer_summary_df.to_csv(answer_summary_path, index=False)
    return str(answer_output_path)


if __name__ == "__main__":
    print(run())
