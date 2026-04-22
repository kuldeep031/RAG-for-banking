import json
import os
import argparse
from typing import Literal

import pandas as pd

from src.config import RESULTS_DIR


def run_ragas_evaluation(
    df: pd.DataFrame,
    evaluator_model: str = "grok-beta",
    evaluator_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    provider: str = "grok",
    metric_set: Literal["core", "full"] = "core",
    max_workers: int = 1,
    metric_names: list[str] | None = None,
) -> pd.DataFrame:
    try:
        from datasets import Dataset
        from dotenv import load_dotenv
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.llms import LangchainLLMWrapper, llm_factory
        from ragas.metrics._answer_relevance import answer_relevancy
        from ragas.metrics._context_precision import context_precision
        from ragas.metrics._context_recall import context_recall
        from ragas.metrics._faithfulness import faithfulness
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "RAGAS dependencies are not installed. Install requirements.txt first."
        ) from exc

    required_columns_legacy = {"question", "answer", "contexts", "ground_truth"}
    required_columns_new = {"user_input", "response", "retrieved_contexts", "reference"}
    
    # Adapt to whatever RAGAS version is installed
    if "question" in df.columns and "user_input" not in df.columns:
        df["user_input"] = df["question"]
    if "response" not in df.columns and "answer" in df.columns:
        df["response"] = df["answer"]
        
    missing = required_columns_new - set(df.columns)
    if missing:
        # Check if legacy columns are present instead
        legacy_missing = {"question", "response", "retrieved_contexts", "reference"} - set(df.columns)
        if legacy_missing:
           raise ValueError(f"Missing required RAGAS columns: {sorted(missing)}")
        required_columns = {"question", "response", "retrieved_contexts", "reference"}
    else:
        required_columns = required_columns_new

    load_dotenv()

    if provider == "google":
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            raise EnvironmentError(
                "Set GOOGLE_API_KEY before running Gemini-based RAGAS evaluation."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for Gemini evaluation through the OpenAI-compatible Gemini endpoint."
            ) from exc

        google_client = OpenAI(
            api_key=google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        evaluator_llm = llm_factory(
            evaluator_model,
            provider="openai",
            client=google_client,
            adapter="instructor",
        )
    elif provider == "grok":
        grok_api_key = os.environ.get("GROK_API_KEY")
        if not grok_api_key:
            raise EnvironmentError(
                "Set GROK_API_KEY before running Grok-based RAGAS evaluation."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for Grok evaluation through the OpenAI-compatible Grok endpoint."
            ) from exc

        grok_client = OpenAI(
            api_key=grok_api_key,
            base_url="https://api.x.ai/v1",
        )
        evaluator_llm = llm_factory(
            evaluator_model,
            provider="openai",
            client=grok_client,
            adapter="instructor",
        )
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        evaluator_llm = LangchainLLMWrapper(
            ChatOllama(
                model=evaluator_model,
                temperature=0,
            )
        )
    else:
        raise ValueError("provider must be either 'grok', 'google', or 'ollama'")

    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name=evaluator_embedding_model,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    metric_registry = {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
    }
    selected_metrics = [context_precision, faithfulness]
    if metric_set == "full":
        selected_metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    if metric_names:
        invalid = [name for name in metric_names if name not in metric_registry]
        if invalid:
            raise ValueError(
                f"Unknown metric names: {invalid}. Valid metrics: {sorted(metric_registry)}"
            )
        selected_metrics = [metric_registry[name] for name in metric_names]

    dataset = Dataset.from_pandas(df[list(required_columns)])
    result = evaluate(
        dataset,
        metrics=selected_metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=max_workers, max_retries=3, max_wait=30),
        raise_exceptions=False,
    )
    return result.to_pandas()


def load_ragas_input(path: str | None = None) -> pd.DataFrame:
    source = path or str(RESULTS_DIR / "ragas" / "ragas_input.csv")
    df = pd.read_csv(source)
    if "retrieved_contexts" in df.columns:
        df["retrieved_contexts"] = df["retrieved_contexts"].apply(
            lambda value: json.loads(value) if isinstance(value, str) else value
        )
    if "question" in df.columns and "user_input" not in df.columns:
        df["user_input"] = df["question"]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on saved experiment outputs.")
    parser.add_argument("--provider", choices=["grok", "google", "ollama"], default="grok")
    parser.add_argument("--model", default="grok-beta")
    parser.add_argument("--input", default=str(RESULTS_DIR / "ragas" / "ragas_input.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--metric-set", choices=["core", "full"], default="core")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--metrics",
        default="",
        help="Optional comma-separated metric names, e.g. context_precision,faithfulness",
    )
    args = parser.parse_args()

    input_df = load_ragas_input(args.input)
    if args.limit > 0:
        input_df = input_df.head(args.limit)

    scores = run_ragas_evaluation(
        input_df,
        evaluator_model=args.model,
        provider=args.provider,
        metric_set=args.metric_set,
        max_workers=args.max_workers,
        metric_names=[item.strip() for item in args.metrics.split(",") if item.strip()],
    )
    output_path = RESULTS_DIR / "ragas" / "ragas_scores.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    print(f"Saved RAGAS scores to {output_path}")
