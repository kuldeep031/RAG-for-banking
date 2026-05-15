from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import RESULTS_DIR, MODEL_SPECS, friendly_embedding_name


RETRIEVAL_SUMMARY_PATH = RESULTS_DIR / "retrieval" / "model_summary.csv"
ANSWER_SUMMARY_PATH = RESULTS_DIR / "answers" / "model_summary.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

def _friendly_labels(series: pd.Series) -> pd.Series:
    return series.map(friendly_embedding_name)


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_retrieval_plot(retrieval_df: pd.DataFrame) -> Path:
    melted = retrieval_df.melt(
        id_vars=["embedding_model"],
        value_vars=["precision_at_3", "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5"],
        var_name="metric",
        value_name="score",
    )
    melted["embedding_model"] = _friendly_labels(melted["embedding_model"])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="metric", y="score", hue="embedding_model", ax=ax)
    ax.set_title("Retrieval Metrics by Embedding Model")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Embedding Model")

    output_path = PLOTS_DIR / "retrieval_metrics.png"
    _save_plot(fig, output_path)
    return output_path


def generate_answer_plot(answer_df: pd.DataFrame) -> Path:
    metrics_to_plot = ["label_accuracy", "citation_hit_rate", "avg_answer_similarity"]
    if "context_precision" in answer_df.columns and "faithfulness" in answer_df.columns:
        metrics_to_plot.extend(["context_precision", "faithfulness"])

    melted = answer_df.melt(
        id_vars=["embedding_model"],
        value_vars=metrics_to_plot,
        var_name="metric",
        value_name="score",
    )
    melted["embedding_model"] = _friendly_labels(melted["embedding_model"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=melted, x="metric", y="score", hue="embedding_model", ax=ax)
    ax.set_title("Answer Quality Metrics by Embedding Model")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Embedding Model")

    output_path = PLOTS_DIR / "answer_metrics.png"
    _save_plot(fig, output_path)
    return output_path


def generate_latency_plot(retrieval_df: pd.DataFrame, answer_df: pd.DataFrame) -> Path:
    merged = retrieval_df[["embedding_model", "avg_retrieval_seconds"]].merge(
        answer_df[["embedding_model", "avg_total_seconds"]],
        on="embedding_model",
        how="inner",
    )
    merged["embedding_model"] = _friendly_labels(merged["embedding_model"])
    melted = merged.melt(
        id_vars=["embedding_model"],
        value_vars=["avg_retrieval_seconds", "avg_total_seconds"],
        var_name="metric",
        value_name="seconds",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=melted, x="embedding_model", y="seconds", hue="metric", ax=ax)
    ax.set_title("Latency Comparison by Embedding Model")
    ax.set_xlabel("Embedding Model")
    ax.set_ylabel("Seconds")
    ax.legend(title="Latency Metric")

    output_path = PLOTS_DIR / "latency_comparison.png"
    _save_plot(fig, output_path)
    return output_path

def generate_cost_accuracy_plot(retrieval_df: pd.DataFrame) -> Path:
    df = retrieval_df.copy()
    if "memory_mb_approx" not in df.columns:
        df["memory_mb_approx"] = df["embedding_model"].apply(
            lambda x: MODEL_SPECS.get(x, {}).get("memory_mb_approx", 0)
        )
    df["embedding_model"] = _friendly_labels(df["embedding_model"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="memory_mb_approx",
        y="ndcg_at_5",
        hue="embedding_model",
        s=400,
        ax=ax
    )
    for _, row in df.iterrows():
        ax.text(row["memory_mb_approx"] + 5, row["ndcg_at_5"], row["embedding_model"], fontsize=9)

    ax.set_title("Local Resource Cost (RAM Proxy) vs Accuracy")
    ax.set_xlabel("Approximate Embedding RAM Footprint (MB)")
    ax.set_ylabel("Retrieval Accuracy (nDCG@5)")
    ax.legend(title="Embedding Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    output_path = PLOTS_DIR / "cost_vs_accuracy.png"
    _save_plot(fig, output_path)
    return output_path


def generate_all_plots() -> list[Path]:
    if not RETRIEVAL_SUMMARY_PATH.exists() or not ANSWER_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Missing summary CSV files. Run `python run_experiments.py` first."
        )

    sns.set_theme(style="whitegrid")
    retrieval_df = pd.read_csv(RETRIEVAL_SUMMARY_PATH)
    answer_df = pd.read_csv(ANSWER_SUMMARY_PATH)

    output_paths = [
        generate_retrieval_plot(retrieval_df),
        generate_answer_plot(answer_df),
        generate_latency_plot(retrieval_df, answer_df),
        generate_cost_accuracy_plot(retrieval_df),
    ]
    return output_paths


if __name__ == "__main__":
    paths = generate_all_plots()
    for path in paths:
        print(path)
