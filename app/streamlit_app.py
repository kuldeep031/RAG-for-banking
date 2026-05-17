import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHUNKS_DIR, EMBEDDING_MODELS, MODEL_SPECS, RetrievalConfig, friendly_embedding_name
from src.ollama_client import (
    OllamaEmbeddingError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from src.rag_pipeline import SimpleBankingRiskRAG


RESULTS_DIR = PROJECT_ROOT / "results"
RETRIEVAL_SUMMARY_PATH = RESULTS_DIR / "retrieval" / "model_summary.csv"
ANSWER_SUMMARY_PATH = RESULTS_DIR / "answers" / "model_summary.csv"
QUESTION_OUTPUTS_PATH = RESULTS_DIR / "answers" / "experiment_outputs.csv"


def friendly_model_name(model_key: str) -> str:
    return friendly_embedding_name(model_key)


def get_indexed_embedding_keys() -> list[str]:
    keys: list[str] = []
    for model_key in EMBEDDING_MODELS:
        index_path = PROJECT_ROOT / "indexes" / model_key / "index.faiss"
        metadata_path = PROJECT_ROOT / "indexes" / model_key / "metadata.csv"
        if index_path.exists() and metadata_path.exists():
            keys.append(model_key)
    return keys


@st.cache_resource(show_spinner=False)
def get_pipeline(model_key: str) -> SimpleBankingRiskRAG:
    return SimpleBankingRiskRAG(RetrievalConfig(embedding_key=model_key))


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_metric(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_ollama_unavailable(error: Exception | None = None) -> None:
    st.error(
        "Ollama is not running, so live generation is unavailable. Start the local server and retry."
    )
    st.code("ollama serve", language="powershell")
    if error is not None:
        st.caption(str(error))


def render_embedding_backend_error(model_key: str, error: Exception) -> None:
    st.warning(
        f"{friendly_model_name(model_key)} could not run its live embedding/query path."
    )
    st.caption(str(error))


def render_model_missing(error: Exception) -> None:
    st.error("The local Ollama generation model required by this project is not installed.")
    st.code("ollama pull llama3.2", language="powershell")
    st.caption(str(error))


def build_evidence_table(rows: list[dict]) -> pd.DataFrame:
    evidence_df = pd.DataFrame(rows)
    if evidence_df.empty:
        return evidence_df

    desired_columns = [
        "chunk_id",
        "source_file",
        "page_no_start",
        "score",
        "chunk_text",
    ]
    available_columns = [column for column in desired_columns if column in evidence_df.columns]
    evidence_df = evidence_df[available_columns].copy()
    if "score" in evidence_df.columns:
        evidence_df["score"] = evidence_df["score"].map(lambda value: round(float(value), 4))
    return evidence_df


def render_pipeline_output(title: str, output) -> None:
    decision = output.decision
    left, right, extra = st.columns([1.1, 1.1, 1.4])

    with left:
        st.metric("Risk Label", decision.get("risk_label", "Unknown"))
        st.metric("Evidence Status", decision.get("evidence_status", "unknown"))

    with right:
        top_score = float(output.retrieved_rows[0]["score"]) if output.retrieved_rows else 0.0
        st.metric("Top Retrieval Score", format_metric(top_score, 3))
        st.metric("Total Latency (s)", format_metric(output.timings.get("total_seconds", 0.0), 2))

    with extra:
        st.caption(title)
        st.write(decision.get("answer", ""))
        justification = decision.get("justification", "")
        if justification:
            st.caption("Justification")
            st.write(justification)

    st.caption("Retrieved Evidence")
    evidence_df = build_evidence_table(output.retrieved_rows)
    if evidence_df.empty:
        st.warning("No evidence retrieved.")
    else:
        st.dataframe(
            evidence_df,
            use_container_width=True,
            column_config={
                "chunk_text": st.column_config.TextColumn(width="large"),
            },
            hide_index=True,
        )


def render_single_query_tab() -> None:
    st.subheader("Live Query")
    available_model_keys = get_indexed_embedding_keys()
    if not available_model_keys:
        st.warning("No built indexes found yet.")
        return

    model_key = st.selectbox(
        "Embedding Model",
        options=available_model_keys,
        format_func=friendly_model_name,
        key="single_model_key",
    )
    query = st.text_area(
        "Ask a banking risk question",
        placeholder="If the board does not review credit risk strategy and policies, what risk level is indicated?",
        key="single_query",
        height=120,
    )

    if st.button("Run Single Model", type="primary"):
        if not query.strip():
            st.warning("Enter a question first.")
            return

        try:
            with st.spinner("Running retrieval and grounded answer generation..."):
                output = get_pipeline(model_key).run(query.strip())
            render_pipeline_output(friendly_model_name(model_key), output)
        except FileNotFoundError:
            st.error("Indexes are not built yet. Build them before opening the demo.")
        except OllamaUnavailableError as exc:
            render_ollama_unavailable(exc)
        except OllamaModelNotFoundError as exc:
            render_model_missing(exc)
        except OllamaEmbeddingError as exc:
            render_embedding_backend_error(model_key, exc)
        except Exception as exc:
            st.exception(exc)


def render_compare_tab() -> None:
    st.subheader("Compare All Embeddings")
    available_model_keys = get_indexed_embedding_keys()
    if not available_model_keys:
        st.warning("No built indexes found yet.")
        return

    compare_query = st.text_area(
        "Comparison Question",
        placeholder="What does SR 11-7 say model validation should verify?",
        key="compare_query",
        height=120,
    )

    if st.button("Compare All Models"):
        if not compare_query.strip():
            st.warning("Enter a question first.")
            return

        try:
            comparison_rows: list[dict] = []
            outputs: dict[str, object] = {}
            failed_models: list[tuple[str, str]] = []

            with st.spinner("Running all available embedding pipelines..."):
                for model_key in available_model_keys:
                    try:
                        output = get_pipeline(model_key).run(compare_query.strip())
                        outputs[model_key] = output
                        comparison_rows.append(
                            {
                                "embedding_model": friendly_model_name(model_key),
                                "risk_label": output.decision.get("risk_label", "Unknown"),
                                "evidence_status": output.decision.get("evidence_status", "unknown"),
                                "top_chunk": output.retrieved_rows[0]["chunk_id"] if output.retrieved_rows else "",
                                "top_source": output.retrieved_rows[0]["source_file"] if output.retrieved_rows else "",
                                "top_score": round(float(output.retrieved_rows[0]["score"]), 4)
                                if output.retrieved_rows
                                else 0.0,
                                "total_seconds": round(float(output.timings.get("total_seconds", 0.0)), 2),
                            }
                        )
                    except (OllamaUnavailableError, OllamaEmbeddingError, OllamaModelNotFoundError) as exc:
                        failed_models.append((model_key, str(exc)))
                        comparison_rows.append(
                            {
                                "embedding_model": friendly_model_name(model_key),
                                "risk_label": "Unavailable",
                                "evidence_status": "error",
                                "top_chunk": "",
                                "top_source": "",
                                "top_score": 0.0,
                                "total_seconds": 0.0,
                            }
                        )

            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

            for model_key in available_model_keys:
                if model_key not in outputs:
                    continue
                with st.expander(friendly_model_name(model_key), expanded=False):
                    render_pipeline_output(friendly_model_name(model_key), outputs[model_key])
            for model_key, message in failed_models:
                with st.expander(friendly_model_name(model_key), expanded=False):
                    st.warning(message)
        except FileNotFoundError:
            st.error("Indexes are not built yet. Build them before opening the demo.")
        except OllamaUnavailableError as exc:
            render_ollama_unavailable(exc)
        except OllamaModelNotFoundError as exc:
            render_model_missing(exc)
        except Exception as exc:
            st.exception(exc)


def render_benchmark_tab() -> None:
    st.subheader("Saved Benchmark Results")
    retrieval_df = load_csv(RETRIEVAL_SUMMARY_PATH)
    answer_df = load_csv(ANSWER_SUMMARY_PATH)
    question_outputs_df = load_csv(QUESTION_OUTPUTS_PATH)

    if retrieval_df.empty and answer_df.empty:
        st.info("Run `python run_experiments.py` to populate benchmark summaries.")
        return

    if not retrieval_df.empty:
        retrieval_view = retrieval_df.copy()
        retrieval_view["embedding_model"] = retrieval_view["embedding_model"].map(friendly_model_name)
        st.caption("Retrieval Metrics")
        st.dataframe(retrieval_view, use_container_width=True, hide_index=True)

        retrieval_chart = retrieval_df.set_index("embedding_model")[
            ["precision_at_3", "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5"]
        ]
        st.bar_chart(retrieval_chart)

    if not answer_df.empty:
        answer_view = answer_df.copy()
        answer_view["embedding_model"] = answer_view["embedding_model"].map(friendly_model_name)
        st.caption("Answer / Decision Metrics")
        st.dataframe(answer_view, use_container_width=True, hide_index=True)

        answer_chart_cols = [
            col
            for col in [
                "label_accuracy",
                "citation_hit_rate",
                "avg_answer_similarity",
                "context_precision",
                "faithfulness",
            ]
            if col in answer_df.columns
        ]
        answer_chart = answer_df.set_index("embedding_model")[answer_chart_cols]
        st.bar_chart(answer_chart)

        latency_chart = answer_df.set_index("embedding_model")[["avg_total_seconds"]]
        st.caption("Average End-to-End Latency")
        st.bar_chart(latency_chart)

    if not question_outputs_df.empty:
        st.caption("Per-Question Outputs")
        display_df = question_outputs_df[
            [
                "question_id",
                "embedding_model",
                "question_type",
                "risk_label",
                "expected_risk_label",
                "evidence_status",
                "total_seconds",
            ]
        ].copy()
        display_df["embedding_model"] = display_df["embedding_model"].map(friendly_model_name)
        display_df["total_seconds"] = display_df["total_seconds"].map(lambda value: round(float(value), 2))
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_recommendations_tab() -> None:
    st.subheader("Cost-Performance Trade-offs & Recommendations")
    st.write(
        """
        **Objective 3 & 4 Analysis**: This section evaluates the trade-offs between performance, latency,
        and local resource cost. For this offline project, cost is represented through model parameters,
        approximate RAM footprint, and index size on disk rather than API billing.
        Finally, it recommends the most suitable embedding models for specific banking risk use cases.
        """
    )
    
    retrieval_df = load_csv(RETRIEVAL_SUMMARY_PATH)
    answer_df = load_csv(ANSWER_SUMMARY_PATH)

    if retrieval_df.empty or answer_df.empty:
        st.info("Run `python run_experiments.py` to populate benchmark summaries.")
        return

    # Merge dataset for unified analysis
    merged_df = pd.merge(retrieval_df, answer_df, on="embedding_model")

    if "params_millions" not in merged_df.columns:
        merged_df["params_millions"] = merged_df["embedding_model"].apply(
            lambda x: MODEL_SPECS.get(x, {}).get("params_millions", 0)
        )
    if "memory_mb_approx" not in merged_df.columns:
        merged_df["memory_mb_approx"] = merged_df["embedding_model"].apply(
            lambda x: MODEL_SPECS.get(x, {}).get("memory_mb_approx", 0)
        )

    st.markdown("### 1. Local resource cost vs Accuracy")
    display_df = merged_df[
        [
            "embedding_model",
            "params_millions",
            "memory_mb_approx",
            "index_disk_mb",
            "ndcg_at_5",
            "label_accuracy",
            "avg_total_seconds",
        ]
    ].copy()
    display_df["embedding_model"] = display_df["embedding_model"].map(friendly_model_name)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Accuracy (nDCG@5) Comparison")
        st.bar_chart(merged_df.set_index("embedding_model")["ndcg_at_5"])
    with col2:
        st.caption("Approximate RAM Footprint (MB)")
        st.bar_chart(merged_df.set_index("embedding_model")["memory_mb_approx"])

    # Finding optimal models for distinct use cases.
    best_retrieval_model = merged_df.loc[merged_df["ndcg_at_5"].idxmax()]
    best_decision_model = merged_df.loc[merged_df["label_accuracy"].idxmax()]
    lowest_cost_model = merged_df.loc[merged_df["memory_mb_approx"].idxmin()]
    fastest_model = merged_df.loc[merged_df["avg_total_seconds"].idxmin()]

    st.markdown("### 2. Recommendations for Banking Risk Use Cases")
    st.caption(
        "Live-query top scores are per-question retrieval similarities. Recommendations below are based on "
        "aggregate benchmark metrics across the full RBI evaluation set."
    )

    st.success(
        f"**Retrieval-Critical Use Case (e.g. Regulatory Audits, Evidence Traceability):**\n\n"
        f"**Recommended Model:** {friendly_model_name(best_retrieval_model['embedding_model'])}\n\n"
        f"**Reason:** Achieved the highest aggregate retrieval quality on the RBI benchmark "
        f"(nDCG@5: *{best_retrieval_model['ndcg_at_5']:.3f}*, MRR: *{best_retrieval_model['mrr']:.3f}*). "
        f"This is the safest choice when the primary requirement is retrieving the strongest supporting evidence."
    )

    st.success(
        f"**Decision-Accuracy Use Case (e.g. Risk Labeling / Automated Triage):**\n\n"
        f"**Recommended Model:** {friendly_model_name(best_decision_model['embedding_model'])}\n\n"
        f"**Reason:** Achieved the highest downstream label accuracy on the RBI benchmark "
        f"(*{best_decision_model['label_accuracy']:.2f}*) with the strongest citation hit rate "
        f"(*{best_decision_model['citation_hit_rate']:.2f}*). This is the best option when the final risk decision matters most."
    )

    st.info(
        f"**Resource-Constrained Local Deployment:**\n\n"
        f"**Recommended Model:** {friendly_model_name(lowest_cost_model['embedding_model'])}\n\n"
        f"**Reason:** Lowest approximate embedding RAM footprint at about *{lowest_cost_model['memory_mb_approx']}MB*, "
        f"while still preserving a usable retrieval baseline for offline deployments on modest hardware."
    )

    st.warning(
        f"**Latency-Sensitive Use Case (e.g. Real-time Customer Support):**\n\n"
        f"**Recommended Model:** {friendly_model_name(fastest_model['embedding_model'])}\n\n"
        f"**Reason:** Fastest end-to-end response time at *{fastest_model['avg_total_seconds']:.2f} seconds*, "
        f"which makes it attractive for interactive RAG scenarios."
    )

st.set_page_config(page_title="Banking Risk RAG", layout="wide")
st.title("Banking Risk RAG Demo")
st.caption(
    "Simple RAG for banking risk assessment with local embeddings, FAISS, Ollama, and evaluation summaries."
)

if not (CHUNKS_DIR / "chunks.csv").exists():
    st.info("Start by placing PDFs into data/raw_pdfs and running the ingestion pipeline.")

tab_live, tab_compare, tab_benchmark, tab_recommendations = st.tabs(
    ["Live Query", "Compare Embeddings", "Benchmark Summary", "Cost & Recommendations"]
)

with tab_live:
    render_single_query_tab()

with tab_compare:
    render_compare_tab()

with tab_benchmark:
    render_benchmark_tab()

with tab_recommendations:
    render_recommendations_tab()
