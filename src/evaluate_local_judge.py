import json
from typing import Any

import pandas as pd
import requests

from src.ollama_client import OllamaClient


PROMPT_VARIANTS = [
    {"max_contexts": 2, "context_chars": 500, "answer_chars": 700, "reference_chars": 500},
    {"max_contexts": 1, "context_chars": 350, "answer_chars": 500, "reference_chars": 350},
    {"max_contexts": 1, "context_chars": 220, "answer_chars": 320, "reference_chars": 220},
]


def _extract_json_object(text: str) -> str | None:
    import re

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def _coerce_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _normalize_contexts(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return []


def _truncate(text: str, limit: int = 1200) -> str:
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def build_local_judge_prompt(
    row: pd.Series,
    max_contexts: int = 2,
    context_chars: int = 500,
    answer_chars: int = 700,
    reference_chars: int = 500,
) -> str:
    contexts = _normalize_contexts(row.get("retrieved_contexts", []))[:max_contexts]
    evidence_block = "\n---\n".join(
        f"Evidence {index + 1}:\n{_truncate(context, context_chars)}"
        for index, context in enumerate(contexts)
    )
    expected_risk_label = str(row.get("expected_risk_label", "")).strip() or "Unknown"

    return (
        "You are evaluating a banking-risk RAG answer.\n"
        "Score strictly from 0.0 to 1.0.\n"
        "Return only valid JSON with exactly these keys:\n"
        "groundedness_score, answer_relevance_score, decision_quality_score, note.\n"
        "groundedness_score: How well the answer is supported by the retrieved evidence.\n"
        "answer_relevance_score: How directly the answer addresses the user question.\n"
        "decision_quality_score: How appropriate the risk decision is for the question.\n"
        "For informational questions, a risk label of Unknown is often the correct decision behavior.\n"
        "If the answer overstates certainty or invents unsupported claims, reduce groundedness.\n\n"
        f"Question:\n{row.get('question', '')}\n\n"
        f"Retrieved evidence:\n{evidence_block}\n\n"
        f"Model answer:\n{_truncate(str(row.get('response', '')), answer_chars)}\n\n"
        f"Model risk label:\n{row.get('risk_label', 'Unknown')}\n\n"
        f"Reference answer:\n{_truncate(str(row.get('reference', '')), reference_chars)}\n\n"
        f"Expected risk label:\n{expected_risk_label}\n"
    )


def run_local_judge_evaluation(
    df: pd.DataFrame,
    evaluator_model: str = "llama3.2",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    client = OllamaClient(evaluator_model)
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        response = None
        last_error: Exception | None = None
        for variant in PROMPT_VARIANTS:
            prompt = build_local_judge_prompt(row, **variant)
            try:
                response = client.generate(
                    prompt,
                    temperature=0.0,
                    options={"num_ctx": 3072, "num_predict": 120},
                )
                break
            except requests.HTTPError as exc:
                last_error = exc
                continue

        if response is None:
            rows.append(
                {
                    "question_id": row.get("question_id", ""),
                    "embedding_model": row.get("embedding_model", ""),
                    "groundedness_score": 0.0,
                    "answer_relevance_score": 0.0,
                    "decision_quality_score": 0.0,
                    "judge_note": f"Local judge failed after prompt backoff: {last_error}",
                }
            )
            continue

        raw_text = response.text.strip()
        json_text = _extract_json_object(raw_text) or raw_text

        parsed: dict[str, Any]
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            parsed = {}

        rows.append(
            {
                "question_id": row.get("question_id", ""),
                "embedding_model": row.get("embedding_model", ""),
                "groundedness_score": _coerce_score(parsed.get("groundedness_score")),
                "answer_relevance_score": _coerce_score(parsed.get("answer_relevance_score")),
                "decision_quality_score": _coerce_score(parsed.get("decision_quality_score")),
                "judge_note": str(parsed.get("note", "")).strip(),
            }
        )

    return pd.DataFrame(rows)
