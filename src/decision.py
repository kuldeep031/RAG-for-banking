import json
import re

import requests

from src.ollama_client import OllamaClient, OllamaModelNotFoundError, OllamaUnavailableError


ALLOWED_RISK_LABELS = {"Low", "Medium", "High", "Unknown"}
ALLOWED_EVIDENCE_STATUS = {"sufficient", "insufficient", "unparsed"}
DEFAULT_PROMPT_VARIANTS = [
    {"max_chunks": 3, "max_chunk_chars": 900},
    {"max_chunks": 2, "max_chunk_chars": 650},
    {"max_chunks": 1, "max_chunk_chars": 500},
]


def _is_risk_label_query(query: str) -> bool:
    lowered = query.lower()
    keywords = [
        "risk level",
        "risk label",
        "what risk",
        "what model risk",
        "what credit risk",
        "what operational risk",
        "is indicated",
        "does this indicate",
        "indicate?",
    ]
    return any(keyword in lowered for keyword in keywords)


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def build_decision_prompt(
    query: str,
    retrieved_rows: list[dict],
    max_chunks: int = 3,
    max_chunk_chars: int = 900,
) -> str:
    evidence_blocks = []
    for row in retrieved_rows[:max_chunks]:
        evidence_blocks.append(
            f"Chunk ID: {row.get('chunk_id')}\n"
            f"Source: {row.get('source_file')} page {row.get('page_no_start')}\n"
            f"Text: {_truncate_text(str(row.get('chunk_text', '')), max_chunk_chars)}\n"
        )

    evidence_text = "\n---\n".join(evidence_blocks)
    question_mode = (
        "risk_classification"
        if _is_risk_label_query(query)
        else "grounded_fact_answering"
    )
    return (
        "You are a single banking risk assessment RAG assistant.\n"
        "Use only the provided evidence.\n"
        "If the evidence is insufficient, say so clearly.\n"
        "Return only valid JSON and no extra text.\n"
        "Use exactly these keys: answer, risk_label, justification, evidence_chunk_ids, evidence_status.\n"
        "Allowed risk_label values are Low, Medium, High, or Unknown.\n"
        "Allowed evidence_status values are sufficient or insufficient.\n"
        "The answer must be a short grounded paragraph, not just the label.\n"
        "The evidence_chunk_ids value must be a JSON array of chunk ids.\n"
        "If evidence is weak, set risk_label to Unknown and evidence_status to insufficient.\n"
        "If the question is informational or explanatory rather than asking for a risk level,"
        " keep risk_label as Unknown even when the answer is supported.\n"
        "Never guess a High, Medium, or Low label unless the question clearly asks for a risk classification.\n"
        f"Question mode: {question_mode}\n"
        'Example output: {"answer":"...","risk_label":"Medium","justification":"...","evidence_chunk_ids":["DOC001_CH001"],"evidence_status":"sufficient"}\n\n'
        f"Question:\n{query}\n\n"
        f"Evidence:\n{evidence_text}\n"
    )


def _extract_json_object(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def _extract_string_field(text: str, field_name: str) -> str:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""
    value = match.group(1)
    value = value.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ")
    return value.strip()


def _extract_list_field(text: str, field_name: str) -> list[str]:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*\[(.*?)\]'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return []
    body = match.group(1)
    items = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
    cleaned: list[str] = []
    for item in items:
        cleaned.append(item.replace('\\"', '"').strip())
    return cleaned


def _recover_partial_decision(text: str, retrieved_rows: list[dict], query: str) -> dict | None:
    answer = _extract_string_field(text, "answer")
    if not answer:
        return None

    parsed = {
        "answer": answer,
        "risk_label": _extract_string_field(text, "risk_label") or "Unknown",
        "justification": _extract_string_field(text, "justification"),
        "evidence_chunk_ids": _extract_list_field(text, "evidence_chunk_ids"),
        "evidence_status": _extract_string_field(text, "evidence_status") or "unparsed",
    }
    recovered = _sanitize_decision(parsed, retrieved_rows, query)
    recovered["evidence_status"] = "unparsed"
    return recovered


def _sanitize_decision(parsed: dict, retrieved_rows: list[dict], query: str) -> dict:
    valid_chunk_ids = {
        str(row.get("chunk_id"))
        for row in retrieved_rows
        if row.get("chunk_id") is not None
    }
    evidence_chunk_ids = parsed.get("evidence_chunk_ids", [])
    if not isinstance(evidence_chunk_ids, list):
        evidence_chunk_ids = []
    evidence_chunk_ids = [
        str(chunk_id)
        for chunk_id in evidence_chunk_ids
        if str(chunk_id) in valid_chunk_ids
    ]

    risk_label = str(parsed.get("risk_label", "Unknown")).strip().title()
    if risk_label not in ALLOWED_RISK_LABELS:
        risk_label = "Unknown"

    evidence_status = str(parsed.get("evidence_status", "insufficient")).strip().lower()
    if evidence_status not in ALLOWED_EVIDENCE_STATUS:
        evidence_status = "insufficient"

    if not _is_risk_label_query(query):
        risk_label = "Unknown"

    if not evidence_chunk_ids and retrieved_rows:
        evidence_chunk_ids = [str(row.get("chunk_id")) for row in retrieved_rows[:2]]

    return {
        "answer": str(parsed.get("answer", "")).strip(),
        "risk_label": risk_label,
        "justification": str(parsed.get("justification", "")).strip(),
        "evidence_chunk_ids": evidence_chunk_ids,
        "evidence_status": evidence_status,
    }


class DecisionAgent:
    def __init__(self, model_name: str) -> None:
        self.client = OllamaClient(model_name)

    def decide(self, query: str, retrieved_rows: list[dict]) -> dict:
        if not retrieved_rows:
            return {
                "answer": "I could not find enough evidence in the indexed banking documents to answer this question reliably.",
                "risk_label": "Unknown",
                "justification": "No retrieved evidence was available.",
                "evidence_chunk_ids": [],
                "evidence_status": "insufficient",
            }

        response = None
        last_error: Exception | None = None
        used_rows = retrieved_rows
        for variant in DEFAULT_PROMPT_VARIANTS:
            used_rows = retrieved_rows[: variant["max_chunks"]]
            prompt = build_decision_prompt(
                query,
                used_rows,
                max_chunks=variant["max_chunks"],
                max_chunk_chars=variant["max_chunk_chars"],
            )
            try:
                response = self.client.generate(
                    prompt,
                    temperature=0.0,
                    options={"num_ctx": 4096, "num_predict": 320},
                )
                break
            except (OllamaUnavailableError, OllamaModelNotFoundError):
                raise
            except requests.HTTPError as exc:
                last_error = exc
                continue

        if response is None:
            return {
                "answer": "I could not generate a grounded answer from the local model within the available context and hardware limits.",
                "risk_label": "Unknown",
                "justification": f"Local generation failed after prompt backoff: {last_error}",
                "evidence_chunk_ids": [row.get("chunk_id") for row in used_rows[:2]],
                "evidence_status": "insufficient",
            }

        raw_text = response.text.strip()
        json_text = _extract_json_object(raw_text) or raw_text
        try:
            parsed = json.loads(json_text)
            return _sanitize_decision(parsed, used_rows, query)
        except json.JSONDecodeError:
            recovered = _recover_partial_decision(raw_text, used_rows, query)
            if recovered is not None:
                return recovered
            return {
                "answer": raw_text,
                "risk_label": "Unknown",
                "justification": "The model did not return strict JSON.",
                "evidence_chunk_ids": [row.get("chunk_id") for row in used_rows[:3]],
                "evidence_status": "unparsed",
            }
