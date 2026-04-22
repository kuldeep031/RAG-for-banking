import json
import re

from src.ollama_client import OllamaClient


def build_decision_prompt(query: str, retrieved_rows: list[dict]) -> str:
    evidence_blocks = []
    for row in retrieved_rows:
        evidence_blocks.append(
            f"Chunk ID: {row.get('chunk_id')}\n"
            f"Source: {row.get('source_file')} page {row.get('page_no_start')}\n"
            f"Text: {row.get('chunk_text')}\n"
        )

    evidence_text = "\n---\n".join(evidence_blocks)
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
        'Example output: {"answer":"...","risk_label":"Medium","justification":"...","evidence_chunk_ids":["DOC001_CH001"],"evidence_status":"sufficient"}\n\n'
        f"Question:\n{query}\n\n"
        f"Evidence:\n{evidence_text}\n"
    )


def _extract_json_object(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


class DecisionAgent:
    def __init__(self, model_name: str) -> None:
        self.client = OllamaClient(model_name)

    def decide(self, query: str, retrieved_rows: list[dict]) -> dict:
        prompt = build_decision_prompt(query, retrieved_rows)
        response = self.client.generate(prompt)

        raw_text = response.text.strip()
        json_text = _extract_json_object(raw_text) or raw_text
        try:
            parsed = json.loads(json_text)
            parsed.setdefault("answer", "")
            parsed.setdefault("risk_label", "Unknown")
            parsed.setdefault("justification", "")
            parsed.setdefault("evidence_chunk_ids", [])
            parsed.setdefault("evidence_status", "insufficient")
            return parsed
        except json.JSONDecodeError:
            return {
                "answer": raw_text,
                "risk_label": "Unknown",
                "justification": "The model did not return strict JSON.",
                "evidence_chunk_ids": [row.get("chunk_id") for row in retrieved_rows[:3]],
                "evidence_status": "unparsed",
            }
