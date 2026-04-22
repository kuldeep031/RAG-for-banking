from dataclasses import dataclass

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class OllamaResponse:
    text: str
    raw: dict


class OllamaClient:
    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.1) -> OllamaResponse:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return OllamaResponse(text=payload.get("response", "").strip(), raw=payload)
