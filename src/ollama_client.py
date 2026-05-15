from dataclasses import dataclass

import numpy as np
import requests


OLLAMA_BASE_URL = "http://localhost:11434/api"
GENERATE_URL = f"{OLLAMA_BASE_URL}/generate"
EMBED_URL = f"{OLLAMA_BASE_URL}/embed"
LEGACY_EMBEDDINGS_URL = f"{OLLAMA_BASE_URL}/embeddings"


@dataclass
class OllamaResponse:
    text: str
    raw: dict


class OllamaClient:
    def __init__(self, model: str) -> None:
        self.model = model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        options: dict | None = None,
    ) -> OllamaResponse:
        request_options = {"temperature": temperature}
        if options:
            request_options.update(options)
        response = requests.post(
            GENERATE_URL,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": request_options,
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        return OllamaResponse(text=payload.get("response", "").strip(), raw=payload)

    def embed(self, prompt: str) -> np.ndarray:
        return self.embed_many([prompt])[0]

    @staticmethod
    def _is_context_length_error(response: requests.Response) -> bool:
        if response.status_code != 400:
            return False
        try:
            payload = response.json()
            message = str(payload.get("error", ""))
        except ValueError:
            message = response.text
        return "context length" in message.lower()

    @staticmethod
    def _shrink_text(text: str) -> str:
        words = text.split()
        if len(words) <= 32:
            return text[: max(256, int(len(text) * 0.8))]
        new_word_count = max(32, int(len(words) * 0.8))
        return " ".join(words[:new_word_count])

    def _embed_single_with_backoff(self, prompt: str) -> np.ndarray:
        candidate = prompt
        for _ in range(8):
            response = requests.post(
                EMBED_URL,
                json={
                    "model": self.model,
                    "input": [candidate],
                },
                timeout=180,
            )
            if response.ok:
                payload = response.json()
                embeddings = payload.get("embeddings")
                if not embeddings:
                    raise RuntimeError(
                        f"Ollama /api/embed returned no embeddings for model {self.model}."
                    )
                return np.asarray(embeddings[0], dtype="float32")

            if not self._is_context_length_error(response):
                raise RuntimeError(
                    f"Ollama /api/embed failed for model {self.model} "
                    f"with status {response.status_code}: {response.text}"
                )

            shrunk = self._shrink_text(candidate)
            if shrunk == candidate:
                break
            candidate = shrunk

        raise RuntimeError(
            f"Ollama /api/embed could not fit the input within context length for model {self.model}."
        )

    def embed_many(self, prompts: list[str], batch_size: int = 16) -> np.ndarray:
        # Use the current batch embeddings endpoint in smaller chunks because
        # large all-at-once requests can trigger server-side 500 errors on
        # consumer hardware. Only fall back to the legacy endpoint if the
        # Ollama install does not support /api/embed.
        batched_vectors: list[np.ndarray] = []

        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            response = requests.post(
                EMBED_URL,
                json={
                    "model": self.model,
                    "input": batch,
                },
                timeout=300,
            )

            if response.ok:
                payload = response.json()
                embeddings = payload.get("embeddings")
                if not embeddings:
                    raise RuntimeError(
                        f"Ollama /api/embed returned no embeddings for model {self.model}."
                    )
                batched_vectors.append(np.asarray(embeddings, dtype="float32"))
                continue

            if self._is_context_length_error(response):
                batch_vectors = [
                    self._embed_single_with_backoff(prompt)
                    for prompt in batch
                ]
                batched_vectors.append(np.vstack(batch_vectors))
                continue

            if response.status_code != 404:
                raise RuntimeError(
                    f"Ollama /api/embed failed for model {self.model} "
                    f"with status {response.status_code}: {response.text}"
                )

            legacy_vectors: list[np.ndarray] = []
            for prompt in batch:
                legacy_response = requests.post(
                    LEGACY_EMBEDDINGS_URL,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                    },
                    timeout=120,
                )
                legacy_response.raise_for_status()
                payload = legacy_response.json()
                vector = payload.get("embedding")
                if not vector:
                    raise RuntimeError(
                        f"Ollama legacy embeddings call returned no vector for model {self.model}."
                    )
                legacy_vectors.append(np.asarray(vector, dtype="float32"))
            batched_vectors.append(np.vstack(legacy_vectors))

        return np.vstack(batched_vectors)
