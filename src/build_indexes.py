import argparse

import pandas as pd

from src.config import CHUNKS_DIR, DEFAULT_EXPERIMENT_EMBEDDING_KEYS, EMBEDDING_MODELS, ensure_directories
from src.retriever import RetrieverAgent


def _resolve_embedding_keys(models_arg: str | None) -> list[str]:
    if not models_arg:
        return list(DEFAULT_EXPERIMENT_EMBEDDING_KEYS)
    keys = [item.strip() for item in models_arg.split(",") if item.strip()]
    unknown = [key for key in keys if key not in EMBEDDING_MODELS]
    if unknown:
        raise KeyError(
            f"Unknown embedding model keys: {', '.join(unknown)}. "
            f"Available keys: {', '.join(EMBEDDING_MODELS)}"
        )
    return keys


def build_all_indexes(model_keys: list[str] | None = None) -> list[str]:
    ensure_directories()
    chunks_path = CHUNKS_DIR / "chunks.csv"
    chunks_df = pd.read_csv(chunks_path)
    if chunks_df.empty:
        raise ValueError("Chunk file is empty. Run extraction, cleaning, and chunking first.")

    built_models: list[str] = []
    for model_key in (model_keys or list(DEFAULT_EXPERIMENT_EMBEDDING_KEYS)):
        agent = RetrieverAgent(model_key)
        agent.build_index_from_chunks(chunks_df)
        built_models.append(model_key)
    return built_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS indexes for selected embedding models.")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_EXPERIMENT_EMBEDDING_KEYS),
        help="Comma-separated embedding keys to build.",
    )
    args = parser.parse_args()

    models = build_all_indexes(_resolve_embedding_keys(args.models))
    print(f"Built indexes for: {', '.join(models)}")
