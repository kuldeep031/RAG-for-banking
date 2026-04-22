import pandas as pd

from src.config import CHUNKS_DIR, EMBEDDING_MODELS, ensure_directories
from src.retriever import RetrieverAgent


def build_all_indexes() -> list[str]:
    ensure_directories()
    chunks_path = CHUNKS_DIR / "chunks.csv"
    chunks_df = pd.read_csv(chunks_path)
    if chunks_df.empty:
        raise ValueError("Chunk file is empty. Run extraction, cleaning, and chunking first.")

    built_models: list[str] = []
    for model_key in EMBEDDING_MODELS:
        agent = RetrieverAgent(model_key)
        agent.build_index_from_chunks(chunks_df)
        built_models.append(model_key)
    return built_models


if __name__ == "__main__":
    models = build_all_indexes()
    print(f"Built indexes for: {', '.join(models)}")
