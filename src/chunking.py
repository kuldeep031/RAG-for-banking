from pathlib import Path

import pandas as pd

from src.config import CHUNKS_DIR, CLEANED_DIR, RetrievalConfig, ensure_directories
from src.utils import load_jsonl


def chunk_words(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]).strip())
        if end == len(words):
            break
        start = end - overlap
    return [chunk for chunk in chunks if chunk]


def build_chunks(config: RetrievalConfig = RetrievalConfig()) -> Path:
    ensure_directories()
    cleaned_rows = load_jsonl(CLEANED_DIR / "cleaned_pages.jsonl")
    chunk_rows: list[dict] = []

    for row in cleaned_rows:
        text = row.get("clean_text", "")
        words = text.split()
        if not words:
            continue

        chunks = chunk_words(
            words,
            chunk_size=config.chunk_size_words,
            overlap=config.chunk_overlap_words,
        )
        for chunk_index, chunk_text in enumerate(chunks, start=1):
            chunk_rows.append(
                {
                    "chunk_id": f"{row['doc_id']}_p{row['page_no']}_c{chunk_index}",
                    "doc_id": row["doc_id"],
                    "source_file": row["source_file"],
                    "page_no_start": row["page_no"],
                    "page_no_end": row["page_no"],
                    "section_heading": "",
                    "chunk_text": chunk_text,
                }
            )

    output_path = CHUNKS_DIR / "chunks.csv"
    pd.DataFrame(chunk_rows).to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    output = build_chunks()
    print(f"Saved chunks to {output}")
