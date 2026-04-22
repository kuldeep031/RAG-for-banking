import re
from pathlib import Path

from src.config import CLEANED_DIR, EXTRACTED_DIR, ensure_directories
from src.utils import load_jsonl, save_jsonl


def clean_page_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"-\n", "", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\d+\s*\n", "\n", cleaned)
    return cleaned.strip()


def run_cleaning(input_path: Path | None = None) -> Path:
    ensure_directories()
    source_path = input_path or (EXTRACTED_DIR / "pages.jsonl")
    rows = load_jsonl(source_path)

    cleaned_rows: list[dict] = []
    for row in rows:
        cleaned_row = dict(row)
        cleaned_row["clean_text"] = clean_page_text(row.get("raw_text", ""))
        cleaned_rows.append(cleaned_row)

    output_path = CLEANED_DIR / "cleaned_pages.jsonl"
    save_jsonl(output_path, cleaned_rows)
    return output_path


if __name__ == "__main__":
    output = run_cleaning()
    print(f"Saved cleaned text to {output}")
