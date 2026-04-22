from pathlib import Path

from src.config import EXTRACTED_DIR, RAW_PDF_DIR, ensure_directories
from src.utils import save_jsonl, slugify


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    rows: list[dict] = []
    text_found = False

    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                rows.append(
                    {
                        "doc_id": slugify(pdf_path.stem),
                        "source_file": pdf_path.name,
                        "page_no": page_index,
                        "raw_text": text,
                    }
                )
                if text.strip():
                    text_found = True
    except Exception:
        rows = []

    if text_found:
        return rows

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    fallback_rows: list[dict] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        fallback_rows.append(
            {
                "doc_id": slugify(pdf_path.stem),
                "source_file": pdf_path.name,
                "page_no": page_index,
                "raw_text": text,
            }
        )
    return fallback_rows


def run_extraction() -> Path:
    ensure_directories()
    all_rows: list[dict] = []
    for pdf_path in sorted(RAW_PDF_DIR.glob("*.pdf")):
        all_rows.extend(extract_pdf_pages(pdf_path))

    output_path = EXTRACTED_DIR / "pages.jsonl"
    save_jsonl(output_path, all_rows)
    return output_path


if __name__ == "__main__":
    output = run_extraction()
    print(f"Saved extracted pages to {output}")
