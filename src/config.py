from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
EXTRACTED_DIR = DATA_DIR / "extracted_pages"
CLEANED_DIR = DATA_DIR / "cleaned_docs"
CHUNKS_DIR = DATA_DIR / "chunks"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = PROJECT_ROOT / "indexes"
RESULTS_DIR = PROJECT_ROOT / "results"
APP_DIR = PROJECT_ROOT / "app"

DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 60
DEFAULT_SCORE_THRESHOLD = 0.35

EMBEDDING_MODELS = {
    "all_minilm_l6_v2": "sentence-transformers/all-MiniLM-L6-v2",
    "e5_small_v2": "intfloat/e5-small-v2",
    "bge_small_en_v1_5": "BAAI/bge-small-en-v1.5",
}

# Proxy for Cost / Memory footprint Objective 3
MODEL_SPECS = {
    "all_minilm_l6_v2": {"params_millions": 22.7, "dim": 384, "memory_mb_approx": 90},
    "e5_small_v2": {"params_millions": 33.4, "dim": 384, "memory_mb_approx": 130},
    "bge_small_en_v1_5": {"params_millions": 33.4, "dim": 384, "memory_mb_approx": 130},
}


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_key: str = "all_minilm_l6_v2"
    top_k: int = DEFAULT_TOP_K
    chunk_size_words: int = DEFAULT_CHUNK_SIZE
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    ollama_model: str = DEFAULT_OLLAMA_MODEL


def ensure_directories() -> None:
    required_dirs = [
        RAW_PDF_DIR,
        EXTRACTED_DIR,
        CLEANED_DIR,
        CHUNKS_DIR,
        EVAL_DIR,
        INDEX_DIR,
        RESULTS_DIR / "retrieval",
        RESULTS_DIR / "answers",
        RESULTS_DIR / "ragas",
        RESULTS_DIR / "plots",
        APP_DIR,
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
