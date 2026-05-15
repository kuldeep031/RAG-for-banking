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

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_TOP_K = 5
# Keep chunks small enough to fit safely inside local embedding model context
# limits while remaining large enough to preserve banking-policy semantics.
DEFAULT_CHUNK_SIZE = 220
DEFAULT_CHUNK_OVERLAP = 40
DEFAULT_SCORE_THRESHOLD = 0.35

EMBEDDING_MODELS = {
    "all_minilm_l6_v2": {
        "backend": "sentence_transformers",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "display_name": "MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "params_millions": 22.7,
        "dim": 384,
        "memory_mb_approx": 90,
    },
    "e5_small_v2": {
        "backend": "sentence_transformers",
        "model_name": "intfloat/e5-small-v2",
        "display_name": "E5-Small-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "params_millions": 33.4,
        "dim": 384,
        "memory_mb_approx": 130,
    },
    "bge_small_en_v1_5": {
        "backend": "sentence_transformers",
        "model_name": "BAAI/bge-small-en-v1.5",
        "display_name": "BGE-Small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "params_millions": 33.4,
        "dim": 384,
        "memory_mb_approx": 130,
    },
    "nomic_embed_text": {
        "backend": "ollama",
        "model_name": "nomic-embed-text",
        "display_name": "Nomic Embed Text",
        "query_prefix": "search_query: ",
        "passage_prefix": "search_document: ",
        "params_millions": 137.0,
        "dim": 768,
        "memory_mb_approx": 600,
    },
    "qwen3_embedding_0_6b": {
        "backend": "ollama",
        "model_name": "qwen3-embedding:0.6b",
        "display_name": "Qwen3 Embedding 0.6B",
        "query_prefix": "Instruct: Represent this query for retrieval.\nQuery: ",
        "passage_prefix": "Instruct: Represent this document for retrieval.\nDocument: ",
        "params_millions": 600.0,
        "dim": 1024,
        "memory_mb_approx": 900,
    },
}

DEFAULT_EXPERIMENT_EMBEDDING_KEYS = [
    "all_minilm_l6_v2",
    "e5_small_v2",
    "bge_small_en_v1_5",
]

RECOMMENDED_LOCAL_EMBEDDING_KEYS = [
    "nomic_embed_text",
    "qwen3_embedding_0_6b",
]

MODEL_SPECS = {
    model_key: {
        "params_millions": float(spec.get("params_millions", 0.0)),
        "dim": int(spec.get("dim", 0)),
        "memory_mb_approx": float(spec.get("memory_mb_approx", 0.0)),
    }
    for model_key, spec in EMBEDDING_MODELS.items()
}


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_key: str = "all_minilm_l6_v2"
    top_k: int = DEFAULT_TOP_K
    chunk_size_words: int = DEFAULT_CHUNK_SIZE
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    ollama_model: str = DEFAULT_OLLAMA_MODEL


def get_embedding_spec(model_key: str) -> dict:
    try:
        return EMBEDDING_MODELS[model_key]
    except KeyError as exc:
        raise KeyError(f"Unknown embedding model key: {model_key}") from exc


def friendly_embedding_name(model_key: str) -> str:
    spec = get_embedding_spec(model_key)
    return str(spec.get("display_name", model_key))


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
        RESULTS_DIR / "local_judge",
        RESULTS_DIR / "plots",
        APP_DIR,
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
