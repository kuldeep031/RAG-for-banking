# Banking Risk RAG

Local semester project for:

`Impact of Embedding Model Selection on Retrieval and Decision Quality in RAG-Based Banking Risk Assessment Systems`

## What this project studies

- How embedding model choice affects retrieval quality
- How retrieval quality affects grounded answer quality
- How embedding choice affects risk-decision quality
- How retrieval quality, answer quality, latency, and local resource cost trade off in a banking RAG system

## Core stack

- Python
- FAISS
- Sentence Transformers
- Ollama
- Streamlit
- Optional RAGAS

## Current project structure

- `data/raw_pdfs/`: manually downloaded public banking/regulatory PDFs
- `data/extracted_pages/`: page-level extracted text
- `data/cleaned_docs/`: cleaned page text
- `data/chunks/`: retrieval chunks with metadata
- `indexes/`: one FAISS index per embedding model
- `results/`: retrieval, answer, and RAGAS outputs
- `src/`: backend code
- `app/`: Streamlit UI

## First setup

Create and activate a virtual environment, then install:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pull the local LLM:

```powershell
ollama pull llama3.2
```

## First run order

1. Put PDFs into `data/raw_pdfs/`
2. Run `src/pdf_extract.py`
3. Run `src/clean_text.py`
4. Run `src/chunking.py`
5. Build FAISS indexes
6. Run the pipeline
7. Evaluate

## Evaluation stack

- Retrieval metrics: `Precision@3`, `Recall@3`, `Recall@5`, `MRR`, `nDCG@5`
- Decision metrics: `label_accuracy`, `citation_hit_rate`, `avg_answer_similarity`
- Latency metrics: average retrieval and total response time
- Local resource cost proxies: model parameters, approximate RAM footprint, index size on disk
- Optional RAGAS metrics: `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`

## Run experiments

```powershell
.\.venv\Scripts\python.exe run_experiments.py
```

By default this runs the core 3-model embedding comparison:

- retrieval benchmarking
- answer / decision benchmarking

This writes:

- `results/retrieval/model_summary.csv`
- `results/answers/model_summary.csv`
- `results/ragas/ragas_input.csv`

## Optional RAGAS

The project is designed to be complete without cloud APIs. RAGAS is optional.
If you still want to run it locally against Ollama:

```powershell
.\.venv\Scripts\python.exe run_experiments.py --run-ragas --ragas-provider ollama --ragas-model llama3.2
```

## Launch the demo

```powershell
.\.venv\Scripts\activate
streamlit run app\streamlit_app.py
```

## Note

The implementation is intentionally local-first and free to run after model downloads.
The project keeps the experimental focus on embedding comparison rather than multi-agent orchestration.
