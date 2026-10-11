# ⚡ Production-Grade RAG Baseline & Evaluation Harness

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Clean](https://img.shields.io/badge/code%20style-pep8-green.svg)](https://pep8.org/)

A modular, lightweight, and extensible **Retrieval-Augmented Generation (RAG)** baseline built with zero bloated frameworks (no LangChain / LlamaIndex overhead). Designed to establish a measurable, reproducible benchmark for retrieval quality, latency, and answer accuracy before introducing complex architectures.

---

## 📌 Architecture Overview

```
                        ┌──────────────────────────────────────────────┐
                        │              INGESTION PIPELINE              │
                        │                                              │
                        │  Documents  ──► Sliding Window ──► Embedder  │
                        │ (data/corpus)     Chunking         (ST / OAI)│
                        └───────────────────────┬──────────────────────┘
                                                │
                                                ▼
┌───────────────────────────────────────┐   Vector Store (index/)
│            QUERY PIPELINE             │   ├── vectors.npy
│                                       │   └── chunks.json
│ User Query ──► Query Embedder         │           │
│                      │                │           │
│                      ▼                │           │
│               Cosine Retrieval ◄──────┴───────────┘
│                 (Top-K Chunks)
│                      │
│                      ▼
│            Prompt Context Assembly
│                      │
│                      ▼
│               LLM Generation ───────► Answer with Citations [1]
│             (OpenAI / Local LLM)
└───────────────────────────────────────┘
```

---

## ✨ Key Features

- **Framework-Agnostic & Zero Bloat**: Built purely with standard Python, NumPy, Requests, and FastAPI. Transparent and easy to trace.
- **Offline Smoke-Test Mode**: Run full end-to-end ingestion, retrieval, and evaluation offline without API keys or GPU downloads (`EMBEDDER=hash`, `LLM=dummy`).
- **Pluggable Embedders**: Supports local Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`), OpenAI embeddings, and deterministic hashing for testing.
- **OpenAI-Compatible LLM Client**: Works seamlessly with OpenAI (`gpt-4o-mini`), Ollama, vLLM, or LiteLLM, featuring built-in exponential backoff retry logic for rate limits (`HTTP 429`).
- **Standardized Evaluation Harness**: Automatically computes Retrieval Hit Rate @ K and LLM-as-a-judge accuracy metrics, exporting structured results to CSV.
- **Production-Ready REST API**: Packaged with a performant FastAPI endpoint (`POST /ask`).

---

## 📂 Repository Structure

```text
├── data/
│   └── corpus/          # Raw Markdown / text documents to index
├── eval/
│   ├── eval_set.jsonl   # Ground-truth evaluation dataset (questions, references, gold docs)
│   └── results.csv      # Generated benchmark performance metrics
├── index/               # Generated vector index (vectors.npy, chunks.json)
├── api.py               # Production FastAPI service (POST /ask)
├── chunking.py          # Word-level sliding window chunker with configurable overlap
├── config.py            # Centralized typed configuration loaded via environment variables
├── embeddings.py        # Abstracted embedder implementations (ST, OpenAI, Hash)
├── ingest.py            # Ingestion pipeline: loads corpus -> chunks -> embeds -> stores index
├── llm.py               # Pluggable LLM client with backoff retry handling & dummy fallback
├── make_eval_set.py     # Synthetic question-answer generator for bootstrap evaluation
├── rag.py               # Core retrieval-augmented generation orchestrator
├── run_eval.py          # Evaluation runner (Hit Rate, LLM Judge, latency reporting)
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── .env.example         # Environment variable template
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/rag-baseline.git
cd rag-baseline

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration Setup

Copy the sample environment file:

```bash
cp .env.example .env
```

Configure your `.env` file according to your target setup:

```env
OPENAI_API_KEY=sk-proj-your-api-key-here
EMBEDDER=st
LLM=openai
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
TOP_K=5
```

---

## 🧪 Usage & Workflows

### Option A: Offline Smoke Test (No API Key Required)

Verify that data flow, chunking, storage, and retrieval work end-to-end without spending credits or downloading models:

```bash
# Set offline overrides
export EMBEDDER=hash
export LLM=dummy

# Build index and run evaluation
python ingest.py
python run_eval.py
```

---

### Option B: Production Run (Standard Baseline)

Build a vector index with Sentence-Transformers and run generation through an LLM:

1. **Ingest the corpus:**
   ```bash
   python ingest.py
   ```
   *Parses `data/corpus/*.md`, splits text into chunks, computes dense embeddings, and serializes the index to `index/`.*

2. **Run evaluation:**
   ```bash
   python run_eval.py
   ```
   *Evaluates retrieval hit rate and scores generations using LLM-as-a-judge against `eval/eval_set.jsonl`. Results are written to `eval/results.csv`.*

3. **Generate synthetic eval pairs (optional):**
   ```bash
   python make_eval_set.py
   ```
   *Generates candidate test questions from your corpus. (Always review candidate pairs manually before benchmarking).*

---

## 🌐 Serving via API

Start the FastAPI application with Uvicorn:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Query Endpoint (`POST /ask`)

**Request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How many days of annual PTO do full-time employees receive?"}'
```

**Response:**
```json
{
  "answer": "Full-time employees receive 20 days of paid time off (PTO) annually [1].",
  "citations": [
    {
      "id": 1,
      "doc": "pto.md",
      "text": "Full-time team members accrue 20 paid time off days per calendar year..."
    }
  ]
}
```

---

## ⚙️ Configuration Reference

Every knob can be controlled via `.env` or system environment variables:

| Variable | Default | Options | Description |
|---|---|---|---|
| `EMBEDDER` | `st` | `st`, `openai`, `hash` | Embedding provider (`st` uses local MiniLM) |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | *model name* | Hugging Face or OpenAI model ID |
| `LLM` | `openai` | `openai`, `dummy` | LLM generation provider |
| `LLM_MODEL` | `gpt-4o-mini` | *model name* | Target generation model |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | *URL* | Base endpoint (supports Ollama / vLLM) |
| `TOP_K` | `5` | Integer | Number of chunks retrieved per query |
| `CHUNK_WORDS` | `350` | Integer | Word count per sliding chunk (~500 tokens) |
| `OVERLAP_WORDS`| `50` | Integer | Word overlap between adjacent chunks |
| `EMBED_BATCH_SIZE` | `32` | Integer | Batch size during embedding computation |

---

## 📊 Evaluation & Benchmarking

The evaluation harness in [`run_eval.py`](run_eval.py) computes key information-retrieval metrics across test queries:

1. **Retrieval Hit Rate (`hit@k`)**: Checks if the target `gold_doc` was present within the top-$k$ retrieved chunks.
2. **Faithfulness & Groundedness**: Uses the LLM as an impartial judge to score whether the synthesized answer directly follows from the retrieved context.
3. **Reference Accuracy**: Compares generated responses against gold-standard references in `eval/eval_set.jsonl`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
