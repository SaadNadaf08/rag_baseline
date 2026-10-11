# RAG Baseline (Step 0)

Plain RAG pipeline with an eval harness. Build once; Projects 1-3 all improve it.

## What's inside
```
ingest.py          chunk corpus -> embed -> save index
rag.py             embed query -> retrieve top-5 -> answer with citations
run_eval.py        retrieval hit rate + LLM-judge scores -> eval/results.csv
make_eval_set.py   LLM-generate candidate eval questions (HAND-CHECK them)
api.py             FastAPI endpoint (POST /ask)
eval/eval_set.jsonl  the eval set - most valuable file in this repo
```

## Quick start (offline smoke test, no API key, no model downloads)
```bash
python -m venv .venv && source .venv/bin/activate   # python 3.10+
pip install -r requirements.txt

export EMBEDDER=hash      # toy embedder, wiring test only
export LLM=dummy
python ingest.py          # build the index from data/corpus
python run_eval.py        # full pipeline + CSV output
uvicorn api:app --reload  # then: curl -X POST localhost:8000/ask \
                         #  -H 'Content-Type: application/json' \
                         #  -d '{"query":"How many PTO days do I get?"}'
```

## Real run (your actual baseline numbers)
```bash
export OPENAI_API_KEY=sk-...
unset EMBEDDER LLM        # defaults: sentence-transformers + gpt-4o-mini
python ingest.py          # downloads MiniLM once (~80MB), then local embeddings
python make_eval_set.py   # generate candidate questions, then HAND-CHECK eval_set.jsonl
python run_eval.py        # <-- record these numbers: this is your "before"
```

## Swap in your own corpus
Replace the files in `data/corpus/` with your own `.md`/`.txt` docs
(50-200 docs works well), delete `index/`, re-run `python ingest.py`,
then hand-write `eval/eval_set.jsonl` (50-100 questions: `question`,
`reference`, `gold_doc`). Run `python run_eval.py` after every change.

## Tuning knobs
| Env var | Default | Meaning |
|---|---|---|
| `EMBEDDER` | `st` | `st` local sentence-transformers, `openai` API, `hash` offline test |
| `LLM` | `openai` | any OpenAI-compatible endpoint via `LLM_BASE_URL`, or `dummy` |
| `LLM_MODEL` | `gpt-4o-mini` | generation model |
| `TOP_K` | `5` | chunks retrieved per query |
| `CHUNK_WORDS` | `350` | ~500 tokens per chunk |
| `OVERLAP_WORDS` | `50` | chunk overlap |

## What NOT to do
- Don't tune chunk size or top-k without running `run_eval.py` before and after.
- Don't trust generated eval questions - hand-check every one.
- Don't report numbers that didn't come out of `eval/results.csv`.

## Next: the three projects
1. **Project 2 (permission-aware RAG)** - add `tenant_id` + ACL payload to chunks,
   pre-filter inside `SimpleVectorStore.query()`, add audit log + leakage tests.
2. **Project 1 (semantic cache + routing)** - cache layer in front of
   `RAGPipeline.answer()`, keyed by embedding similarity AND tenant; log
   cost/latency per request.
3. **Project 3 (GraphRAG)** - entity extraction during ingest, Neo4j alongside
   the vector store, hybrid retrieval.
