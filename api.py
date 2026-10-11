"""FastAPI endpoint wrapping the pipeline.

    uvicorn api:app --reload
    curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
         -d '{"query": "How many PTO days do I get?"}'
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import settings
from embeddings import get_embedder
from llm import get_llm
from rag import RAGPipeline
from vector_store import SimpleVectorStore

app = FastAPI(title="RAG baseline")
pipeline = RAGPipeline(SimpleVectorStore(settings.index_dir).load(),
                       get_embedder(), get_llm())


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")
    return pipeline.answer(req.query)


@app.get("/health")
def health():
    return {"chunks": len(pipeline.store.chunks)}
