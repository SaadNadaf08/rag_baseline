"""The RAG pipeline itself: embed query -> retrieve top-k -> prompt -> answer."""
import time

from config import settings

PROMPT_TEMPLATE = """You answer questions using ONLY the context chunks below,
which come from an internal document corpus.

Context:
{context}

Question: {question}

Rules:
- Answer using only the context above.
- Cite the chunk number in square brackets after each claim, like [1].
- If the context does not contain the answer, reply exactly: I don't know.

Answer:"""


class RAGPipeline:
    def __init__(self, store, embedder, llm, top_k: int = None):
        self.store, self.embedder, self.llm = store, embedder, llm
        self.top_k = top_k or settings.top_k

    def retrieve(self, query: str) -> list[dict]:
        q_vec = self.embedder.embed([query])[0]
        return self.store.query(q_vec, self.top_k)

    def answer(self, query: str) -> dict:
        t0 = time.time()
        hits = self.retrieve(query)
        context = "\n\n".join(
            f"[{i+1}] (source: {h['doc_id']}) {h['text']}" for i, h in enumerate(hits))
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)
        answer = self.llm.chat(prompt)
        return {
            "query": query,
            "answer": answer,
            "sources": [{"doc_id": h["doc_id"], "chunk_id": h["chunk_id"],
                         "score": round(h["score"], 4)} for h in hits],
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
