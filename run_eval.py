"""The before/after machine: run every eval question, score retrieval hit rate
and answer quality, write eval/results.csv. Run after EVERY pipeline change.

    python run_eval.py
"""
import csv
import json
import os

from config import settings
from embeddings import get_embedder
from llm import get_llm
from rag import RAGPipeline
from vector_store import SimpleVectorStore

JUDGE_PROMPT = """Score how well the answer matches the reference.

Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

Return ONLY one integer 1-5 (5 = fully correct, 1 = wrong or "I don't know")."""


def main():
    with open("eval/eval_set.jsonl") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    store = SimpleVectorStore(settings.index_dir).load()
    pipeline = RAGPipeline(store, get_embedder(), get_llm())
    judge = get_llm()

    rows, hits, scores = [], 0, []
    for q in questions:
        result = pipeline.answer(q["question"])
        retrieved_docs = {s["doc_id"] for s in result["sources"]}
        hit = q.get("gold_doc") in retrieved_docs
        hits += hit
        try:
            score = int(judge.chat(JUDGE_PROMPT.format(
                question=q["question"], reference=q.get("reference", ""),
                candidate=result["answer"])).strip()[:1])
        except (ValueError, IndexError):
            score = 0
        scores.append(score)
        rows.append({
            "question": q["question"], "gold_doc": q.get("gold_doc", ""),
            "retrieval_hit": hit, "judge_score": score,
            "latency_ms": result["latency_ms"], "answer": result["answer"],
            "retrieved_docs": ";".join(sorted(retrieved_docs)),
        })

    os.makedirs("eval", exist_ok=True)
    with open("eval/results.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=list(rows[0])).writeheader()
        csv.DictWriter(f, fieldnames=list(rows[0])).writerows(rows)

    n = len(questions)
    print(f"\n=== Eval summary ({n} questions) ===")
    print(f"retrieval hit rate : {hits}/{n} = {hits / n:.0%}")
    print(f"avg judge score    : {sum(scores) / n:.2f} / 5")
    print("Full per-question results: eval/results.csv")


if __name__ == "__main__":
    main()
