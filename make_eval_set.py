"""Generate candidate eval questions (2 per doc) with the LLM, then HAND-CHECK
them - this is the step that makes the numbers believable.

    python make_eval_set.py          # writes eval/eval_set.jsonl
"""
import json
import os

from config import settings
from ingest import load_corpus
from llm import get_llm

GEN_PROMPT = """Here is one document from an internal corpus.

Document id: {doc_id}
---
{doc_text}
---

Write 2 questions that can ONLY be answered using this document (not generic
knowledge). For each give a short reference answer.
Return ONLY JSON, no prose:
[{{"question": "...", "reference": "...", "gold_doc": "{doc_id}"}}, ...]"""


def main():
    corpus = load_corpus(settings.corpus_dir)
    llm = get_llm()
    if type(llm).__name__ == "DummyLLM":
        print("LLM=dummy: skipping generation. eval/eval_set.jsonl already has "
              "starter questions for the sample corpus - hand-check those, or "
              "add your own in the same JSONL format.")
        return
    rows = []
    for doc_id, text in corpus.items():
        raw = llm.chat(GEN_PROMPT.format(doc_id=doc_id, doc_text=text[:4000]))
        raw = raw[raw.index("["):raw.rindex("]") + 1]
        rows.extend(json.loads(raw))
    os.makedirs("eval", exist_ok=True)
    with open("eval/eval_set.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} candidate questions to eval/eval_set.jsonl")
    print("NEXT: open the file, remove bad ones, fix gold_doc values by hand.")


if __name__ == "__main__":
    main()
