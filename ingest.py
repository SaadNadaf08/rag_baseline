"""Chunk + embed the corpus and save the index. Run after any corpus change.

    python ingest.py
"""
import os
import sys

from chunking import chunk_corpus
from config import settings
from embeddings import get_embedder
from vector_store import SimpleVectorStore


def load_corpus(corpus_dir: str) -> dict[str, str]:
    text_by_doc = {}
    for fname in sorted(os.listdir(corpus_dir)):
        if fname.endswith((".md", ".txt")):
            with open(os.path.join(corpus_dir, fname), encoding="utf-8") as f:
                text_by_doc[fname] = f.read()
    return text_by_doc


def main():
    corpus = load_corpus(settings.corpus_dir)
    if not corpus:
        sys.exit(f"No .md/.txt files found in {settings.corpus_dir}")
    chunks = chunk_corpus(corpus, settings.chunk_words, settings.overlap_words)
    print(f"{len(corpus)} docs -> {len(chunks)} chunks "
          f"({settings.chunk_words} words, {settings.overlap_words} overlap)")

    embedder = get_embedder()
    vectors = embedder.embed([c["text"] for c in chunks])
    store = SimpleVectorStore(settings.index_dir)
    store.build(chunks, vectors)
    store.save()
    print(f"Index saved to {settings.index_dir}/ "
          f"(dim={vectors.shape[1]}, embedder={type(embedder).__name__})")


if __name__ == "__main__":
    main()
