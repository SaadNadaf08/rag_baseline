"""Tiny cosine-similarity vector store (numpy + files). Intentionally simple
so the baseline has no infrastructure. Project 1-3 swap in Qdrant later."""
import json
import os

import numpy as np


class SimpleVectorStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.chunks: list[dict] = []
        self.matrix: np.ndarray | None = None  # rows are L2-normalized

    def build(self, chunks: list[dict], vectors: np.ndarray):
        assert len(chunks) == len(vectors)
        self.chunks = chunks
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = vectors / norms

    def save(self):
        os.makedirs(self.index_dir, exist_ok=True)
        np.save(os.path.join(self.index_dir, "vectors.npy"), self.matrix)
        with open(os.path.join(self.index_dir, "chunks.json"), "w") as f:
            json.dump(self.chunks, f)

    def load(self):
        self.matrix = np.load(os.path.join(self.index_dir, "vectors.npy"))
        with open(os.path.join(self.index_dir, "chunks.json")) as f:
            self.chunks = json.load(f)
        return self

    def query(self, vector: np.ndarray, top_k: int) -> list[dict]:
        v = vector / (np.linalg.norm(vector) or 1.0)
        scores = self.matrix @ v
        order = np.argsort(-scores)[:top_k]
        return [dict(self.chunks[i], score=float(scores[i])) for i in order]
