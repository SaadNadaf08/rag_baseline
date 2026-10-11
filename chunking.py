"""Word-based chunking with overlap. ~350 words is roughly 500 tokens."""

def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_words - overlap_words, 1)
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


def chunk_corpus(text_by_doc: dict[str, str], chunk_words: int, overlap_words: int) -> list[dict]:
    """Turn {doc_id: text} into a flat list of chunk records with stable ids."""
    out = []
    for doc_id, text in sorted(text_by_doc.items()):
        for i, chunk in enumerate(chunk_text(text, chunk_words, overlap_words)):
            out.append({
                "chunk_id": f"{doc_id}::part{i}",
                "doc_id": doc_id,
                "text": chunk,
            })
    return out
