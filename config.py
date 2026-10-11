"""Central settings. Every knob is an environment variable so the
eval script can reproduce runs with the exact same config."""
import os
from dataclasses import dataclass

try:  # optional .env support
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
except FileNotFoundError:
    pass


@dataclass
class Settings:
    corpus_dir: str = "data/corpus"
    index_dir: str = "index"
    chunk_words: int = int(os.environ.get("CHUNK_WORDS", "350"))    # ~500 tokens
    overlap_words: int = int(os.environ.get("OVERLAP_WORDS", "50"))
    top_k: int = int(os.environ.get("TOP_K", "5"))
    embedder: str = os.environ.get("EMBEDDER", "st")                # st | openai | hash
    embed_model: str = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm: str = os.environ.get("LLM", "openai")                     # openai | dummy
    llm_model: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    embed_batch_size: int = int(os.environ.get("EMBED_BATCH_SIZE", "32"))


settings = Settings()
