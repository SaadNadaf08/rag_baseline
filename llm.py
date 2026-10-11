"""Pluggable LLM client with retry/backoff for rate limits.
Uses plain requests against any OpenAI-compatible /chat/completions
endpoint. `dummy` mode lets you test the full pipeline offline."""
import os
import time

import requests

from config import settings


class DummyLLM:
    def chat(self, prompt: str) -> str:
        return ("[offline test mode] The pipeline works end-to-end. "
                "Set LLM=openai and OPENAI_API_KEY for real answers [1].")


class OpenAICompatLLM:
    def __init__(self, model: str, api_key: str, base_url: str, max_retries: int = 6):
        self.model, self.api_key, self.base_url = model, api_key, base_url
        self.max_retries = max_retries

    def chat(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                reason = ""
                try:
                    reason = r.json()["error"].get("code", "")
                except Exception:
                    pass
                wait = min(2 ** attempt, 60)
                try:
                    wait = max(wait, int(r.headers.get("Retry-After", 0)))
                except ValueError:
                    pass
                print(f"[warn] HTTP {r.status_code} ({reason}) - retrying in {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"still rate-limited after {self.max_retries} retries")


def get_llm():
    if settings.llm.lower() == "dummy":
        return DummyLLM()
    return OpenAICompatLLM(settings.llm_model, os.environ["OPENAI_API_KEY"],
                           settings.llm_base_url)