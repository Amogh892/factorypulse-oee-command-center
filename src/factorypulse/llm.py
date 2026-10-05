"""Local LLM wrapper around IBM Granite 4.0 1B (GGUF) served in-process by llama.cpp.

Graceful fallback: if llama-cpp-python is missing or the model file is absent, `available()` is False
and callers switch to deterministic, rule-based explanations instead of failing.
"""
from __future__ import annotations

import time

from . import config

SYSTEM_RCA = (
    "You are a senior reliability engineer in a manufacturing plant. Answer strictly from the EVIDENCE provided. "
    "Do not invent sensor values, dates or part numbers. If the evidence is insufficient, say so. "
    "Be concise and use plain language a shift supervisor can act on."
)


class LocalLLM:
    def __init__(self, model_path=None):
        self.model_path = model_path or config.LLM_MODEL_PATH
        self._llm = None
        self.status = "not_loaded"
        self.error: str | None = None
        self.load_seconds: float | None = None

    def available(self) -> bool:
        if self._llm is not None:
            return True
        if self.status == "failed":
            return False
        if not self.model_path.exists():
            self.status, self.error = "failed", f"Model file not found: {self.model_path}"
            return False
        try:
            from llama_cpp import Llama  # imported lazily so the rest of the app works without it
            t0 = time.time()
            self._llm = Llama(model_path=str(self.model_path), n_ctx=config.LLM_CTX,
                              n_threads=config.LLM_THREADS, verbose=False)
            self.load_seconds = round(time.time() - t0, 1)
            self.status = "ready"
            return True
        except Exception as exc:  # noqa: BLE001 - we want *any* failure to degrade gracefully
            self.status, self.error = "failed", f"{type(exc).__name__}: {exc}"
            return False

    def chat(self, user: str, system: str = SYSTEM_RCA, max_tokens: int | None = None, temperature: float = 0.2) -> str:
        if not self.available():
            raise RuntimeError(self.error or "LLM unavailable")
        out = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens or config.LLM_MAX_TOKENS, temperature=temperature, top_p=0.9,
            repeat_penalty=1.1)
        return out["choices"][0]["message"]["content"].strip()

    def describe(self) -> dict:
        return dict(model=self.model_path.name, status=self.status, error=self.error,
                    load_seconds=self.load_seconds, ctx=config.LLM_CTX, threads=config.LLM_THREADS)


_singleton: LocalLLM | None = None


def get_llm() -> LocalLLM:
    global _singleton
    if _singleton is None:
        _singleton = LocalLLM()
    return _singleton
