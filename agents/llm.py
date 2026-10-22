"""Minimal LLM clients (agents/llm.py): a local Ollama server and the Gemini API.

Both use the standard library only and mimic what ``RepairerAgent`` expects from a chat model
(``invoke(messages)`` returning an object with ``.content``). Each records calls, tokens and wall
time so that LLM runs report their cost, and each uses temperature 0 (and a fixed seed where the
API has one), so a run is reproducible for a fixed model build, up to server-side nondeterminism.

Gemini specifics
* The API key is read from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``), for example from a
  git-ignored ``.env`` file. It is sent in the ``x-goog-api-key`` header, never in the URL, and is
  scrubbed from every error message.
* Free-tier quotas change and differ by model, so requests are spaced by
  ``QREPAIR_LLM_MIN_INTERVAL`` seconds (default 4) and 429/5xx replies are retried with backoff,
  honouring a server-supplied ``retryDelay``. Calls that still fail are counted in ``errors`` so
  that infrastructure failures can be told apart from the model simply being wrong.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, List, Optional


class LLMError(RuntimeError):
    """An API or transport failure (not a wrong answer)."""


class QuotaExhausted(LLMError):
    """The API says the quota will not reset for a long time; retrying would only burn time."""


QUOTA_WAIT_LIMIT = 300.0          # a server-supplied retryDelay above this (seconds) means "out of quota"


class _Usage:
    calls = 0
    prompt_tokens = 0
    output_tokens = 0
    seconds = 0.0
    errors = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


class OllamaChat(_Usage):
    def __init__(self, model: str, host: str = "", temperature: float = 0.0,
                 num_predict: int = 500, seed: int = 0, timeout: float = 600.0):
        self.model, self.temperature = model, temperature
        self.host = (host or os.environ.get("QREPAIR_OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.num_predict, self.seed, self.timeout = num_predict, seed, timeout

    def invoke(self, messages: List[Any]) -> SimpleNamespace:
        role = {"system": "system", "human": "user", "ai": "assistant"}
        payload = {
            "model": self.model, "stream": False,
            "messages": [{"role": role.get(getattr(m, "type", "human"), "user"), "content": m.content} for m in messages],
            "options": {"temperature": self.temperature, "num_predict": self.num_predict, "seed": self.seed},
        }
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read().decode("utf-8"))
        self.seconds += time.time() - t0
        self.calls += 1
        self.prompt_tokens += int(out.get("prompt_eval_count", 0))
        self.output_tokens += int(out.get("eval_count", 0))
        return SimpleNamespace(content=out["message"]["content"])


class GeminiChat(_Usage):
    """Gemini ``generateContent`` over REST, with spacing and retry for free-tier quotas."""

    RETRYABLE = (429, 500, 502, 503, 504)

    @property
    def tokens(self) -> int:                      # thinking tokens are billed as output
        return self.prompt_tokens + self.output_tokens + self.thought_tokens

    def __init__(self, model: str, api_key: str = "", temperature: float = 0.0, max_output_tokens: int = 8192,
                 timeout: float = 120.0, min_interval: Optional[float] = None, max_retries: int = 6,
                 thinking_budget: Optional[int] = None, base: str = ""):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        if not key:
            raise LLMError("no Gemini API key: set GEMINI_API_KEY (for example in a git-ignored .env file)")
        self._key = key
        self.model, self.temperature, self.max_output_tokens = model, temperature, max_output_tokens
        self.timeout, self.max_retries = timeout, max_retries
        env_think = os.environ.get("QREPAIR_GEMINI_THINKING")
        self.thinking_budget = thinking_budget if thinking_budget is not None else (int(env_think) if env_think else None)
        self.min_interval = float(os.environ.get("QREPAIR_LLM_MIN_INTERVAL", "4")) if min_interval is None else min_interval
        self.base = (base or os.environ.get("QREPAIR_GEMINI_BASE", "https://generativelanguage.googleapis.com")).rstrip("/")
        self.retries = 0
        self.rate_limited = 0
        self.thought_tokens = 0
        self.inline_system = False          # models without systemInstruction (Gemma): system text goes in the user turn
        self._last = 0.0

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def _payload(self, messages: List[Any]) -> dict:
        system = [m.content for m in messages if getattr(m, "type", "") == "system"]
        contents = [{"role": "model" if getattr(m, "type", "") == "ai" else "user", "parts": [{"text": m.content}]}
                    for m in messages if getattr(m, "type", "") != "system"]
        cfg = {"temperature": self.temperature, "maxOutputTokens": self.max_output_tokens}
        if self.thinking_budget is not None:
            cfg["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        payload = {"contents": contents, "generationConfig": cfg}
        if system and self.inline_system and contents:
            contents[0]["parts"][0]["text"] = "\n\n".join(system) + "\n\n" + contents[0]["parts"][0]["text"]
        elif system:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        return payload

    @staticmethod
    def _retry_delay(body: str) -> float:
        m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', body)
        return float(m.group(1)) if m else 0.0

    def invoke(self, messages: List[Any]) -> SimpleNamespace:
        url = f"{self.base}/v1beta/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._key}
        last_err = ""
        for attempt in range(self.max_retries + 1):
            wait = self._last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            t0 = time.time()
            try:
                data = json.dumps(self._payload(messages)).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = self._scrub(e.read().decode("utf-8", "replace"))
                self._last = time.time()
                last_err = f"HTTP {e.code}: {body[:200]}"
                if (e.code == 400 and "instruction" in body.lower() and not self.inline_system
                        and any(getattr(m, "type", "") == "system" for m in messages)):
                    self.inline_system = True                           # retry once with the system text inlined
                    continue
                if e.code == 429 and self._retry_delay(body) > QUOTA_WAIT_LIMIT:
                    self.errors += 1
                    raise QuotaExhausted(self._scrub(f"Gemini quota exhausted (resets in about "
                                                     f"{self._retry_delay(body) / 3600:.1f} h)")) from None
                if e.code in self.RETRYABLE and attempt < self.max_retries:
                    self.retries += 1
                    self.rate_limited += int(e.code == 429)
                    time.sleep(min(max(self._retry_delay(body), 2.0 * 2 ** attempt), 120.0))
                    continue
                break
            except Exception as e:                                     # network, timeout, bad JSON
                self._last = time.time()
                last_err = type(e).__name__
                if attempt < self.max_retries:
                    self.retries += 1
                    time.sleep(min(2.0 * 2 ** attempt, 60.0))
                    continue
                break
            self._last = time.time()
            self.seconds += self._last - t0
            self.calls += 1
            u = out.get("usageMetadata", {})
            self.prompt_tokens += int(u.get("promptTokenCount", 0))
            self.output_tokens += int(u.get("candidatesTokenCount", 0))
            self.thought_tokens += int(u.get("thoughtsTokenCount", 0))
            cands = out.get("candidates") or []
            parts = cands[0].get("content", {}).get("parts", []) if cands else []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            return SimpleNamespace(content=text)
        self.errors += 1
        raise LLMError(self._scrub(f"Gemini request failed after {self.max_retries + 1} attempts: {last_err}"))


def make_llm(spec: str):
    """``"gemini:<model>"`` -> GeminiChat; ``"ollama:<model>"`` or a bare model name -> OllamaChat."""
    kind, _, name = spec.partition(":")
    if kind == "gemini":
        return GeminiChat(name)
    if kind == "ollama":
        return OllamaChat(name)
    return OllamaChat(spec)
