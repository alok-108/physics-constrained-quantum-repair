"""Minimal local-LLM client (agents/llm.py).

Talks to an Ollama server (default http://localhost:11434) with the standard library only. The
interface mimics what ``RepairerAgent`` expects from a chat model (``invoke(messages)`` returning an
object with ``.content``) and records token and time usage so that LLM runs report their cost.

Temperature is 0 and the sampling seed is fixed, so a run is reproducible for a fixed model build.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from types import SimpleNamespace
from typing import Any, List


class OllamaChat:
    def __init__(self, model: str, host: str = "", temperature: float = 0.0,
                 num_predict: int = 700, seed: int = 0, timeout: float = 600.0):
        self.model, self.temperature = model, temperature
        self.host = (host or os.environ.get("QREPAIR_OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.num_predict, self.seed, self.timeout = num_predict, seed, timeout
        self.calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

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
