"""Real-bug pilot (src/eval_real.py): circuits from the Bugs4Q benchmark of real Qiskit bugs.

Bugs4Q (Apache-2.0, https://github.com/Z-928/Bugs4Q) stores each real bug as a buggy and a fixed Python
script written for old Qiskit releases. Both scripts of every entry were run under a compatibility shim
and the circuits they build were dumped; the entries whose buggy and fixed circuits differ in logic on
the same qubits became ``data_real/pairs/*``. Most Bugs4Q entries are library, API or version bugs and
have no circuit-level logic fix (see data_real/README.md), so this is a *pilot of 3 pairs*, not a benchmark.

    python src/eval_real.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src import dynamic as D


def evaluate(root: str = PROJECT_ROOT, budget: int = 3000) -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(root, "data_real", "metadata.csv"))
    rows = []
    for m in meta.to_dict("records"):
        read = lambda kind: open(os.path.join(root, "data_real", "pairs", f"{m['circuit_id']}_{kind}.qasm"), encoding="utf-8").read()
        bug, ref = read("buggy"), read("reference")
        verdict = D.validate(bug, D.parse(ref), device=bool(m.get("device_check", 1)))
        for arm in ("dyn_full", "dyn_random"):
            t = time.time()
            dev = bool(m.get("device_check", 1))
            res = (D.repair(bug, ref, 3, budget, seed_key=m["circuit_id"], device=dev) if arm == "dyn_full"
                   else D.random_repair(bug, ref, 3, budget, seed=1, device=dev))
            rows.append({"circuit_id": m["circuit_id"], "bugs4q_entry": m["bugs4q_entry"], "arm": arm,
                         "buggy_fails_at": verdict["failed_stage"], "accepted": int(res["success"]),
                         "oracle_queries": res.get("queries", 0), "duration_s": round(time.time() - t, 2),
                         "description": m["description"]})
    return pd.DataFrame(rows)


SYSTEM = ("You are a quantum circuit repair expert. Produce corrected OpenQASM 3.0 respecting device connectivity, "
          "depth, and coherence limits. Distinguish between logical fixes and noise-aware routing.\n"
          "Output ONLY the repaired OpenQASM 3.0 code, wrapped inside ```qasm ... ``` code block. "
          "Do not include conversational preamble.")
HW = "{'device': 'IBM FakeSherbrooke (127 qubits)', 'max_depth': 250, 'T1_typical': '250 us', 'T2_typical': '150 us'}"
FENCE = re.compile(r"```(?:qasm)?\n(.*?)```", re.DOTALL)


def llm_repair(llm, bug: str, reference, attempts: int = 3, with_example: bool = False, device: bool = True) -> dict:
    """LLM as the only repair engine: same K and stage-named feedback as the other LLM arms, no oracle, no search."""
    current, feedback, history, errors = bug, None, [], []
    for k in range(1, attempts + 1):
        spec = {"num_qubits": reference.n_qubits, "num_clbits": reference.n_clbits}
        if with_example:                       # a test-style specification: what the correct program outputs on |0...0>
            zero = np.zeros(2 ** reference.n_qubits, dtype=complex)
            zero[0] = 1.0
            dist = D.distribution(reference, zero)
            spec["expected_output_on_all_zero_input"] = {kk: round(v, 4) for kk, v in sorted(dist.items()) if v > 1e-9}
        parts = [f"Task specification:\n{spec}", f"Faulty OpenQASM 3.0:\n```qasm\n{current}\n```",
                 f"Hardware physics constraints:\n{HW}"]
        if feedback:
            parts.append(f"Validator feedback:\n{feedback}")
        parts.append("Produce the complete, corrected OpenQASM 3.0 circuit now.")
        msg = "\n\n".join(parts)
        cand = None
        try:
            text = llm.invoke([SimpleNamespace(type="system", content=SYSTEM),
                               SimpleNamespace(type="human", content=msg)]).content.strip()
            m = FENCE.search(text)
            cand = m.group(1).strip() if m else text
        except Exception as e:
            if type(e).__name__ == "QuotaExhausted":
                raise
            errors.append(type(e).__name__ + ": " + str(e)[:80])
        verdict = D.validate(cand, reference, device=device) if cand else {"status": "FAIL", "failed_stage": "generation",
                                                           "feedback": "no usable reply"}
        history.append({"attempt": k, "status": verdict["status"], "failed_stage": verdict["failed_stage"]})
        if verdict["status"] == "PASS":
            return {"success": True, "history": history, "errors": errors}
        feedback = f"Attempt {k} rejected at stage '{verdict['failed_stage']}'. {verdict['feedback']}"
        current = cand or current
    return {"success": False, "history": history, "errors": errors}


def evaluate_llm(spec: str, root: str = PROJECT_ROOT, with_example: bool = False) -> pd.DataFrame:
    from agents.llm import make_llm
    meta = pd.read_csv(os.path.join(root, "data_real", "metadata.csv"))
    llm = make_llm(spec)
    rows = []
    for m in meta.to_dict("records"):
        read = lambda kind: open(os.path.join(root, "data_real", "pairs", f"{m['circuit_id']}_{kind}.qasm"), encoding="utf-8").read()
        bug, ref = read("buggy"), D.parse(read("reference"))
        calls0 = llm.calls
        res = llm_repair(llm, bug, ref, with_example=with_example, device=bool(m.get("device_check", 1)))
        rows.append({"circuit_id": m["circuit_id"], "bugs4q_entry": m["bugs4q_entry"], "arm": "llm:" + spec + (" +example" if with_example else ""),
                     "accepted": int(res["success"]), "attempts": len(res["history"]),
                     "first_attempt_accepted": int(res["history"][0]["status"] == "PASS"),
                     "last_failed_stage": None if res["success"] else res["history"][-1]["failed_stage"],
                     "llm_calls": llm.calls - calls0, "api_errors": len(res["errors"]), "first_error": (res["errors"] or [""])[0], "description": m["description"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", action="store_true", help="give the LLM the expected output on |0...0> as a test-style specification")
    ap.add_argument("--llm", nargs="*", default=None, help="model specs such as gemini:gemini-3.5-flash-lite or ollama:qwen2.5-coder:1.5b")
    a = ap.parse_args()
    if a.llm:
        out = os.path.join(PROJECT_ROOT, "experiments", "results_real_llm.csv")
        old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
        done = set(old.arm) if len(old) else set()
        new = [evaluate_llm(spec, with_example=a.example) for spec in a.llm
               if "llm:" + spec + (" +example" if a.example else "") not in done]
        df = pd.concat([old] + new) if new else old
        df.to_csv(out, index=False)
        print(df.drop(columns="description").to_string(index=False))
        print(df.groupby("arm").accepted.agg(["sum", "count"]).to_string())
        return
    df = evaluate()
    out = os.path.join(PROJECT_ROOT, "experiments", "results_real.csv")
    df.to_csv(out, index=False)
    print(df.drop(columns="description").to_string(index=False))
    print(df.groupby("arm").accepted.agg(["sum", "count"]).rename(columns={"sum": "accepted", "count": "pairs"}).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
