"""Real-bug pilot (src/eval_real.py): circuits from the Bugs4Q benchmark of real Qiskit bugs.

Bugs4Q (Apache-2.0, https://github.com/Z-928/Bugs4Q) stores each real bug as a buggy and a fixed Python
script written for old Qiskit releases. Both scripts of every entry were run under a compatibility shim
and the circuits they build were dumped; the entries whose buggy and fixed circuits differ in logic on
the same qubits became ``data_real/pairs/*``. Most Bugs4Q entries are library, API or version bugs and
have no circuit-level logic fix (see data_real/README.md), so this is a *pilot of 3 pairs*, not a benchmark.

    python src/eval_real.py
"""

from __future__ import annotations

import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src import dynamic as D


def evaluate(root: str = PROJECT_ROOT, budget: int = 3000) -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(root, "data_real", "metadata.csv"))
    rows = []
    for m in meta.to_dict("records"):
        read = lambda kind: open(os.path.join(root, "data_real", "pairs", f"{m['circuit_id']}_{kind}.qasm"), encoding="utf-8").read()
        bug, ref = read("buggy"), read("reference")
        verdict = D.validate(bug, D.parse(ref))
        for arm in ("dyn_full", "dyn_random"):
            t = time.time()
            res = (D.repair(bug, ref, 3, budget, seed_key=m["circuit_id"]) if arm == "dyn_full"
                   else D.random_repair(bug, ref, 3, budget, seed=1))
            rows.append({"circuit_id": m["circuit_id"], "bugs4q_entry": m["bugs4q_entry"], "arm": arm,
                         "buggy_fails_at": verdict["failed_stage"], "accepted": int(res["success"]),
                         "oracle_queries": res.get("queries", 0), "duration_s": round(time.time() - t, 2),
                         "description": m["description"]})
    return pd.DataFrame(rows)


def main():
    df = evaluate()
    out = os.path.join(PROJECT_ROOT, "experiments", "results_real.csv")
    df.to_csv(out, index=False)
    print(df.drop(columns="description").to_string(index=False))
    print(df.groupby("arm").accepted.agg(["sum", "count"]).rename(columns={"sum": "accepted", "count": "pairs"}).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
