"""Evaluate fault localisation on the benchmark (src/eval_localization.py).

Ground truth comes from the benchmark itself, with no extra annotation:
* same-length faults (semantic, invert_cnot, wrong_qubit, twoqubit_replace): the instructions that
  differ between the faulty circuit and the reference;
* remove_gate: the position where the instruction is missing (parsed from the fault description).
A localisation hits at k if any ground-truth position is among the top-k ranked instructions.
Baseline: a random ranking of the circuit's instructions (exact expected hit rate).

    python src/eval_localization.py --workers 8 --out experiments/results_localization.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from math import comb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import qiskit.qasm3

KS = (1, 3, 5)


def _sig(qc, ins):
    return (ins.operation.name, tuple(qc.find_bit(q).index for q in ins.qubits),
            tuple(round(float(p), 4) for p in ins.operation.params))


def ground_truth(meta, buggy, ref):
    """Set of instruction positions in ``buggy`` that localisation should point to, or None."""
    if meta["fault_mode"] == "remove_gate":
        m = re.search(r"index (\d+)", meta["bug_description"])
        return {int(m.group(1))} if m else None
    if len(buggy.data) != len(ref.data) or buggy.num_qubits != ref.num_qubits:
        return None
    diff = {i for i, (a, b) in enumerate(zip(buggy.data, ref.data)) if _sig(buggy, a) != _sig(ref, b)}
    return diff or None


def random_hit(L, t, k):
    return 1.0 if k >= L else 1.0 - comb(L - t, k) / comb(L, k)


def _one(args):
    meta, root, extended = args
    warnings.filterwarnings("ignore")
    from src import product
    try:
        buggy_txt = open(os.path.join(root, meta["buggy_file"]), encoding="utf-8").read()
        ref_txt = open(os.path.join(root, meta["ground_truth_file"]), encoding="utf-8").read()
        buggy, ref = qiskit.qasm3.loads(buggy_txt), qiskit.qasm3.loads(ref_txt)
    except Exception:
        return None
    truth = ground_truth(meta, buggy, ref)
    if not truth:
        return None
    oracle = product.reference_oracle(ref_txt, key=meta["circuit_id"])
    rows = product.localize(buggy_txt, oracle, top=max(KS), extended=extended)
    ranked = [r["instruction"] for r in rows]
    out = {"circuit_id": meta["circuit_id"], "bug_type": meta["bug_type"], "fault_mode": meta["fault_mode"],
           "circuit_type": meta["circuit_type"], "length": len(buggy.data), "n_truth": len(truth),
           "queries": oracle.queries_used, "n_ranked": len(ranked)}
    for k in KS:
        out[f"hit@{k}"] = int(any(i in truth for i in ranked[:k]))
        out[f"random@{k}"] = random_hit(len(buggy.data) + 1, len(truth), k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=os.path.join(PROJECT_ROOT, "data", "metadata.csv"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "experiments", "results_localization.csv"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--base", action="store_true", help="use the base (non-extended) edit neighbourhood")
    a = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(a.metadata), ".."))
    meta = pd.read_csv(a.metadata)
    meta = meta[meta.bug_type.isin(["structural", "semantic_deviation"])]
    jobs = [(m, root, not a.base) for m in meta.to_dict("records")]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            res = list(pool.map(_one, jobs, chunksize=4))
    else:
        res = [_one(j) for j in jobs]
    df = pd.DataFrame([r for r in res if r])
    df.to_csv(a.out, index=False)
    cols = [f"hit@{k}" for k in KS] + [f"random@{k}" for k in KS]
    print(f"evaluated {len(df)} of {len(jobs)} structural/semantic faults (rest have no unique location)")
    print((df[cols].mean() * 100).round(1).to_string())
    print(df.groupby("fault_mode")[cols].mean().mul(100).round(1).assign(n=df.groupby("fault_mode").size()).to_string())


if __name__ == "__main__":
    main()
