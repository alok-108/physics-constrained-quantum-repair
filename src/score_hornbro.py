"""Judge the circuits HornBro returned with this repository's acceptance predicate (src/score_hornbro.py).

    python src/score_hornbro.py --dir hornbro_out

Reads ``<circuit_id>.json`` / ``<circuit_id>.qasm`` written by src/hornbro_adapter.py, scores each circuit with the same
validator and the same pass criterion as every other system, and writes experiments/results_hornbro.csv. Circuits for which
HornBro returned nothing (unparsable input, crash) count as failures; the reason is kept in ``outcome``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import qiskit.qasm3


def functionally_equivalent(candidate_qasm: str, reference_qasm: str, threshold: float = 0.99) -> bool:
    """Unitary equivalence only (process fidelity), ignoring device adjacency, depth and minimality. HornBro has no device
    model, so the strict shared predicate alone would judge it on a criterion it was never built for; both are reported."""
    from src.sim import process_fidelity
    try:
        return bool(process_fidelity(qiskit.qasm3.loads(candidate_qasm), qiskit.qasm3.loads(reference_qasm)) >= threshold)
    except Exception:
        return False


def score(directory: str, metadata: str, validator=None) -> pd.DataFrame:
    from agents.validator import ValidatorAgent
    validator = validator or ValidatorAgent()
    meta = pd.read_csv(metadata).set_index("circuit_id")
    root = os.path.abspath(os.path.join(os.path.dirname(metadata), ".."))
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "circuit_*.json"))):
        rec = json.load(open(path))
        cid = rec["circuit_id"]
        m = meta.loc[cid]
        accepted, stage, functional = 0, rec["outcome"], 0
        if rec["outcome"] == "crashed" and str(rec.get("error", "")).startswith(("ModuleNotFoundError", "ImportError")):
            rec["outcome"] = stage = "environment_error"            # a missing package says nothing about HornBro
        qpath = os.path.join(directory, cid + ".qasm")
        if rec["outcome"] == "returned_circuit" and os.path.exists(qpath):
            ref = open(os.path.join(root, m["ground_truth_file"]), encoding="utf-8").read()
            v = validator.validate(open(qpath, encoding="utf-8").read(), ref, gt_key=cid)
            accepted, stage = int(v["status"] == "PASS"), (None if v["status"] == "PASS" else v["failed_stage"])
            functional = int(functionally_equivalent(open(qpath, encoding="utf-8").read(), ref))
        rows.append({"circuit_id": cid, "circuit_type": m["circuit_type"], "num_qubits": int(m["num_qubits"]),
                     "bug_type": m["bug_type"], "system_name": "hornbro", "pass_at_3": accepted, "functionally_equivalent": functional,
                     "outcome": rec["outcome"], "failed_stage": stage,
                     "hornbro_says_success": rec.get("hornbro_says_success"), "seconds": rec.get("seconds"),
                     "solver_timeout_scale": rec.get("solver_timeout_scale", 1.0)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hornbro_out")
    ap.add_argument("--metadata", default=os.path.join(PROJECT_ROOT, "data", "metadata.csv"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "experiments", "results_hornbro.csv"))
    a = ap.parse_args()
    df = score(a.dir, a.metadata)
    df.to_csv(a.out, index=False)
    env = int((df.outcome == "environment_error").sum())
    if env:
        print(f"WARNING: {env} of {len(df)} circuits hit an environment error (missing package), not a HornBro failure. "
              "Fix the environment and run the adapter again before reading any number below.")
    print(df.groupby("bug_type").pass_at_3.agg(["sum", "count"]).to_string())
    print(f"HornBro accepted {int(df.pass_at_3.sum())} of {len(df)} (full shared predicate, one attempt)")
    print(f"HornBro returned a functionally equivalent circuit (unitary only, no device/minimality) for "
          f"{int(df.functionally_equivalent.sum())} of {len(df)}")
    print(df.outcome.value_counts().to_string())
    agree = df[df.hornbro_says_success.notna()]
    if len(agree):
        print(f"HornBro's own success flag agrees with the shared predicate on "
              f"{int(((agree.hornbro_says_success.astype(bool)) == (agree.pass_at_3 == 1)).sum())} of {len(agree)}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
