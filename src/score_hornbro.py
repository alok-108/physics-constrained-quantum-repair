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
        accepted, stage = 0, rec["outcome"]
        qpath = os.path.join(directory, cid + ".qasm")
        if rec["outcome"] == "returned_circuit" and os.path.exists(qpath):
            ref = open(os.path.join(root, m["ground_truth_file"]), encoding="utf-8").read()
            v = validator.validate(open(qpath, encoding="utf-8").read(), ref, gt_key=cid)
            accepted, stage = int(v["status"] == "PASS"), (None if v["status"] == "PASS" else v["failed_stage"])
        rows.append({"circuit_id": cid, "circuit_type": m["circuit_type"], "num_qubits": int(m["num_qubits"]),
                     "bug_type": m["bug_type"], "system_name": "hornbro", "pass_at_3": accepted,
                     "outcome": rec["outcome"], "failed_stage": stage,
                     "hornbro_says_success": rec.get("hornbro_says_success"), "seconds": rec.get("seconds")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hornbro_out")
    ap.add_argument("--metadata", default=os.path.join(PROJECT_ROOT, "data", "metadata.csv"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "experiments", "results_hornbro.csv"))
    a = ap.parse_args()
    df = score(a.dir, a.metadata)
    df.to_csv(a.out, index=False)
    print(df.groupby("bug_type").pass_at_3.agg(["sum", "count"]).to_string())
    print(f"HornBro accepted {int(df.pass_at_3.sum())} of {len(df)} (shared predicate, one attempt)")
    print(df.outcome.value_counts().to_string())
    agree = df[df.hornbro_says_success.notna()]
    if len(agree):
        print(f"HornBro's own success flag agrees with the shared predicate on "
              f"{int(((agree.hornbro_says_success.astype(bool)) == (agree.pass_at_3 == 1)).sum())} of {len(agree)}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
