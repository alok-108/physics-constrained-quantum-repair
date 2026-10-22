"""Compare layout strategies with the calibration model on benchmark reference circuits.

    python src/eval_hardware.py --workers 8 --per-family 50 --out experiments/results_hardware.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd


def _one(args):
    meta, root = args
    warnings.filterwarnings("ignore")
    from src import hardware as H
    qasm = open(os.path.join(root, meta["ground_truth_file"]), encoding="utf-8").read()
    row = {"circuit_id": meta["circuit_id"], "circuit_type": meta["circuit_type"], "num_qubits": meta["num_qubits"]}
    for layout in ("identity", "auto"):
        e = H.estimate_qasm(qasm, layout)
        for k in ("native_depth", "native_two_qubit_gates", "estimated_duration_us", "coherence_ratio", "esp"):
            row[f"{layout}_{k}"] = getattr(e, k)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=os.path.join(PROJECT_ROOT, "data", "metadata.csv"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "experiments", "results_hardware.csv"))
    ap.add_argument("--per-family", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(a.metadata), ".."))
    meta = pd.read_csv(a.metadata)
    meta = pd.concat([g.sample(min(a.per_family, len(g)), random_state=0) for _, g in meta.groupby("circuit_type")])
    jobs = [(m, root) for m in meta.to_dict("records")]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            rows = list(pool.map(_one, jobs, chunksize=2))
    else:
        rows = [_one(j) for j in jobs]
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    g = df.groupby("circuit_type")[["identity_esp", "auto_esp", "identity_native_two_qubit_gates", "auto_native_two_qubit_gates",
                                    "identity_coherence_ratio", "auto_coherence_ratio"]].mean().round(3)
    print(f"{len(df)} reference circuits (calibration snapshot of FakeSherbrooke)")
    print(g.to_string())
    print("auto layout has higher ESP in %.1f%% of circuits; median ESP identity %.3f vs auto %.3f" % (
        100 * (df.auto_esp > df.identity_esp).mean(), df.identity_esp.median(), df.auto_esp.median()))


if __name__ == "__main__":
    main()
