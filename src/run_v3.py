"""Run the v3 (dynamic-circuit) benchmark (src/run_v3.py).

Arms (same acceptance predicate, same K = 3 attempts, same oracle budget):
  dyn_full          guided greedy search + counterexample feedback
  dyn_reject_only   guided greedy search, bare rejection (no counterexample)
  dyn_random        random-mutation hill climbing, same edit grammar, same budget (no guidance)
  dyn_noop          submit the faulty program unchanged (checks that every fault is a real fault)

Also records ``unitary_blind``: whether a unitary-equivalence check (gates only, measurements,
resets and conditionals dropped) would fail to see the fault.

    python src/run_v3.py --workers 8 --out experiments/results_v3.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

ARMS = ["dyn_full", "dyn_reject_only", "dyn_random", "dyn_noop"]


def _row(meta, arm, res, dt, blind):
    hist = res["history"]
    k = lambda n: int(any(h["status"] == "PASS" for h in hist[:n]))
    gap = res.get("gap") or {}
    return {"circuit_id": meta["circuit_id"], "family": meta["family"], "bug_class": meta["bug_class"],
            "n_faults": meta["n_faults"], "fault_kinds": meta["fault_kinds"], "system_name": arm,
            "pass_at_1": k(1), "pass_at_2": k(2), "pass_at_3": k(3), "accepted": int(res["success"]),
            "attempts": len(hist), "oracle_queries": res.get("queries", 0),
            "holdout_gap": gap.get("gap"), "unitary_blind": int(blind),
            "final_failed_stage": None if res["success"] else (hist[-1]["failed_stage"] if hist else None),
            "duration_s": round(dt, 3)}


def _one(args):
    meta, root, budget = args
    warnings.filterwarnings("ignore")
    from src import dynamic as D
    from src.sim import process_fidelity
    buggy = open(os.path.join(root, meta["buggy_file"]), encoding="utf-8").read()
    ref = open(os.path.join(root, meta["reference_file"]), encoding="utf-8").read()
    try:
        blind = process_fidelity(D.unitary_view(D.parse(buggy)), D.unitary_view(D.parse(ref))) > 0.999
    except Exception:
        blind = False
    rows = []
    for arm in ARMS:
        t = time.time()
        if arm == "dyn_noop":
            v = D.validate(buggy, D.parse(ref))
            res = {"success": v["status"] == "PASS", "history": [{"status": v["status"], "failed_stage": v["failed_stage"]}],
                   "queries": 0}
        elif arm == "dyn_random":
            res = D.random_repair(buggy, ref, 3, budget, seed=int(meta["circuit_id"].split("_")[1]))
        else:
            res = D.repair(buggy, ref, 3, budget, feedback=(arm == "dyn_full"), seed_key=meta["circuit_id"])
        rows.append(_row(meta, arm, res, time.time() - t, blind))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=os.path.join(PROJECT_ROOT, "data_v3", "metadata.csv"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "experiments", "results_v3.csv"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(a.metadata), ".."))
    meta = pd.read_csv(a.metadata)
    if a.limit:
        per = max(1, a.limit // meta.bug_class.nunique())
        meta = pd.concat([g.head(per) for _, g in meta.groupby("bug_class")])
    jobs = [(m, root, a.budget) for m in meta.to_dict("records")]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            res = [r for rows in pool.map(_one, jobs, chunksize=2) for r in rows]
    else:
        res = [r for j in jobs for r in _one(j)]
    df = pd.DataFrame(res)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} rows in {time.time() - t0:.0f}s -> {a.out}")
    print((df.groupby("system_name")[["pass_at_1", "pass_at_3", "oracle_queries"]].mean()
           .assign(pass_at_1=lambda x: x.pass_at_1 * 100, pass_at_3=lambda x: x.pass_at_3 * 100).round(1)).to_string())
    print((df.pivot_table(index="bug_class", columns="system_name", values="pass_at_3") * 100).round(1).to_string())
    print("unitary-blind faults: %.1f%%" % (100 * df[df.system_name == "dyn_noop"].unitary_blind.mean()))


if __name__ == "__main__":
    main()
