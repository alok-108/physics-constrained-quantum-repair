"""Guarded inverse cancellation against the unguarded extended agent (src/analyze_guard.py).

``ours_full_ext_guard`` was designed after reading the 12 failures of ``ours_full_ext`` on the benchmark, so it is a
post-hoc change and its 1000-circuit result is a development-set number, not an independent test. Both arms ran on the
same 1000 circuits; the comparison is paired (exact McNemar).
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.analyze_results import mcnemar_exact, wilson


def main():
    exp = os.path.join(PROJECT_ROOT, "experiments")
    g = pd.read_csv(os.path.join(exp, "results_extra_guard.csv"))
    e = pd.read_csv(os.path.join(exp, "results_extra_ext.csv"))
    e = e[e.system_name == "ours_full_ext"]
    w = pd.DataFrame({"ext": e.set_index("circuit_id").pass_at_3, "guard": g.set_index("circuit_id").pass_at_3}).dropna().astype(int)
    r = mcnemar_exact(w.guard.values, w.ext.values)
    lo, hi = wilson(int(w.guard.sum()), len(w))
    summary = pd.DataFrame([{"n": len(w), "ext_P3": round(100 * w.ext.mean(), 1), "guard_P3": round(100 * w.guard.mean(), 1),
                             "guard_CI_lo": round(100 * lo, 1), "guard_CI_hi": round(100 * hi, 1), "only_guard": r["b10"],
                             "only_ext": r["b01"], "rd": round(r["rd"], 4), "p_exact": r["p_exact"],
                             "mean_queries_ext": round(float(e.oracle_queries.mean())),
                             "mean_queries_guard": round(float(g.oracle_queries.mean()))}])
    m = g.set_index("circuit_id").join(e.set_index("circuit_id")[["pass_at_3"]].rename(columns={"pass_at_3": "ext"}))
    modes = (m.groupby(["bug_type", "fault_mode"]).agg(n=("pass_at_3", "size"), guard=("pass_at_3", "mean"), ext=("ext", "mean"))
             .mul([1, 100, 100]).round(1))
    modes = modes[(modes.guard < 100) | (modes.ext < 100)]
    out = os.path.join(PROJECT_ROOT, "paper", "tables")
    summary.to_csv(os.path.join(out, "guard.csv"), index=False)
    modes.to_csv(os.path.join(out, "guard_modes.csv"))
    print(summary.to_string(index=False)); print(modes.to_string())


if __name__ == "__main__":
    main()
