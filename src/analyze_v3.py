"""Tables for the v3 (dynamic-circuit) benchmark (src/analyze_v3.py).

Reads experiments/results_v3.csv; writes paper/tables/v3_*.csv. Statistics as elsewhere: Wilson
intervals, exact McNemar on paired outcomes, Newcombe interval on the paired risk difference, Holm.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.analyze_results import holm, mcnemar_exact, to_latex, wilson

LABEL = {"dyn_full": "guided + feedback", "dyn_reject_only": "guided, bare rejection",
         "dyn_random": "random search", "dyn_noop": "unchanged program"}


def main(path=None, outdir=None):
    path = path or os.path.join(PROJECT_ROOT, "experiments", "results_v3.csv")
    outdir = outdir or os.path.join(PROJECT_ROOT, "paper", "tables")
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(path)
    rows = []
    for s, g in df.groupby("system_name"):
        n = len(g)
        c1, c3 = int(g.pass_at_1.sum()), int(g.pass_at_3.sum())
        lo, hi = wilson(c3, n)
        rows.append({"system": LABEL.get(s, s), "n": n, "Pass@1 (%)": round(100 * c1 / n, 1),
                     "Pass@3 (%)": round(100 * c3 / n, 1), "Pass@3 95% CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]",
                     "mean oracle queries": round(g.oracle_queries.mean(), 1)})
    rates = pd.DataFrame(rows).sort_values("Pass@3 (%)", ascending=False)
    cls = (df.pivot_table(index="bug_class", columns="system_name", values="pass_at_3") * 100).round(1)
    fam = (df.pivot_table(index="family", columns="system_name", values="pass_at_3") * 100).round(1)
    wide = df.pivot(index="circuit_id", columns="system_name", values="pass_at_3").fillna(0).astype(int)
    con = []
    for a, b, why in (("dyn_full", "dyn_random", "guidance"), ("dyn_full", "dyn_reject_only", "counterexample feedback"),
                      ("dyn_full", "dyn_noop", "any repair at all")):
        r = mcnemar_exact(wide[a].values, wide[b].values)
        con.append({"contrast": why, "b10": r["b10"], "b01": r["b01"], "risk_difference": round(r["rd"], 3),
                    "ci_lo": round(r["rd_lo"], 3), "ci_hi": round(r["rd_hi"], 3), "p_exact": r["p_exact"]})
    con = pd.DataFrame(con)
    con["p_holm"] = holm(con.p_exact.tolist())
    noop = df[df.system_name == "dyn_noop"]
    blind = noop.groupby("bug_class").unitary_blind.mean().mul(100).round(1)
    full = df[df.system_name == "dyn_full"].merge(noop[["circuit_id"]].assign(), on="circuit_id")
    blind_ids = set(noop[noop.unitary_blind == 1].circuit_id)
    inb = full[full.circuit_id.isin(blind_ids)]
    gap = df[(df.system_name == "dyn_full") & df.holdout_gap.notna()].holdout_gap
    for name, t in (("rates", rates), ("by_class", cls), ("by_family", fam), ("contrasts", con),
                    ("unitary_blind_by_class", blind)):
        t.to_csv(os.path.join(outdir, f"v3_{name}.csv"), index=name in ("by_class", "by_family", "unitary_blind_by_class"))
    to_latex(rates.rename(columns={"Pass@1 (%)": "P@1 (\\%)", "Pass@3 (%)": "P@3 (\\%)", "Pass@3 95% CI": "P@3 95\\% CI",
                                   "mean oracle queries": "Queries"}), os.path.join(outdir, "v3_rates.tex"),
             "Benchmark v3 (600 dynamic programs), three attempts, oracle budget 3000. CI: Wilson 95\\%.",
             "tab:v3rates", fmt="%.1f", star=True)
    cls_t = cls[["dyn_full", "dyn_reject_only", "dyn_random", "dyn_noop"]].rename(
        columns={"dyn_full": "guided+fb", "dyn_reject_only": "guided", "dyn_random": "random", "dyn_noop": "unchanged"})
    to_latex(cls_t.reset_index().rename(columns={"bug_class": "fault class"}), os.path.join(outdir, "v3_by_class.tex"),
             "Benchmark v3: Pass@3 (\\%) by fault class (100 programs per class).", "tab:v3class", fmt="%.1f")
    hw = os.path.join(PROJECT_ROOT, "experiments", "results_hardware.csv")
    if os.path.exists(hw):
        h = pd.read_csv(hw)
        t = h.groupby("circuit_type").agg(esp_identity=("identity_esp", "median"), esp_auto=("auto_esp", "median"),
                                          cx_identity=("identity_native_two_qubit_gates", "median"),
                                          cx_auto=("auto_native_two_qubit_gates", "median"),
                                          ratio_identity=("identity_coherence_ratio", "median"),
                                          ratio_auto=("auto_coherence_ratio", "median")).round(3)
        t.to_csv(os.path.join(outdir, "hardware.csv"))
        to_latex(t.reset_index().rename(columns={"circuit_type": "family", "esp_identity": "ESP id.", "esp_auto": "ESP auto",
                                                 "cx_identity": "2q id.", "cx_auto": "2q auto",
                                                 "ratio_identity": "T ratio id.", "ratio_auto": "T ratio auto"}),
                 os.path.join(outdir, "hardware.tex"),
                 "Calibration-based estimate on 200 reference circuits (medians): estimated success probability (ESP), "
                 "native two-qubit gates and duration/min(T1,T2), for the identity layout assumed by the benchmark and a "
                 "compiler-chosen layout (FakeSherbrooke snapshot).", "tab:hardware", fmt="%.3g", star=True)
    print(rates.to_string(index=False)); print("\nPass@3 by class (%)"); print(cls.to_string())
    print("\nPass@3 by family (%)"); print(fam.to_string()); print("\nContrasts"); print(con.to_string(index=False))
    print(f"\nfaults invisible to a unitary-equivalence check: {100 * noop.unitary_blind.mean():.1f}% overall")
    print(blind.to_string())
    print(f"guided search repairs {100 * inb.pass_at_3.mean():.1f}% of those unitary-blind faults (n={len(inb)})")
    print(f"mean suite-minus-heldout gap (dyn_full): {gap.mean():.4f}; share with gap > 0.01: {100 * (gap > 0.01).mean():.1f}%")


if __name__ == "__main__":
    main()
