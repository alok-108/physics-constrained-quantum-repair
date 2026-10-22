"""Compare the local-LLM arm with the other arms on the *same* circuits (src/analyze_llm.py).

The LLM run covers a stratified subset (class x family cells), so every comparison is paired on
those circuit ids; nothing here is mixed into the 1000-circuit tables.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.analyze_results import holm, mcnemar_exact, to_latex, wilson

NAMES = {"llm_ollama": "local LLM only (1.5B)", "ours_full": "guided search + oracle", "static_no_oracle": "static repairs, no oracle",
         "unconstrained_diag": "no hardware knowledge", "qiskit_transpile": "Qiskit compiler (O1)", "pyzx_zx_calculus": "PyZX"}


def main():
    exp = os.path.join(PROJECT_ROOT, "experiments")
    llm = pd.read_csv(os.path.join(exp, "results_llm.csv"))
    main_df = pd.read_csv(os.path.join(exp, "results_main.csv"))
    extra = pd.read_csv(os.path.join(exp, "results_extra.csv"))
    ids = set(llm.circuit_id)
    ref = pd.concat([main_df, extra])
    ref = ref[ref.circuit_id.isin(ids)]
    allr = pd.concat([llm, ref])
    outdir = os.path.join(PROJECT_ROOT, "paper", "tables")
    rows = []
    for s in NAMES:
        g = allr[allr.system_name == s]
        if not len(g):
            continue
        n, c1, c3 = len(g), int(g.pass_at_1.sum()), int(g.pass_at_3.sum())
        lo, hi = wilson(c3, n)
        rows.append({"system": NAMES[s], "n": n, "P@1 (%)": round(100 * c1 / n, 1), "P@3 (%)": round(100 * c3 / n, 1),
                     "P@3 95% CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]"})
    rates = pd.DataFrame(rows)
    cls = (allr[allr.system_name.isin(["llm_ollama", "ours_full", "static_no_oracle"])]
           .pivot_table(index="bug_type", columns="system_name", values="pass_at_3") * 100).round(1)
    wide = allr.pivot(index="circuit_id", columns="system_name", values="pass_at_3").fillna(0).astype(int)
    con = []
    for b, why in (("ours_full", "guided search vs LLM"), ("static_no_oracle", "static repairs vs LLM")):
        r = mcnemar_exact(wide[b].values, wide["llm_ollama"].values)
        con.append({"contrast (A vs LLM)": why, "b10": r["b10"], "b01": r["b01"], "risk_difference": round(r["rd"], 3),
                    "ci_lo": round(r["rd_lo"], 3), "ci_hi": round(r["rd_hi"], 3), "p_exact": r["p_exact"]})
    con = pd.DataFrame(con)
    con["p_holm"] = holm(con.p_exact.tolist())
    L = llm
    cost = {"circuits": len(L), "mean LLM calls": round(L.llm_calls.mean(), 2), "mean tokens": round(L.llm_tokens.mean()),
            "mean seconds": round(L.duration_s.mean(), 1), "total hours": round(L.duration_s.sum() / 3600, 2),
            "never produced a parsable circuit (%)": round(100 * (L.strategy == "llm_failed").mean(), 1),
            "final stage of failures": L[L.accepted == 0].final_failed_stage.value_counts().to_dict()}
    rates.to_csv(os.path.join(outdir, "llm_rates.csv"), index=False)
    cls.to_csv(os.path.join(outdir, "llm_by_class.csv"))
    con.to_csv(os.path.join(outdir, "llm_contrasts.csv"), index=False)
    to_latex(rates.rename(columns={"P@1 (%)": "P@1 (\\%)", "P@3 (%)": "P@3 (\\%)", "P@3 95% CI": "P@3 95\\% CI"}),
             os.path.join(outdir, "llm_rates.tex"),
             "Local LLM arm and reference arms on the same " + str(len(L)) + " circuits (stratified by fault class and family), "
             "three attempts. CI: Wilson 95\\%.", "tab:llm", fmt="%.1f")
    print(rates.to_string(index=False)); print("\nPass@3 by class (%)"); print(cls.to_string()); print("\nPaired contrasts"); print(con.to_string(index=False))
    print("\ncost:", cost)


if __name__ == "__main__":
    main()
