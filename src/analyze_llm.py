"""Compare the LLM-only arms with the other arms on the *same* circuits (src/analyze_llm.py).

Each LLM run covers a stratified subset (class x family cells), so every comparison is paired on
that arm's circuit ids; nothing here is mixed into the 1000-circuit tables. Arms present on disk:

    experiments/results_llm.csv          llm_ollama  (local model)
    experiments/results_llm_gemini.csv   llm_gemini  (Gemini API, free tier)

Circuits where the API itself failed (``llm_errors`` > 0) are reported separately, because an
outage is not evidence about the model.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.analyze_results import holm, mcnemar_exact, to_latex, wilson

LLM_FILES = {"llm_ollama": "results_llm.csv", "llm_gemini": "results_llm_gemini.csv"}
NAMES = {"llm_ollama": "local LLM only (1.5B)", "llm_gemini": "Gemini only (free tier)",
         "ours_full": "guided search + oracle", "static_no_oracle": "static repairs, no oracle",
         "unconstrained_diag": "no hardware knowledge", "qiskit_transpile": "Qiskit compiler (O1)", "pyzx_zx_calculus": "PyZX"}
ORDER = ["llm_ollama", "llm_gemini", "ours_full", "static_no_oracle", "unconstrained_diag", "qiskit_transpile", "pyzx_zx_calculus"]


def _wilson_row(g, label):
    n, c1, c3 = len(g), int(g.pass_at_1.sum()), int(g.pass_at_3.sum())
    lo, hi = wilson(c3, n)
    return {"system": label, "n": n, "P@1 (%)": round(100 * c1 / n, 1), "P@3 (%)": round(100 * c3 / n, 1),
            "P@3 95% CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]"}


def cost_of(L):
    out = {"circuits": len(L), "mean LLM calls": round(L.llm_calls.mean(), 2), "mean tokens": round(L.llm_tokens.mean()),
           "mean seconds": round(L.duration_s.mean(), 1), "total hours": round(L.duration_s.sum() / 3600, 2),
           "never produced a parsable circuit (%)": round(100 * (L.strategy == "llm_failed").mean(), 1),
           "final stage of failures": L[L.accepted == 0].final_failed_stage.value_counts().to_dict()}
    if "llm_errors" in L:
        out["circuits with API errors"] = int((L.llm_errors > 0).sum())
    return out


def main():
    exp = os.path.join(PROJECT_ROOT, "experiments")
    arms = {a: pd.read_csv(os.path.join(exp, f)) for a, f in LLM_FILES.items() if os.path.exists(os.path.join(exp, f))}
    if not arms:
        raise SystemExit("no LLM result files found in experiments/")
    base = pd.concat([pd.read_csv(os.path.join(exp, "results_main.csv")), pd.read_csv(os.path.join(exp, "results_extra.csv"))])
    outdir = os.path.join(PROJECT_ROOT, "paper", "tables")

    # ---- one rate table: every LLM arm on its own circuits, references on the union of those circuits
    union = set().union(*[set(d.circuit_id) for d in arms.values()])
    rows, costs = [], {}
    for s in ORDER:
        if s in arms:
            rows.append(_wilson_row(arms[s], NAMES[s]))
            costs[s] = cost_of(arms[s])
        else:
            g = base[(base.system_name == s) & base.circuit_id.isin(union)]
            if len(g):
                rows.append(_wilson_row(g, NAMES[s]))
    rates = pd.DataFrame(rows)

    # ---- paired contrasts per LLM arm, on that arm's own circuits
    con = []
    classes = []
    for a, d in arms.items():
        ref = base[base.circuit_id.isin(set(d.circuit_id))]
        allr = pd.concat([d, ref])
        wide = allr.pivot(index="circuit_id", columns="system_name", values="pass_at_3").fillna(0).astype(int)
        for b, why in (("ours_full", "guided search"), ("static_no_oracle", "static repairs")):
            r = mcnemar_exact(wide[b].values, wide[a].values)
            con.append({"contrast (A vs LLM)": f"{why} vs {NAMES[a]}", "n": len(wide), "b10": r["b10"], "b01": r["b01"],
                        "risk_difference": round(r["rd"], 3), "ci_lo": round(r["rd_lo"], 3), "ci_hi": round(r["rd_hi"], 3),
                        "p_exact": r["p_exact"]})
        classes.append(allr[allr.system_name.isin([a, "ours_full", "static_no_oracle"])]
                       .pivot_table(index="bug_type", columns="system_name", values="pass_at_3") * 100)
    if len(arms) == 2:                                   # head to head on the circuits both arms ran
        both = set.intersection(*[set(d.circuit_id) for d in arms.values()])
        w = pd.concat(arms.values()); w = w[w.circuit_id.isin(both)]
        wide = w.pivot(index="circuit_id", columns="system_name", values="pass_at_3").fillna(0).astype(int)
        r = mcnemar_exact(wide["llm_gemini"].values, wide["llm_ollama"].values)
        con.append({"contrast (A vs LLM)": f"{NAMES['llm_gemini']} vs {NAMES['llm_ollama']}", "n": len(wide), "b10": r["b10"],
                    "b01": r["b01"], "risk_difference": round(r["rd"], 3), "ci_lo": round(r["rd_lo"], 3),
                    "ci_hi": round(r["rd_hi"], 3), "p_exact": r["p_exact"]})
    con = pd.DataFrame(con)
    con["p_holm"] = holm(con.p_exact.tolist())
    cls = pd.concat(classes, axis=1)
    cls = cls.loc[:, ~cls.columns.duplicated()].round(1)
    cls = cls[[c for c in ORDER if c in cls.columns]]

    rates.to_csv(os.path.join(outdir, "llm_rates.csv"), index=False)
    cls.to_csv(os.path.join(outdir, "llm_by_class.csv"))
    con.to_csv(os.path.join(outdir, "llm_contrasts.csv"), index=False)
    to_latex(rates.rename(columns={"P@1 (%)": "P@1 (\\%)", "P@3 (%)": "P@3 (\\%)", "P@3 95% CI": "P@3 95\\% CI"}),
             os.path.join(outdir, "llm_rates.tex"),
             "LLM-only arms and reference arms on the same circuits (stratified by fault class and family; "
             + ", ".join(f"{NAMES[a]}: n={len(d)}" for a, d in arms.items()) + "), three attempts. CI: Wilson 95\\%.",
             "tab:llm", fmt="%.1f")
    print(rates.to_string(index=False)); print("\nPass@3 by class (%)"); print(cls.to_string())
    print("\nPaired contrasts"); print(con.to_string(index=False))
    for a, c in costs.items():
        print(f"\ncost [{a}]:", c)


if __name__ == "__main__":
    main()
