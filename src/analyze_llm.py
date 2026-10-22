"""Compare the LLM-only runs with the other arms on the *same* circuits (src/analyze_llm.py).

Every LLM run is one model (``llm_model`` column) under the identical protocol: the model is the
only repair engine, three attempts, validator feedback by stage name, no oracle, no search.

Inputs (all in ``experiments/``)
    results_llm.csv          the first local run (qwen2.5-coder:1.5b, 60 circuits)
    llm_runs/*.csv           any further run written by ``run_experiments.py --systems llm_ollama|llm_gemini``
                             (for example from the Colab notebook), one model per file

The 60-circuit subset is the stratified set of the first run (class x family cells). Runs on more
circuits are restricted to it for the common table and used in full for the large-sample table, so
nothing here is mixed into the 1000-circuit tables of the repair agent.

Circuits where the API itself failed (``llm_errors`` > 0) are reported separately: an outage is
not evidence about the model.
"""

from __future__ import annotations

import glob
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.analyze_results import holm, mcnemar_exact, to_latex, wilson

NAMES = {"ours_full": "guided search + oracle", "static_no_oracle": "static repairs, no oracle",
         "unconstrained_diag": "no hardware knowledge", "qiskit_transpile": "Qiskit compiler (O1)", "pyzx_zx_calculus": "PyZX"}
REFS = list(NAMES)
LEGACY_MODEL = {"results_llm.csv": "qwen2.5-coder:1.5b"}


def load_runs(exp):
    """{model: DataFrame}; one entry per model, later files replacing nothing (duplicates are an error)."""
    files = [os.path.join(exp, f) for f in LEGACY_MODEL] + sorted(glob.glob(os.path.join(exp, "llm_runs", "*.csv")))
    runs = {}
    for p in files:
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if "llm_model" not in d or d.llm_model.isna().all():
            d["llm_model"] = LEGACY_MODEL.get(os.path.basename(p), os.path.splitext(os.path.basename(p))[0])
        d["llm_model"] = d.llm_model.fillna(LEGACY_MODEL.get(os.path.basename(p), "unknown"))
        for m, g in d.groupby("llm_model"):
            if m in runs:
                raise SystemExit(f"model {m} appears in two result files")
            runs[m] = g.reset_index(drop=True)
    return runs


def _row(g, label):
    n, c1, c3 = len(g), int(g.pass_at_1.sum()), int(g.pass_at_3.sum())
    lo, hi = wilson(c3, n)
    return {"system": label, "n": n, "P@1 (%)": round(100 * c1 / n, 1), "P@3 (%)": round(100 * c3 / n, 1),
            "P@3 95% CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]"}


def cost_of(L):
    out = {"circuits": len(L), "mean LLM calls": round(float(L.llm_calls.mean()), 2),
           "mean tokens": round(float(L.llm_tokens.mean())), "mean seconds": round(float(L.duration_s.mean()), 1),
           "total hours": round(float(L.duration_s.sum()) / 3600, 2),
           "never produced a parsable circuit (%)": round(100 * float((L.strategy == "llm_failed").mean()), 1),
           "final stage of failures": {k: int(v) for k, v in L[L.accepted == 0].final_failed_stage.value_counts().items()}}
    if "llm_errors" in L:
        out["circuits with API errors"] = int((L.llm_errors > 0).sum())
    return out


def _contrasts(runs, base, label_of):
    con = []
    for m, d in runs.items():
        ref = base[base.circuit_id.isin(set(d.circuit_id))]
        wide = pd.concat([d.assign(system_name="llm"), ref]).pivot(
            index="circuit_id", columns="system_name", values="pass_at_3").fillna(0).astype(int)
        for b, why in (("ours_full", "guided search"), ("static_no_oracle", "static repairs")):
            r = mcnemar_exact(wide[b].values, wide["llm"].values)
            con.append({"contrast (A vs LLM)": f"{why} vs {label_of(m)}", "n": len(wide), "b10": r["b10"], "b01": r["b01"],
                        "risk_difference": round(r["rd"], 3), "ci_lo": round(r["rd_lo"], 3), "ci_hi": round(r["rd_hi"], 3),
                        "p_exact": r["p_exact"]})
    return con


def main():
    exp = os.path.join(PROJECT_ROOT, "experiments")
    runs = load_runs(exp)
    if not runs:
        raise SystemExit("no LLM result files found in experiments/")
    base = pd.concat([pd.read_csv(os.path.join(exp, "results_main.csv")), pd.read_csv(os.path.join(exp, "results_extra.csv"))])
    outdir = os.path.join(PROJECT_ROOT, "paper", "tables")
    label_of = lambda m: m
    common = set(runs["qwen2.5-coder:1.5b"].circuit_id) if "qwen2.5-coder:1.5b" in runs else set.intersection(
        *[set(d.circuit_id) for d in runs.values()])

    # ---- common table: every model on the shared subset (a larger run is restricted to it)
    sub = {m: d[d.circuit_id.isin(common)] for m, d in runs.items()}
    sub = {m: d for m, d in sub.items() if len(d) == len(common)}
    rows = [_row(d, label_of(m)) for m, d in sub.items()]
    rows += [_row(base[(base.system_name == s) & base.circuit_id.isin(common)], NAMES[s]) for s in REFS]
    rates = pd.DataFrame(rows)

    # ---- large-sample table: the largest run, in full, against the references on the same circuits
    big_m = max(runs, key=lambda m: len(runs[m]))
    big = runs[big_m]
    full_rows = [_row(big, label_of(big_m))] + [_row(base[(base.system_name == s) & base.circuit_id.isin(set(big.circuit_id))],
                                                      NAMES[s]) for s in REFS]
    full = pd.DataFrame(full_rows) if len(big) > len(common) else None

    # ---- paired contrasts, each run on its own circuits
    con = _contrasts(runs, base, label_of)
    if len(runs) > 1:                                    # every pair of models on the circuits both ran
        ms = list(runs)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = runs[ms[i]], runs[ms[j]]
                both = set(a.circuit_id) & set(b.circuit_id)
                wide = pd.DataFrame({"a": a[a.circuit_id.isin(both)].set_index("circuit_id").pass_at_3,
                                     "b": b[b.circuit_id.isin(both)].set_index("circuit_id").pass_at_3}).dropna().astype(int)
                if len(wide) < 10:
                    continue
                r = mcnemar_exact(wide["b"].values, wide["a"].values)
                con.append({"contrast (A vs LLM)": f"{ms[j]} vs {ms[i]}", "n": len(wide), "b10": r["b10"], "b01": r["b01"],
                            "risk_difference": round(r["rd"], 3), "ci_lo": round(r["rd_lo"], 3), "ci_hi": round(r["rd_hi"], 3),
                            "p_exact": r["p_exact"]})
    con = pd.DataFrame(con)
    con["p_holm"] = holm(con.p_exact.tolist())

    # ---- by fault class (common subset)
    cl = {label_of(m): d.groupby("bug_type").pass_at_3.mean() * 100 for m, d in sub.items()}
    for s in ("ours_full", "static_no_oracle"):
        cl[NAMES[s]] = base[(base.system_name == s) & base.circuit_id.isin(common)].groupby("bug_type").pass_at_3.mean() * 100
    cls = pd.DataFrame(cl).round(1)

    rates.to_csv(os.path.join(outdir, "llm_rates.csv"), index=False)
    cls.to_csv(os.path.join(outdir, "llm_by_class.csv"))
    con.to_csv(os.path.join(outdir, "llm_contrasts.csv"), index=False)
    if full is not None:
        full.to_csv(os.path.join(outdir, "llm_large.csv"), index=False)
    ren = {"P@1 (%)": "P@1 (\\%)", "P@3 (%)": "P@3 (\\%)", "P@3 95% CI": "P@3 95\\% CI"}
    to_latex(rates.rename(columns=ren), os.path.join(outdir, "llm_rates.tex"),
             f"LLM-only runs and reference arms on the same {len(common)} circuits (stratified by fault class and family), "
             "three attempts, no oracle or search for the LLMs. CI: Wilson 95\\%.", "tab:llm", fmt="%.1f")
    if full is not None:
        to_latex(full.rename(columns=ren), os.path.join(outdir, "llm_large.tex"),
                 f"Large-sample LLM run ({big_m}, {len(big)} circuits) and reference arms on the same circuits. "
                 "CI: Wilson 95\\%.", "tab:llmlarge", fmt="%.1f")
    print(rates.to_string(index=False))
    if full is not None:
        print(f"\nLarge run ({big_m})"); print(full.to_string(index=False))
    print("\nPass@3 by class (%), common subset"); print(cls.to_string())
    print("\nPaired contrasts"); print(con.to_string(index=False))
    for m, d in runs.items():
        print(f"\ncost [{m}]:", cost_of(d))


if __name__ == "__main__":
    main()
