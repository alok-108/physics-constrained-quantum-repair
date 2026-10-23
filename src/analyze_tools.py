"""LLM-as-controller arms against their ablations (src/analyze_tools.py).

Reads experiments/tools/*.csv (``tools_greedy`` and ``llm_tools_*`` runs) and the plain LLM runs, and writes
paper/tables/tools_rates.csv|tex and tools_contrasts.csv. Every comparison is paired on the same circuits.

The question is not "how good is the model" but "what does the model add to the tools": ``tools_greedy`` uses the same
tools with no model at all. A run that stopped early is reported as partial and left out, as in analyze_llm.py.
"""

from __future__ import annotations

import glob
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.analyze_llm import load_runs
from src.analyze_results import holm, mcnemar_exact, to_latex, wilson


def load_tool_runs(exp):
    runs = {}
    for p in sorted(glob.glob(os.path.join(exp, "tools", "*.csv"))):
        d = pd.read_csv(p)
        if d.empty:
            continue
        key = "no model (greedy)" if d.system_name.iloc[0] == "tools_greedy" else str(d.llm_model.dropna().iloc[0])
        runs[key] = d.reset_index(drop=True)
    return runs


def _row(label, d):
    n, c1, c3 = len(d), int(d.pass_at_1.sum()), int(d.pass_at_3.sum())
    lo, hi = wilson(c3, n)
    return {"arm": label, "n": n, "P@1 (%)": round(100 * c1 / n, 1), "P@3 (%)": round(100 * c3 / n, 1),
            "P@3 95% CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]"}


def _paired(a, b, label):
    both = sorted(set(a.circuit_id) & set(b.circuit_id))
    x = a.set_index("circuit_id").loc[both].pass_at_3.astype(int).values
    y = b.set_index("circuit_id").loc[both].pass_at_3.astype(int).values
    r = mcnemar_exact(x, y)
    return {"contrast": label, "n": len(both), "b10": r["b10"], "b01": r["b01"], "risk_difference": round(r["rd"], 3),
            "ci_lo": round(r["rd_lo"], 3), "ci_hi": round(r["rd_hi"], 3), "p_exact": r["p_exact"]}


def main():
    exp = os.path.join(PROJECT_ROOT, "experiments")
    outdir = os.path.join(PROJECT_ROOT, "paper", "tables")
    tools = load_tool_runs(exp)
    if not tools:
        raise SystemExit("no tool runs in experiments/tools/")
    plain = load_runs(exp)
    greedy = tools.get("no model (greedy)")
    common = set(greedy.circuit_id) if greedy is not None else set.intersection(*[set(d.circuit_id) for d in tools.values()])
    complete = {m: d for m, d in tools.items() if common <= set(d.circuit_id)}
    partial = {m: d for m, d in tools.items() if m not in complete}

    rows, extra, con = [], [], []
    for m, d in complete.items():
        d = d[d.circuit_id.isin(common)]
        rows.append(_row("tools: " + m, d))
        if m != "no model (greedy)":
            e = {"arm": "tools: " + m, "mean tool steps": round(float(d.tool_steps.mean()), 1),
                 "invalid replies / circuit": round(float(d.invalid_replies.mean()), 2),
                 "LLM calls": round(float(d.llm_calls.mean()), 1), "tokens": round(float(d.llm_tokens.mean())),
                 "oracle queries": round(float(d.oracle_queries.mean())),
                 "never submitted (%)": round(100 * float((d.iterations_used == 0).mean()), 1)}
            extra.append(e)
            if m in plain:
                p = plain[m][plain[m].circuit_id.isin(common)]
                rows.append(_row("plain LLM: " + m, p))
                con.append(_paired(d, p, f"{m}: tools vs writing the circuit"))
            if greedy is not None:
                con.append(_paired(d, greedy, f"{m}: tools vs no model"))
    rates = pd.DataFrame(rows)
    ex = pd.DataFrame(extra)
    cdf = pd.DataFrame(con)
    if len(cdf):
        cdf["p_holm"] = holm(cdf.p_exact.tolist())
    rates.to_csv(os.path.join(outdir, "tools_rates.csv"), index=False)
    ex.to_csv(os.path.join(outdir, "tools_usage.csv"), index=False)
    cdf.to_csv(os.path.join(outdir, "tools_contrasts.csv"), index=False)
    to_latex(rates.rename(columns={"P@1 (%)": "P@1 (\\%)", "P@3 (%)": "P@3 (\\%)", "P@3 95% CI": "P@3 95\\% CI"}),
             os.path.join(outdir, "tools_rates.tex"),
             f"Language model as controller of verified tools, on the same {len(common)} circuits. "
             "``no model'' applies the same tools greedily; ``plain LLM'' writes the whole circuit. CI: Wilson 95\\%.",
             "tab:tools", fmt="%.1f")
    print(rates.to_string(index=False))
    print("\nusage"); print(ex.to_string(index=False))
    print("\ncontrasts"); print(cdf.to_string(index=False))
    for m, d in partial.items():
        print(f"\nPARTIAL (excluded): {m} covers {len(d)} of {len(common)} circuits")
    cls = {m: d.groupby("bug_type").pass_at_3.mean() * 100 for m, d in complete.items()}
    print("\nPass@3 by class (%)"); print(pd.DataFrame(cls).round(1).to_string())


if __name__ == "__main__":
    main()
