"""Statistical analysis, tables and figures (src/analyze_results.py).

Inputs: ``experiments/results_main.csv`` (main configuration), optionally
``results_weak_oracle.csv`` and ``results_budget_*.csv``.
Outputs: ``paper/tables/*.csv|tex`` and ``paper/figures/*.pdf|png``.

Statistics
----------
* Outcomes are paired binary variables over a common circuit set, so every
  comparison is an exact McNemar test on discordant pairs.
* Effect size: paired risk difference with a Newcombe (method 10) score
  interval built from Wilson intervals; rates carry Wilson 95% intervals.
* Multiplicity: Holm correction over the comparison family printed in each
  table caption.
* Resource overhead is computed over accepted repairs only.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")
TAB_DIR = os.path.join(PROJECT_ROOT, "paper", "tables")

LABELS = {
    "ours_full": "Physics + diagnostic feedback",
    "ours_reject_only": "Physics, bare rejection",
    "unconstrained_diag": "No physics, diagnostic feedback",
    "unconstrained_reject_only": "No physics, bare rejection",
    "static_no_oracle": "Static repair (no oracle)",
    "pyzx_zx_calculus": "PyZX",
    "ours_full_qiskit_router": "Physics + diagnostic, Qiskit router",
    "qiskit_transpile": "Qiskit transpiler only",
    "qiskit_transpile_l0": "Qiskit transpiler O0",
    "qiskit_transpile_l2": "Qiskit transpiler O2",
    "qiskit_transpile_l3": "Qiskit transpiler O3",
}
ORDER = list(LABELS)

# Pre-specified contrasts (system_a, system_b, purpose)
CONTRASTS = [
    ("ours_full", "unconstrained_diag", "physics context (feedback on)"),
    ("ours_reject_only", "unconstrained_reject_only", "physics context (feedback off)"),
    ("ours_full", "ours_reject_only", "diagnostic feedback (physics on)"),
    ("unconstrained_diag", "unconstrained_reject_only", "diagnostic feedback (physics off)"),
    ("ours_full", "static_no_oracle", "behavioural oracle"),
    ("ours_full", "pyzx_zx_calculus", "transformation baseline"),
    ("ours_full", "qiskit_transpile", "compiler baseline"),
    ("ours_full", "ours_full_qiskit_router", "router choice (own vs Qiskit)"),
]


# --------------------------------------------------------------------------- #
def wilson(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe_paired(n11: int, n10: int, n01: int, n00: int) -> Tuple[float, float, float]:
    """Paired risk difference p1 - p2 with Newcombe's method-10 interval."""
    n = n11 + n10 + n01 + n00
    p1, p2 = (n11 + n10) / n, (n11 + n01) / n
    l1, u1 = wilson(n11 + n10, n)
    l2, u2 = wilson(n11 + n01, n)
    denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = (n11 * n00 - n10 * n01) / denom if denom > 0 else 0.0
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2))
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2))
    return d, float(lo), float(hi)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    n11 = int(((a == 1) & (b == 1)).sum()); n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum()); n00 = int(((a == 0) & (b == 0)).sum())
    disc = n10 + n01
    p = 1.0 if disc == 0 else min(1.0, 2.0 * stats.binom.cdf(min(n10, n01), disc, 0.5))
    d, lo, hi = newcombe_paired(n11, n10, n01, n00)
    return {"n": len(a), "b10": n10, "b01": n01, "p_exact": p, "rd": d, "rd_lo": lo, "rd_hi": hi}


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = np.empty(m), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


# --------------------------------------------------------------------------- #
def paired_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    w = df.pivot(index="circuit_id", columns="system_name", values=metric)
    return w.fillna(0).astype(int)


def rate_table(df: pd.DataFrame, budget: int = 3) -> pd.DataFrame:
    rows = []
    for s in [x for x in ORDER if x in df.system_name.unique()]:
        g = df[df.system_name == s]
        rec = {"System": LABELS[s], "n": len(g)}
        for k in range(1, budget + 1):
            c = int(g[f"pass_at_{k}"].sum())
            lo, hi = wilson(c, len(g))
            rec[f"Pass@{k} (%)"] = round(100 * c / len(g), 1)
            if k in (1, budget):
                rec[f"Pass@{k} 95% CI"] = f"[{100 * lo:.1f}, {100 * hi:.1f}]"
        pv = int(g.physics_valid_final.fillna(0).sum())
        rec["Physically valid (%)"] = round(100 * pv / len(g), 1)
        rec["Mean attempts"] = round(g.iterations_used.mean(), 2)
        rec["Mean oracle queries"] = round(g.oracle_queries.mean(), 1)
        acc = g[(g.accepted == 1) & g.overhead_vs_ref_pct.notna()]
        rec["Gate overhead vs ref (%), accepted"] = round(acc.overhead_vs_ref_pct.mean(), 1) if len(acc) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def contrast_table(df: pd.DataFrame, metrics=("pass_at_1", "pass_at_3")) -> pd.DataFrame:
    present = set(df.system_name.unique())
    rows = []
    for a, b, purpose in CONTRASTS:
        if a not in present or b not in present:
            continue
        for m in metrics:
            w = paired_wide(df, m)
            r = mcnemar_exact(w[a].values, w[b].values)
            rows.append({"contrast": purpose, "A": LABELS[a], "B": LABELS[b],
                         "metric": m.replace("pass_at_", "Pass@"), **r})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_exact"].tolist())
    out.attrs["family_size"] = len(out)
    return out


def class_table(df: pd.DataFrame, budget: int = 3) -> pd.DataFrame:
    p = df.pivot_table(index="bug_type", columns="system_name", values=f"pass_at_{budget}", aggfunc="mean") * 100
    return p[[s for s in ORDER if s in p.columns]].round(1).rename(columns=LABELS)


def mode_table(df: pd.DataFrame, budget: int = 3) -> pd.DataFrame:
    sub = df[df.system_name.isin(["ours_full", "unconstrained_reject_only", "static_no_oracle"])]
    p = sub.pivot_table(index=["bug_type", "fault_mode", "mode_family"], columns="system_name",
                        values=f"pass_at_{budget}", aggfunc="mean") * 100
    n = sub[sub.system_name == "ours_full"].groupby(["bug_type", "fault_mode", "mode_family"]).size().rename("n")
    out = p.join(n).round(1).reset_index()
    return out.rename(columns=LABELS)


def stage_table(df: pd.DataFrame) -> pd.DataFrame:
    un = df[df.accepted == 0]
    if not len(un):
        return pd.DataFrame()
    t = pd.crosstab(un.final_failed_stage.fillna("none"), un.system_name)
    return t[[s for s in ORDER if s in t.columns]].rename(columns=LABELS)


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in [x for x in ORDER if x in df.system_name.unique() and x not in ("pyzx_zx_calculus", "static_no_oracle")]:
        g = df[df.system_name == s]
        succ = g[g.accepted == 1]
        searched = g[g.oracle_queries > 0]
        gap = g.overfit_gap.dropna()
        rows.append({
            "System": LABELS[s],
            "Mean queries (all)": round(g.oracle_queries.mean(), 1),
            "Median queries (searches)": float(searched.oracle_queries.median()) if len(searched) else np.nan,
            "p95 queries": float(g.oracle_queries.quantile(0.95)),
            "Counterexamples used": round(g.counterexamples.mean(), 3),
            "Mean suite-minus-process gap": round(gap.mean(), 4) if len(gap) else np.nan,
            "Gap > 0.01 (%)": round(100 * (gap > 0.01).mean(), 1) if len(gap) else np.nan,
            "Accepted with gap > 0.01 (%)": round(100 * (succ.overfit_gap.dropna() > 0.01).mean(), 1) if len(succ) else np.nan,
            "Mean wall time (s)": round(g.duration_s.mean(), 2),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def _tex_escape(text: str) -> str:
    out = str(text)
    for ch, rep in (("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        out = out.replace(ch, rep)
    return out


def _tex_cell(value, fmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if value != 0 and abs(value) < 1e-3:
            mant, exp = f"{value:.1e}".split("e")
            return "$" + mant + "\\times 10^{" + str(int(exp)) + "}$"
        return fmt % value
    return _tex_escape(value)


def to_latex(df: pd.DataFrame, path: str, caption: str, label: str, fmt: str = "%.3g",
             star: bool = False, size: str = "footnotesize") -> None:
    """Minimal booktabs writer (no jinja2 dependency). Caption and label are written verbatim."""
    env = "table*" if star else "table"
    cols = [str(c) if "\\%" in str(c) else _tex_escape(c) for c in df.columns]
    spec = "l" * min(1, len(cols)) + "r" * max(0, len(cols) - 1)
    lines = ["\\begin{" + env + "}[t]", "\\centering", "\\caption{" + caption + "}", "\\label{" + label + "}",
             "\\" + size, "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{" + spec + "}", "\\toprule",
             " & ".join(cols) + " \\\\", "\\midrule"]
    def column_fmt(name: str) -> str:
        n = str(name)
        if "%" in n:
            return "%.1f"
        if "Attempts" in n or "Counterex" in n:
            return "%.2f"
        if "Queries" in n or "queries" in n:
            return "%.0f"
        return fmt

    col_fmts = [column_fmt(c) for c in df.columns]
    for row in df.itertuples(index=False):          # itertuples keeps per-column dtypes (ints stay ints)
        lines.append(" & ".join(_tex_cell(v, f) for v, f in zip(row, col_fmts)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{" + env + "}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save(fig, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


PALETTE = ["#2a6f97", "#61a5c2", "#c97b2a", "#e5b982", "#7a7a7a", "#b5b5b5", "#3b8b5a", "#8f5aa6", "#b07a9c", "#6c9e4f", "#d1a33f"]


def fig_passk(df: pd.DataFrame, budget: int = 3) -> None:
    systems = [s for s in ORDER if s in df.system_name.unique()]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.8 / len(systems)
    for i, s in enumerate(systems):
        g = df[df.system_name == s]
        m, e_lo, e_hi = [], [], []
        for k in range(1, budget + 1):
            c = int(g[f"pass_at_{k}"].sum()); n = len(g); lo, hi = wilson(c, n)
            m.append(100 * c / n); e_lo.append(100 * c / n - 100 * lo); e_hi.append(100 * hi - 100 * c / n)
        ax.bar(np.arange(budget) + i * w, m, w, yerr=[e_lo, e_hi], capsize=2, color=PALETTE[i],
               label=LABELS[s], linewidth=0)
    ax.set_xticks(np.arange(budget) + 0.4 - w / 2)
    ax.set_xticklabels([f"Pass@{k}" for k in range(1, budget + 1)])
    ax.set_ylabel("Accepted circuits (%)"); ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    save(fig, "pass_at_k")


def fig_class_heatmap(df: pd.DataFrame, budget: int = 3) -> None:
    t = class_table(df, budget)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.imshow(t.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(t.shape[1])); ax.set_xticklabels(t.columns, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(t.shape[0])); ax.set_yticklabels(t.index, fontsize=8)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, f"{t.values[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if t.values[i, j] > 60 else "black")
    ax.set_title(f"Pass@{budget} (%) by injected fault class", fontsize=9)
    save(fig, "class_heatmap")


def fig_budget(sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for i, (s, g) in enumerate(sweep.groupby("system_name")):
        r = g.groupby("budget").pass_at_3.mean() * 100
        ax.plot(r.index, r.values, marker="o", ms=3, label=LABELS.get(s, s), color=PALETTE[ORDER.index(s)])
    ax.set_xscale("log"); ax.set_xlabel("Oracle query budget per repair (log scale)")
    ax.set_ylabel("Pass@3 (%)"); ax.set_ylim(0, 105)
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    save(fig, "budget_curve")


def fig_queries_cdf(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for i, s in enumerate(["ours_full", "unconstrained_reject_only"]):
        g = df[(df.system_name == s) & (df.accepted == 1) & (df.oracle_queries > 0)]
        if not len(g):
            continue
        x = np.sort(g.oracle_queries.values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=LABELS[s], color=PALETTE[ORDER.index(s)])
    ax.set_xscale("log"); ax.set_xlabel("Oracle queries used by accepted searches")
    ax.set_ylabel("Cumulative fraction"); ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    save(fig, "queries_cdf")


def fig_gap(df: pd.DataFrame) -> None:
    g = df[(df.system_name == "ours_full") & df.overfit_gap.notna() & (df.oracle_queries > 0)]
    if not len(g):
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.hist(np.clip(g.overfit_gap.values, -0.05, 0.5), bins=40, color=PALETTE[0])
    ax.set_xlabel("Suite agreement minus exact process fidelity (final candidate)")
    ax.set_ylabel("Searched repairs"); ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "overfit_gap")


# --------------------------------------------------------------------------- #
def load_sweep(pattern: str) -> pd.DataFrame:
    parts = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"results_budget_(\d+)\.csv$", path)
        if m:
            d = pd.read_csv(path)
            d["budget"] = int(m.group(1))
            parts.append(d)
    return pd.concat(parts) if parts else pd.DataFrame()


SHORT = {
    "ours_full": "P+D", "ours_reject_only": "P+R", "unconstrained_diag": "U+D",
    "unconstrained_reject_only": "U+R", "static_no_oracle": "Static", "pyzx_zx_calculus": "PyZX",
    "ours_full_qiskit_router": "P+D/Q", "qiskit_transpile": "Qiskit O1",
    "qiskit_transpile_l0": "Qiskit O0", "qiskit_transpile_l2": "Qiskit O2", "qiskit_transpile_l3": "Qiskit O3",
}
SHORT_LEGEND = ("P/U = physics context on/off; D/R = diagnostic feedback / bare rejection; "
                "Static = no oracle; P+D/Q = P+D with Qiskit routing; Qiskit On = transpiler only at optimisation level n")


def shorten(df: pd.DataFrame) -> pd.DataFrame:
    """Rename system columns from long labels to the short codes used in paper tables."""
    inv = {LABELS[k]: SHORT[k] for k in LABELS}
    return df.rename(columns=lambda c: inv.get(c, c))


def write(df: pd.DataFrame, name: str, caption: str, label: str, fmt: str = "%.3g", index: bool = False,
          tex_df: pd.DataFrame = None, star: bool = False) -> None:
    """Full table to CSV; (optionally narrower) table to LaTeX."""
    os.makedirs(TAB_DIR, exist_ok=True)
    df.to_csv(os.path.join(TAB_DIR, f"{name}.csv"), index=index)
    t = tex_df if tex_df is not None else df
    to_latex(t.reset_index() if index else t, os.path.join(TAB_DIR, f"{name}.tex"), caption, label, fmt, star)


def compact_rates(rt: pd.DataFrame) -> pd.DataFrame:
    inv = {LABELS[k]: SHORT[k] for k in LABELS}
    out = rt[["System", "Pass@1 (%)", "Pass@3 (%)", "Pass@3 95% CI", "Physically valid (%)",
              "Mean attempts", "Mean oracle queries"]].copy()
    out["System"] = out["System"].map(lambda s: inv.get(s, s))
    return out.rename(columns={"Pass@1 (%)": "P@1 (\\%)", "Pass@3 (%)": "P@3 (\\%)",
                               "Pass@3 95% CI": "P@3 95\\% CI", "Physically valid (%)": "Phys.\\ valid (\\%)",
                               "Mean attempts": "Attempts", "Mean oracle queries": "Queries"})


def compact_cost(co: pd.DataFrame) -> pd.DataFrame:
    inv = {LABELS[k]: SHORT[k] for k in LABELS}
    out = co[["System", "Mean queries (all)", "p95 queries", "Counterexamples used",
              "Gap > 0.01 (%)", "Mean wall time (s)"]].copy()
    out["System"] = out["System"].map(lambda s: inv.get(s, s))
    return out.rename(columns={"Mean queries (all)": "Mean queries", "Counterexamples used": "Counterex.",
                               "Gap > 0.01 (%)": "Gap$>$0.01 (\\%)", "Mean wall time (s)": "Time (s)"})


def contrast_view(ct: pd.DataFrame) -> pd.DataFrame:
    v = ct.copy()
    v["RD [95\\% CI]"] = v.apply(lambda r: f"{r.rd:+.3f} [{r.rd_lo:+.3f}, {r.rd_hi:+.3f}]", axis=1)
    return v[["contrast", "metric", "b10", "b01", "RD [95\\% CI]", "p_holm"]]


def main(results: str, weak: str, sweep_glob: str, budget: int = 3) -> None:
    df = pd.read_csv(results)
    for extra in sorted(glob.glob(os.path.join(os.path.dirname(results), "results_extra*.csv"))):
        df = pd.concat([df, pd.read_csv(extra)], ignore_index=True)   # compiler / router arms
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

    rt = rate_table(df, budget)
    ct = contrast_table(df)
    cl = class_table(df, budget)
    mo = mode_table(df, budget)
    st = stage_table(df)
    co = cost_table(df)

    print("\n=== Rates (n=%d circuits) ===" % df.circuit_id.nunique()); print(rt.to_string(index=False))
    print("\n=== Pre-specified contrasts (Holm family m=%d) ===" % ct.attrs["family_size"])
    print(ct[["contrast", "metric", "b10", "b01", "rd", "rd_lo", "rd_hi", "p_exact", "p_holm"]].to_string(index=False))
    print("\n=== Pass@%d by class ===" % budget); print(cl.to_string())
    print("\n=== Rejection stage among unresolved ==="); print(st.to_string())
    print("\n=== Oracle cost / overfitting ==="); print(co.to_string(index=False))

    write(rt, "rates", "Acceptance under the shared predicate, $n=1000$ circuits, attempt budget $K=3$. "
          "CI: Wilson 95\\%. " + SHORT_LEGEND + ".", "tab:rates", tex_df=compact_rates(rt), star=True)
    write(contrast_view(ct), "contrasts", "Pre-specified paired contrasts (exact McNemar; Newcombe interval "
          "on the paired risk difference; Holm correction over the "
          f"{ct.attrs['family_size']} rows shown).", "tab:contrasts", star=True)
    write(cl, "class_breakdown", f"Pass@{budget} (\\%) by injected fault class (200 circuits per class). "
          + SHORT_LEGEND + ".", "tab:class", fmt="%.1f", index=True, tex_df=shorten(cl), star=True)
    write(mo, "mode_breakdown", f"Pass@{budget} (\\%) by fault mode. ``legacy'' modes come from the v1 injector; "
          "``extended'' modes were added in v2. " + SHORT_LEGEND + ".", "tab:mode", fmt="%.1f",
          tex_df=shorten(mo.drop(columns=[c for c in mo.columns if c == LABELS['static_no_oracle']])), star=True)
    write(st, "rejection_stages", "Rejecting stage of the final attempt among unresolved circuits. "
          + SHORT_LEGEND + ".", "tab:stages", fmt="%d", index=True, tex_df=shorten(st), star=True)
    write(co, "oracle_cost", "Oracle cost and overfitting of the search signal (searches that used the "
          "oracle). Gap = suite agreement minus exact process fidelity of the final candidate.", "tab:cost",
          tex_df=compact_cost(co), star=True)

    fig_passk(df, budget); fig_class_heatmap(df, budget); fig_queries_cdf(df); fig_gap(df)

    if weak and os.path.exists(weak):
        dw = pd.read_csv(weak)
        rw = rate_table(dw, budget); cw = contrast_table(dw)
        print("\n=== WEAK ORACLE (suite size 1) ==="); print(rw.to_string(index=False))
        print(cw[["contrast", "metric", "b10", "b01", "rd", "rd_lo", "rd_hi", "p_holm"]].to_string(index=False))
        write(rw, "weak_oracle_rates", "Weak-oracle regime (initial suite = $|0\\ldots0\\rangle$ only; "
              "oracle budget unchanged). " + SHORT_LEGEND + ".", "tab:weak", tex_df=compact_rates(rw),
              star=True)
        write(contrast_view(cw), "weak_oracle_contrasts", "Paired contrasts in the weak-oracle regime "
              f"(Holm over the {cw.attrs['family_size']} rows shown).", "tab:weakc", star=True)
        wcl = class_table(dw, budget)
        write(wcl, "weak_oracle_class", f"Weak-oracle Pass@{budget} (\\%) by fault class. "
              + SHORT_LEGEND + ".", "tab:weakclass", fmt="%.1f", index=True, tex_df=shorten(wcl))

    sw = load_sweep(sweep_glob)
    if len(sw):
        t = (sw.groupby(["system_name", "budget"]).pass_at_3.mean() * 100).unstack(0).round(1)
        print("\n=== Query-budget sweep (Pass@3 %) ==="); print(t.to_string())
        t2 = t.rename(columns=LABELS)
        write(t2, "budget_sweep", "Pass@3 (\\%) as a function of the per-repair oracle query budget "
              "(suite size 4). " + SHORT_LEGEND + ".", "tab:budget", fmt="%.1f", index=True,
              tex_df=shorten(t2))
        fig_budget(sw)
    print(f"\nTables -> {TAB_DIR}\nFigures -> {FIG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(PROJECT_ROOT, "experiments", "results_main.csv"))
    ap.add_argument("--weak", default=os.path.join(PROJECT_ROOT, "experiments", "results_weak_oracle.csv"))
    ap.add_argument("--sweep", default=os.path.join(PROJECT_ROOT, "experiments", "results_budget_*.csv"))
    ap.add_argument("--budget", type=int, default=3)
    a = ap.parse_args()
    main(a.results, a.weak, a.sweep, a.budget)
