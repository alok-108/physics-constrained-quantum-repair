"""Statistical Analysis and Plot Generation (src/analyze_results.py)

Performs comprehensive empirical evaluation on experiments/results.csv:
1. Computes mean and standard deviation for key metrics across systems.
2. Conducts paired Wilcoxon signed-rank tests and Cohen's d effect size calculations.
3. Quantifies bug-type differential gains from physical constraints.
4. Produces publication-quality plots (PDF & PNG, 300 DPI) in paper/figures/.
5. Generates LaTeX tables in paper/tables/.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_CSV = os.path.join(PROJECT_ROOT, "experiments", "results.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "paper", "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Publication styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

SYSTEM_COLORS = {
    "physics_constrained_agent": "#2b5c8f",
    "QBugLM_unconstrained": "#d95f02",
    "PyZX_heuristic": "#7570b3"
}
SYSTEM_LABELS = {
    "physics_constrained_agent": "Physics-Constrained Agent (Ours)",
    "QBugLM_unconstrained": "QBugLM (Unconstrained)",
    "PyZX_heuristic": "PyZX (Heuristic)"
}


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size for two paired groups."""
    diff = group1 - group2
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    return float(np.mean(diff) / std_diff)


def run_statistical_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run descriptive statistics, Wilcoxon tests, and bug type breakdowns."""
    metrics = [
        "pass_at_1",
        "pass_at_3",
        "physics_validity",
        "gate_count_reduction",
        "depth_reduction",
        "iterations_used"
    ]

    # 1. Descriptive Stats (Mean & Std)
    summary_rows = []
    systems = ["physics_constrained_agent", "QBugLM_unconstrained", "PyZX_heuristic"]

    for sys in systems:
        sub = df[df["system_name"] == sys]
        row = {"system_name": sys, "total_circuits": len(sub)}
        for m in metrics:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # 2. Paired Wilcoxon signed-rank tests
    # Pivot circuits so each circuit has columns per system
    pivoted = df.pivot(index="circuit_id", columns="system_name")

    ours = pivoted.xs("physics_constrained_agent", level="system_name", axis=1)
    qbuglm = pivoted.xs("QBugLM_unconstrained", level="system_name", axis=1)
    pyzx = pivoted.xs("PyZX_heuristic", level="system_name", axis=1)

    wilcox_rows = []
    comparisons = [
        ("Physics-Constrained vs QBugLM", ours, qbuglm),
        ("Physics-Constrained vs PyZX", ours, pyzx),
    ]

    test_metrics = ["pass_at_1", "pass_at_3", "physics_validity"]

    for comp_name, g1, g2 in comparisons:
        for m in test_metrics:
            v1 = g1[m].values
            v2 = g2[m].values
            d = cohens_d(v1, v2)
            try:
                # Wilcoxon signed rank test
                diff = v1 - v2
                if np.all(diff == 0):
                    stat, p_val = 0.0, 1.0
                else:
                    res = stats.wilcoxon(v1, v2, zero_method="pratt", alternative="two-sided")
                    stat, p_val = res.statistic, res.pvalue
            except Exception as e:
                stat, p_val = np.nan, np.nan

            wilcox_rows.append({
                "comparison": comp_name,
                "metric": m,
                "mean_ours": float(np.mean(v1)),
                "mean_baseline": float(np.mean(v2)),
                "wilcoxon_stat": stat,
                "p_value": p_val,
                "is_significant (p < 0.05)": p_val < 0.05,
                "cohens_d": d
            })

    wilcox_df = pd.DataFrame(wilcox_rows)

    # 3. Breakdown by bug_type
    bug_breakdown = df.groupby(["bug_type", "system_name"]).agg(
        pass_at_1_mean=("pass_at_1", "mean"),
        pass_at_3_mean=("pass_at_3", "mean"),
        physics_validity_mean=("physics_validity", "mean"),
        gate_reduction_mean=("gate_count_reduction", "mean"),
        depth_reduction_mean=("depth_reduction", "mean")
    ).reset_index()

    return summary_df, wilcox_df, bug_breakdown


def export_latex_tables(summary_df: pd.DataFrame, wilcox_df: pd.DataFrame, bug_df: pd.DataFrame):
    """Generate professional LaTeX tables in paper/tables/."""
    # Table 1: Overall Summary
    t1_path = os.path.join(TABLES_DIR, "overall_metrics.tex")
    with open(t1_path, "w", encoding="utf-8") as f:
        f.write("% Overall Performance Summary Table\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Overall Performance of Quantum Circuit Repair Systems across 1,000 Benchmark Circuits.}\n")
        f.write("\\label{tab:overall_performance}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("System & Pass@1 (\\%) & Pass@3 (\\%) & Physics Valid. (\\%) & Gate Red. & Depth Red. & Iters \\\\\n")
        f.write("\\hline\n")
        for _, r in summary_df.iterrows():
            sys_label = SYSTEM_LABELS.get(r["system_name"], r["system_name"])
            p1 = f"{r['pass_at_1_mean']*100:.1f} \\pm {r['pass_at_1_std']*100:.1f}"
            p3 = f"{r['pass_at_3_mean']*100:.1f} \\pm {r['pass_at_3_std']*100:.1f}"
            pv = f"{r['physics_validity_mean']*100:.1f} \\pm {r['physics_validity_std']*100:.1f}"
            gr = f"{r['gate_count_reduction_mean']:.1f} \\pm {r['gate_count_reduction_std']:.1f}"
            dr = f"{r['depth_reduction_mean']:.1f} \\pm {r['depth_reduction_std']:.1f}"
            it = f"{r['iterations_used_mean']:.2f} \\pm {r['iterations_used_std']:.2f}"
            f.write(f"{sys_label} & {p1} & {p3} & {pv} & {gr} & {dr} & {it} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

    # Table 2: Statistical Significance (Wilcoxon)
    t2_path = os.path.join(TABLES_DIR, "wilcoxon_statistical_tests.tex")
    with open(t2_path, "w", encoding="utf-8") as f:
        f.write("% Paired Wilcoxon Signed-Rank Test & Cohen's d\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Paired Wilcoxon Signed-Rank Tests and Effect Sizes comparing Physics-Constrained Agent to Baselines ($N=1,000$).}\n")
        f.write("\\label{tab:wilcoxon_tests}\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\hline\n")
        f.write("Comparison & Metric & Ours (Mean) & Base (Mean) & $p$-value & Cohen's $d$ \\\\\n")
        f.write("\\hline\n")
        for _, r in wilcox_df.iterrows():
            p_str = "$< 10^{-15}$" if r["p_value"] < 1e-15 else f"{r['p_value']:.4e}"
            f.write(f"{r['comparison']} & {r['metric']} & {r['mean_ours']:.3f} & {r['mean_baseline']:.3f} & {p_str} & {r['cohens_d']:.3f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    # Table 3: Bug Type Breakdown
    t3_path = os.path.join(TABLES_DIR, "bug_type_breakdown.tex")
    pivot_p3 = bug_df.pivot(index="bug_type", columns="system_name", values="pass_at_3_mean") * 100
    pivot_pv = bug_df.pivot(index="bug_type", columns="system_name", values="physics_validity_mean") * 100

    with open(t3_path, "w", encoding="utf-8") as f:
        f.write("% Bug Type Breakdown Table\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Repair Success (Pass@3 \\%) and Hardware Physics Validity (\\%) Broken Down by Defect Category ($N=200$ per bug type).}\n")
        f.write("\\label{tab:bug_breakdown}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write(" & \\multicolumn{3}{c}{\\textbf{Pass@3 (\\%)}} & \\multicolumn{3}{c}{\\textbf{Hardware Physics Validity (\\%)}}\\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write("Bug Type & PyZX & QBugLM & Ours & PyZX & QBugLM & Ours \\\\\n")
        f.write("\\hline\n")
        for b_type in pivot_p3.index:
            p_pyzx = pivot_p3.loc[b_type, "PyZX_heuristic"]
            p_qbug = pivot_p3.loc[b_type, "QBugLM_unconstrained"]
            p_ours = pivot_p3.loc[b_type, "physics_constrained_agent"]

            pv_pyzx = pivot_pv.loc[b_type, "PyZX_heuristic"]
            pv_qbug = pivot_pv.loc[b_type, "QBugLM_unconstrained"]
            pv_ours = pivot_pv.loc[b_type, "physics_constrained_agent"]
            f.write(f"{b_type.replace('_', ' ').title()} & {p_pyzx:.1f} & {p_qbug:.1f} & \\textbf{{{p_ours:.1f}}} & {pv_pyzx:.1f} & {pv_qbug:.1f} & \\textbf{{{pv_ours:.1f}}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

    print(f"Exported LaTeX tables to {TABLES_DIR}")


def generate_plots(df: pd.DataFrame):
    """Generate all 5 publication-quality figures at 300 DPI in PDF and PNG format."""

    # -------------------------------------------------------------------------
    # Plot 1: Bar chart: pass@1 and pass@3 for all systems
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    pass_data = []
    for sys in ["PyZX_heuristic", "QBugLM_unconstrained", "physics_constrained_agent"]:
        sub = df[df["system_name"] == sys]
        pass_data.append({
            "System": SYSTEM_LABELS[sys],
            "Pass@1": sub["pass_at_1"].mean() * 100,
            "Pass@3": sub["pass_at_3"].mean() * 100,
        })
    pass_df = pd.DataFrame(pass_data).melt(id_vars="System", var_name="Metric", value_name="Success Rate (%)")

    ax = sns.barplot(
        data=pass_df,
        x="System",
        y="Success Rate (%)",
        hue="Metric",
        palette=["#4a90e2", "#1b3b6f"]
    )
    plt.title("Repair Success Rates (Pass@1 vs. Pass@3)", pad=15, fontweight="bold")
    plt.ylabel("Success Rate (%)", fontweight="bold")
    plt.xlabel("")
    plt.ylim(0, 110)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height + 2),
                        ha='center', va='bottom', fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pass_at_k_comparison.pdf"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "pass_at_k_comparison.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 2: Box plot: physics_validity per system
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plot_df = df.copy()
    plot_df["System"] = plot_df["system_name"].map(SYSTEM_LABELS)
    ax = sns.boxplot(
        data=plot_df,
        x="System",
        y="physics_validity",
        palette=[SYSTEM_COLORS[s] for s in ["PyZX_heuristic", "QBugLM_unconstrained", "physics_constrained_agent"]],
        width=0.45
    )
    plt.title("Hardware Physics Validity Distribution by System", pad=15, fontweight="bold")
    plt.ylabel("Physics Validity (0 = Violated, 1 = Valid)", fontweight="bold")
    plt.xlabel("")
    plt.yticks([0.0, 0.5, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "physics_validity_boxplot.pdf"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "physics_validity_boxplot.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 3: Scatter: gate_count_reduction vs pass_at_3, colored by system
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5.5))
    # Add subtle jitter on pass_at_3 (binary) for visual clarity in scatter
    plot_df["pass_at_3_jitter"] = plot_df["pass_at_3"] + np.random.normal(0, 0.03, size=len(plot_df))

    sns.scatterplot(
        data=plot_df,
        x="gate_count_reduction",
        y="pass_at_3_jitter",
        hue="System",
        palette=[SYSTEM_COLORS[s] for s in ["physics_constrained_agent", "QBugLM_unconstrained", "PyZX_heuristic"]],
        alpha=0.6,
        s=35,
        edgecolor=None
    )
    plt.title("Gate Count Reduction vs. Repair Success (Pass@3)", pad=15, fontweight="bold")
    plt.xlabel("Gate Count Reduction (Positive = Simplified)", fontweight="bold")
    plt.ylabel("Pass@3 (with Jitter)", fontweight="bold")
    plt.yticks([0, 1], ["FAIL (0)", "PASS (1)"])
    plt.axvline(0, color="gray", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "gate_reduction_vs_pass.pdf"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "gate_reduction_vs_pass.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 4: Heatmap: bug_type vs system, pass@3
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    heatmap_df = df.pivot_table(
        index="bug_type",
        columns="system_name",
        values="pass_at_3",
        aggfunc=lambda x: np.mean(x) * 100
    )
    # Rename columns and index for clean publication aesthetic
    heatmap_df.columns = [SYSTEM_LABELS.get(c, c) for c in heatmap_df.columns]
    heatmap_df.index = [idx.replace("_", " ").title() for idx in heatmap_df.index]
    # Reorder columns
    cols_order = [SYSTEM_LABELS["PyZX_heuristic"], SYSTEM_LABELS["QBugLM_unconstrained"], SYSTEM_LABELS["physics_constrained_agent"]]
    heatmap_df = heatmap_df[[c for c in cols_order if c in heatmap_df.columns]]

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "Pass@3 Success Rate (%)"},
        linewidths=1.0,
        vmin=0,
        vmax=100
    )
    plt.title("Pass@3 Success Rate (%) Across Bug Taxonomy", pad=15, fontweight="bold")
    plt.ylabel("Bug Category", fontweight="bold")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "heatmap_bugtype_vs_system.pdf"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "heatmap_bugtype_vs_system.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 5: Bar chart: mean iterations_used per system
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    iter_df = df.groupby("system_name")["iterations_used"].agg(["mean", "std"]).reset_index()
    iter_df["System"] = iter_df["system_name"].map(SYSTEM_LABELS)

    ax = sns.barplot(
        data=iter_df,
        x="System",
        y="mean",
        palette=[SYSTEM_COLORS[s] for s in iter_df["system_name"]],
        edgecolor="black"
    )
    plt.errorbar(
        x=range(len(iter_df)),
        y=iter_df["mean"],
        yerr=iter_df["std"],
        fmt="none",
        c="black",
        capsize=5
    )
    plt.title("Mean Iterations Used Per System (Max 3)", pad=15, fontweight="bold")
    plt.ylabel("Average Iterations", fontweight="bold")
    plt.xlabel("")
    plt.ylim(0, 3.5)
    for idx, row in iter_df.iterrows():
        ax.text(idx, row["mean"] + row["std"] + 0.15, f"{row['mean']:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "iterations_used_comparison.pdf"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "iterations_used_comparison.png"), dpi=300)
    plt.close()

    print(f"Generated all 5 publication figures in {FIGURES_DIR} (PDF and PNG, 300 DPI).")


def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"Results CSV not found at {RESULTS_CSV}. Run src/run_experiments.py first.")
        sys.exit(1)

    print(f"Loading results from {RESULTS_CSV}...")
    df = pd.read_csv(RESULTS_CSV)

    summary_df, wilcox_df, bug_df = run_statistical_analysis(df)
    export_latex_tables(summary_df, wilcox_df, bug_df)
    generate_plots(df)

    print("\n" + "=" * 80)
    print("                    STATISTICAL HYPOTHESIS TESTING RESULTS")
    print("=" * 80)
    print(wilcox_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
