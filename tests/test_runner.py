"""End-to-end smoke test of the benchmark runner and its output schema."""

import pandas as pd

from src.run_experiments import run_experiments

EXPECTED = {
    "circuit_id", "circuit_type", "num_qubits", "bug_type", "fault_mode", "mode_family", "system_name",
    "pass_at_1", "pass_at_2", "pass_at_3", "accepted", "iterations_used", "distinct_candidates",
    "oracle_queries", "counterexamples", "repeated_candidate_any", "final_failed_stage",
    "physics_valid_final", "process_fidelity_final", "overfit_gap", "gates_after", "ref_gates",
    "overhead_vs_ref_pct", "duration_s",
}


def test_runner_produces_consistent_rows_for_repair_static_and_compiler_arms(root, tmp_path):
    out = tmp_path / "res.csv"
    systems = ["ours_full", "static_no_oracle", "qiskit_transpile"]
    df = run_experiments(f"{root}/data/metadata.csv", str(out), systems, limit=10, workers=1)
    assert EXPECTED <= set(df.columns)
    assert set(df.system_name) == set(systems)
    assert (df.groupby("system_name").size() == df.circuit_id.nunique()).all()
    # Pass@k is monotone and consistent with the accepted flag
    assert (df.pass_at_1 <= df.pass_at_2).all() and (df.pass_at_2 <= df.pass_at_3).all()
    assert (df.accepted == df.pass_at_3).all()
    # the static and compiler arms never use the oracle
    assert (df[df.system_name != "ours_full"].oracle_queries == 0).all()
    # resuming from the partial file reproduces the same table
    again = run_experiments(f"{root}/data/metadata.csv", str(out), systems, limit=10, workers=1, resume=True)
    pd.testing.assert_frame_equal(df.drop(columns="duration_s"), again.drop(columns="duration_s"))
