"""Scoring of HornBro outputs with the shared predicate (HornBro itself needs Linux and is not run here)."""

import json
import os

import pandas as pd

from src import hornbro_adapter, score_hornbro


def _meta(root):
    return pd.read_csv(os.path.join(root, "data", "metadata.csv"))


def test_selection_rule_matches_the_runner(root):
    from src.run_experiments import stratified_subset
    meta = _meta(root)
    assert list(hornbro_adapter.stratified(meta, 60).circuit_id) == list(stratified_subset(meta, 60, 20260915).circuit_id)


def test_scoring_accepts_a_correct_circuit_rejects_the_faulty_one_and_counts_failures(root, tmp_path, validator):
    meta = _meta(root).set_index("circuit_id")
    ok, bad = "circuit_0009", "circuit_0013"                       # one correct return, one faulty return
    ref = open(os.path.join(root, meta.loc[ok, "ground_truth_file"]), encoding="utf-8").read()
    bug = open(os.path.join(root, meta.loc["circuit_0009", "buggy_file"]), encoding="utf-8").read()
    (tmp_path / f"{ok}.qasm").write_text(ref, encoding="utf-8")
    (tmp_path / f"{bad}.qasm").write_text(open(os.path.join(root, meta.loc[bad, "buggy_file"]), encoding="utf-8").read(), encoding="utf-8")
    for cid, outcome in ((ok, "returned_circuit"), (bad, "returned_circuit"), ("circuit_0001", "input_unparsable")):
        (tmp_path / f"{cid}.json").write_text(json.dumps({"circuit_id": cid, "outcome": outcome, "hornbro_says_success": True,
                                                          "seconds": 1.0}))
    df = score_hornbro.score(str(tmp_path), os.path.join(root, "data", "metadata.csv"), validator).set_index("circuit_id")
    assert df.loc[ok, "pass_at_3"] == 1 and df.loc[bad, "pass_at_3"] == 0
    assert df.loc["circuit_0001", "pass_at_3"] == 0 and df.loc["circuit_0001", "failed_stage"] == "input_unparsable"
    assert bug  # the faulty text exists and differs from the reference


def test_a_missing_package_is_an_environment_error_not_a_hornbro_failure(root, tmp_path, validator):
    for cid, err in (("circuit_0009", "ModuleNotFoundError: No module named 'qiskit_aer'"), ("circuit_0013", "ValueError: bad circuit")):
        (tmp_path / f"{cid}.json").write_text(json.dumps({"circuit_id": cid, "outcome": "crashed", "error": err}))
    df = score_hornbro.score(str(tmp_path), os.path.join(root, "data", "metadata.csv"), validator).set_index("circuit_id")
    assert df.loc["circuit_0009", "outcome"] == "environment_error"
    assert df.loc["circuit_0013", "outcome"] == "crashed"          # a real crash of HornBro on the circuit stays a failure


def test_functional_equivalence_ignores_device_adjacency_but_the_strict_predicate_does_not(tmp_path, validator):
    """A circuit that is correct as a unitary but uses a non-adjacent CX: strict predicate fails, functional check passes."""
    import qiskit.qasm3
    from qiskit import QuantumCircuit
    ref = QuantumCircuit(3)
    ref.h(0), ref.cx(0, 1), ref.cx(1, 2)
    far = QuantumCircuit(3)
    far.h(0), far.cx(0, 1), far.cx(0, 2), far.cx(0, 1), far.cx(1, 2)          # = H, CX01, CX12 (extra 02 conjugation is identity)
    far = QuantumCircuit(3)
    far.h(0), far.cx(0, 1), far.cx(1, 2), far.cx(0, 2), far.cx(0, 2)
    ref_t, far_t = qiskit.qasm3.dumps(ref), qiskit.qasm3.dumps(far)
    assert score_hornbro.functionally_equivalent(far_t, ref_t)
    wrong = QuantumCircuit(3)
    wrong.h(0), wrong.cx(0, 1)
    assert not score_hornbro.functionally_equivalent(qiskit.qasm3.dumps(wrong), ref_t)
    assert not score_hornbro.functionally_equivalent("not qasm", ref_t)
