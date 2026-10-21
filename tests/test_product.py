"""The qrepair product surface: lint, localize, verify, repair, CLI, and the extended edit space."""

import json
import numpy as np
import pytest
import qiskit.qasm3
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from agents.oracle import BehaviouralOracle
from agents.repairer import RepairerAgent
from src import cli, product


def ghz_reference():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


def ghz_buggy():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(1, 0)                              # control and target reversed
    qc.cx(1, 2)
    return qc


def ghz_tests(inputs=("000", "100", "001", "010")):
    ref = ghz_reference()
    tests = []
    for bits in inputs:
        out = Statevector.from_label(bits).evolve(ref).data
        exp = {format(i, "03b"): [float(a.real), float(a.imag)] for i, a in enumerate(out) if abs(a) > 1e-9}
        tests.append({"input": bits, "expect": exp})
    return {"num_qubits": 3, "tests": tests}


# ------------------------------------------------------------------ lint
def test_lint_reports_each_class_of_finding():
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.x(1)
    qc.x(1)
    qc.cx(0, 3)
    qc.rz(0.0, 2)
    codes = {f.code for f in product.lint(qiskit.qasm3.dumps(qc))}
    assert {"W101", "W201", "W202", "I301", "I302"} <= codes


def test_lint_reports_parse_errors_with_a_line_number():
    bad = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nh q[0]\ncx q[0], q[1];\n'
    findings = product.lint(bad)
    assert findings[0].code == "E001" and findings[0].severity == "error"


def test_clean_circuit_has_no_warnings():
    assert not [f for f in product.lint(qiskit.qasm3.dumps(ghz_reference())) if f.severity != "info"]


# ------------------------------------------------------------------ specs
def test_test_file_parsing_and_validation():
    inputs, expected, n = product.load_tests(ghz_tests())
    assert n == 3 and inputs.shape == expected.shape == (8, 4)
    assert np.allclose(np.linalg.norm(expected, axis=0), 1.0)
    with pytest.raises(ValueError):
        product.load_tests({"num_qubits": 2, "tests": [{"input": "0x", "expect": {"00": 1}}]})


def test_tests_only_oracle_has_no_reference_features():
    oracle = product.tests_oracle(ghz_tests())
    assert oracle.counterexample(ghz_buggy()) is None          # no pool
    assert oracle.holdout_gap(ghz_buggy())["process"] is None  # no exact process fidelity


# ------------------------------------------------------------------ localize / verify / repair
def test_localize_ranks_the_reversed_cnot_first():
    oracle = product.tests_oracle(ghz_tests())
    rows = product.localize(qiskit.qasm3.dumps(ghz_buggy()), oracle)
    assert rows[0]["instruction"] == 1 and rows[0]["agreement_gain"] > 0.5


def test_verify_with_tests_and_with_reference(validator):
    bug = qiskit.qasm3.dumps(ghz_buggy())
    ref = qiskit.qasm3.dumps(ghz_reference())
    assert product.verify(bug, tests=ghz_tests(), validator=validator)["failed_stage"] == "tests"
    assert product.verify(bug, reference=ref, validator=validator)["failed_stage"] == "semantic"
    assert product.verify(ref, reference=ref, validator=validator)["status"] == "PASS"


@pytest.mark.parametrize("mode", ["tests", "reference"])
def test_repair_returns_a_minimal_diff_and_the_right_certificate(mode):
    bug = qiskit.qasm3.dumps(ghz_buggy())
    kwargs = ({"tests": ghz_tests()} if mode == "tests"
              else {"reference": qiskit.qasm3.dumps(ghz_reference())})
    rep = product.repair(bug, **kwargs)
    assert rep.status == "PASS" and rep.gates_before == rep.gates_after == 3
    assert "-cx q[1], q[0];" in rep.diff and "+cx q[0], q[1];" in rep.diff
    assert ("NOT an equivalence certificate" in rep.certificate) == (mode == "tests")
    assert (len(rep.warnings) > 0) == (mode == "tests")


def test_repair_of_a_correct_circuit_is_reported_as_already_correct():
    rep = product.repair(qiskit.qasm3.dumps(ghz_reference()), tests=ghz_tests())
    assert rep.status == "ALREADY_CORRECT" and rep.repaired_qasm


def test_width_mismatch_is_a_clear_error_not_a_failed_repair():
    wide = QuantumCircuit(4)
    wide.h(0)
    with pytest.raises(ValueError, match="qubits"):
        product.repair(qiskit.qasm3.dumps(wide), tests=ghz_tests())


def test_tests_that_are_too_weak_are_flagged_by_a_stronger_oracle():
    """A fault that never triggers on the weak suite's inputs passes it; a stronger suite exposes it."""
    wrong = QuantumCircuit(3)
    wrong.cx(2, 1)                            # control q2 is |0> on |000>, so this is invisible there
    wrong.h(0)
    wrong.cx(0, 1)
    wrong.cx(1, 2)
    weak = ghz_tests(inputs=("000",))
    strong = ghz_tests(inputs=("000", "100"))
    assert product.verify(qiskit.qasm3.dumps(wrong), tests=strong)["status"] == "FAIL"
    assert product.verify(qiskit.qasm3.dumps(wrong), tests=weak)["status"] == "PASS"


# ------------------------------------------------------------------ extended edit space
def test_extended_edits_repair_a_cx_to_cz_substitution():
    ref = ghz_reference()
    bad = QuantumCircuit(3)
    bad.h(0)
    bad.cz(0, 1)
    bad.cx(1, 2)
    oracle = BehaviouralOracle(ref, query_budget=5000, seed_key="ext")
    fixed = RepairerAgent(extended_edits=True)._search(bad, oracle, set())
    assert [i.operation.name for i in fixed.data] == ["h", "cx", "cx"]


def test_extended_edits_reinsert_a_removed_rotation():
    ref = QuantumCircuit(3)
    ref.h(0)
    ref.cx(0, 1)
    ref.rz(0.9, 1)
    ref.cx(1, 2)
    bad = QuantumCircuit(3)
    bad.h(0)
    bad.cx(0, 1)
    bad.cx(1, 2)
    base = RepairerAgent(extended_edits=False)._search(bad, BehaviouralOracle(ref, 5000, seed_key="r"), set())
    ext = RepairerAgent(extended_edits=True)._search(bad, BehaviouralOracle(ref, 5000, seed_key="r"), set())
    names = lambda c: [i.operation.name for i in c.data]
    assert "rz" not in names(base) and "rz" in names(ext)
    angle = [i.operation.params[0] for i in ext.data if i.operation.name == "rz"][0]
    assert abs(angle - 0.9) < 1e-3


# ------------------------------------------------------------------ CLI
def test_cli_exit_codes_and_json(tmp_path, capsys):
    bug, ref, tests = tmp_path / "bug.qasm", tmp_path / "ref.qasm", tmp_path / "tests.json"
    bug.write_text(qiskit.qasm3.dumps(ghz_buggy()))
    ref.write_text(qiskit.qasm3.dumps(ghz_reference()))
    tests.write_text(json.dumps(ghz_tests()))
    out = tmp_path / "fixed.qasm"

    assert cli.main(["verify", str(bug), "--tests", str(tests)]) == 1
    assert cli.main(["verify", str(ref), "--reference", str(ref)]) == 0
    capsys.readouterr()
    assert cli.main(["repair", str(bug), "--tests", str(tests), "--out", str(out), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS" and payload["diff"] and payload["trace"]
    assert cli.main(["verify", str(out), "--reference", str(ref)]) == 0     # the written fix really is correct
    assert cli.main(["lint", str(out)]) == 0
    assert cli.main(["verify", str(tmp_path / "missing.qasm"), "--tests", str(tests)]) == 2
