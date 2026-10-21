"""The acceptance predicate: each stage rejects what it claims to reject."""

import math

import qiskit.qasm3
from qiskit import QuantumCircuit

from conftest import read_pair
from src.routing import route_and_cancel


def qft(n, negate=False):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp((-1 if negate else 1) * math.pi / 2 ** (k - j), k, j)
    return qc


def test_reference_is_accepted_and_verdict_is_deterministic(validator, metadata, root):
    _, gt = read_pair(root, metadata.iloc[10])
    v1 = validator.validate(gt, gt, gt_key="a")
    v2 = validator.validate(gt, gt, gt_key="a")
    assert v1["status"] == "PASS" and v1 == v2
    assert v1["metrics"]["tvd"] == 0.0


def test_every_reference_family_is_accepted_as_its_own_repair(validator, metadata, root):
    """No correct repair may be rejected: a reference submitted as a candidate passes."""
    for ctype in metadata.circuit_type.unique():
        row = metadata[metadata.circuit_type == ctype].iloc[0]
        _, gt = read_pair(root, row)
        assert validator.validate(gt, gt, gt_key=row["circuit_id"])["status"] == "PASS", ctype


def test_syntax_stage(validator, metadata, root):
    buggy, gt = read_pair(root, metadata[metadata.bug_type == "syntax"].iloc[0])
    v = validator.validate(buggy, gt)
    assert v["status"] == "FAIL" and v["failed_stage"] == "syntax"


def test_width_stage(validator):
    ref = QuantumCircuit(3)
    ref.h(0)
    cand = QuantumCircuit(4)
    cand.h(0)
    assert validator.validate(qiskit.qasm3.dumps(cand), ref)["failed_stage"] == "structural"


def test_connectivity_stage_rejects_unrouted_circuit(validator):
    ref = QuantumCircuit(3)
    ref.h(0)
    cand = QuantumCircuit(3)
    cand.h(0)
    cand.cx(0, 2)                         # (0, 2) is not an edge on Sherbrooke
    assert validator.validate(qiskit.qasm3.dumps(cand), ref)["failed_stage"] == "connectivity"


def test_coherence_stage_rejects_long_delay(validator):
    ref = QuantumCircuit(2)
    ref.h(0)
    cand = ref.copy()
    cand.delay(1200, 0, unit="us")
    assert validator.validate(qiskit.qasm3.dumps(cand), ref)["failed_stage"] == "coherence"


def test_minimality_stage_rejects_unreduced_circuit(validator):
    ref = QuantumCircuit(2)
    ref.h(0)
    ref.cx(0, 1)
    cand = ref.copy()
    cand.x(1)
    cand.x(1)
    assert validator.validate(qiskit.qasm3.dumps(cand), ref)["failed_stage"] == "minimality"


def test_semantic_stage_is_not_blind_to_qft_phase_faults(validator):
    """Regression for the v1 single-input check: on |0...0> a QFT with every
    controlled phase negated has overlap 1 with the reference."""
    ref, bad = route_and_cancel(qft(4)), route_and_cancel(qft(4, negate=True))   # admissible, so the semantic stage is reached
    v = validator.validate(qiskit.qasm3.dumps(bad), ref)
    assert v["status"] == "FAIL" and v["failed_stage"] == "semantic"
    assert v["metrics"]["process_fidelity"] < 0.1


def test_global_phase_is_not_a_fault(validator):
    ref = QuantumCircuit(2)
    ref.h(0)
    ref.cx(0, 1)
    cand = ref.copy()
    cand.global_phase = 0.7
    assert validator.validate(qiskit.qasm3.dumps(cand), ref)["status"] == "PASS"


def test_noise_stage_is_sensitive_to_gate_count_not_just_unitary(validator):
    """Same unitary, many extra noisy two-qubit gates: cx . rz(control) . cx == rz(control)."""
    ref = QuantumCircuit(3)
    ref.h(0)
    ref.cx(0, 1)
    ref.cx(1, 2)
    cand = ref.copy()
    for _ in range(6):
        cand.cx(0, 1)
        cand.rz(0.2, 0)
        cand.cx(0, 1)
        cand.cx(1, 2)
        cand.rz(0.2, 1)
        cand.cx(1, 2)
    cand.rz(-1.2, 0)
    cand.rz(-1.2, 1)
    v = validator.validate(qiskit.qasm3.dumps(cand), ref)
    assert v["metrics"]["process_fidelity"] > 0.999
    assert v["failed_stage"] == "noise"
