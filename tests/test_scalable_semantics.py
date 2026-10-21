"""Sampled semantic check: agrees with the exact check and scales past dense-unitary limits."""

import math

import pytest
import qiskit.qasm3
from qiskit import QuantumCircuit

from src.routing import route_and_cancel
from src.sim import process_fidelity, sampled_process_fidelity


def qaoa_like(n, gamma=0.7, beta=0.3):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
        qc.rz(gamma, i + 1)
        qc.cx(i, i + 1)
    for i in range(n):
        qc.rx(2 * beta, i)
    return qc


def test_sampled_estimate_matches_exact_on_small_circuits():
    ref = qaoa_like(6)
    for delta in (0.0, 0.3, 1.2, math.pi):
        bad = qaoa_like(6, gamma=0.7 + delta)
        exact = process_fidelity(ref, bad)
        est, worst, se = sampled_process_fidelity(ref, bad, n_states=64, seed=1)
        assert est == pytest.approx(exact, abs=0.08), (delta, exact, est)


def test_identical_circuits_give_fidelity_one_and_faults_are_flagged_beyond_dense_limit():
    n = 14                                   # dense 2^14 x 2^14 unitary would need ~4 GB
    ref = qaoa_like(n)
    est, worst, _ = sampled_process_fidelity(ref, ref.copy(), n_states=8)
    assert est == pytest.approx(1.0, abs=1e-9) and worst == pytest.approx(1.0, abs=1e-9)
    bad = ref.copy()
    bad.rx(1.0, 3)                           # one extra rotation
    est_bad, worst_bad, _ = sampled_process_fidelity(ref, bad, n_states=8)
    assert est_bad < 0.9 and worst_bad < 0.95


def test_validator_uses_sampled_check_above_the_dense_limit(validator):
    n = 12
    ref = route_and_cancel(qaoa_like(n))
    ok = validator.validate(qiskit.qasm3.dumps(ref), ref, gt_key="scale")
    assert ok["status"] == "PASS" and ok["metrics"]["semantic_check"] == "sampled"
    assert ok["metrics"]["noise_check"].startswith("skipped")
    bad = ref.copy()
    bad.rz(0.9, 5)
    v = validator.validate(qiskit.qasm3.dumps(bad), ref, gt_key="scale")
    assert v["failed_stage"] == "semantic"
