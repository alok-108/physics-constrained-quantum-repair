"""Numerical building blocks: batched simulator, routing, cancellation, batched edit scoring."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, HGate, XGate
from qiskit.quantum_info import Statevector

from agents.oracle import BehaviouralOracle
from src.edits import Edit, apply_edit
from src.physics_model import get_default_physics_model
from src.routing import cancel_inverses, is_irreducible, route_and_cancel, route_with_swap_back
from src.sim import evolve_circuit, process_fidelity, random_product_states, random_states


def random_circuit(n, gates, rng):
    qc = QuantumCircuit(n)
    for _ in range(gates):
        kind = rng.integers(0, 6)
        a, b = (int(x) for x in rng.choice(n, 2, replace=False))
        if kind == 0:
            qc.h(a)
        elif kind == 1:
            qc.rx(float(rng.normal()), a)
        elif kind == 2:
            qc.cx(a, b)
        elif kind == 3:
            qc.cp(float(rng.normal()), a, b)
        elif kind == 4:
            qc.swap(a, b)
        else:
            qc.ry(float(rng.normal()), a)
    return qc


@pytest.mark.parametrize("n", [2, 3, 5, 7])
def test_fast_simulator_matches_qiskit(n):
    rng = np.random.default_rng(n)
    qc = random_circuit(n, 40, rng)
    init = np.concatenate([random_product_states(n, 2, rng), random_states(n, 2, rng)], axis=1)
    out = evolve_circuit(qc, init)
    for j in range(init.shape[1]):
        ref = Statevector(init[:, j]).evolve(qc).data
        assert np.allclose(out[:, j], ref, atol=1e-12)


def test_routing_preserves_unitary_and_is_admissible():
    physics = get_default_physics_model()
    rng = np.random.default_rng(0)
    qc = random_circuit(6, 30, rng)
    routed = route_and_cancel(qc, physics)
    assert physics.validate_circuit_connectivity(routed)[0]
    assert process_fidelity(qc, routed) == pytest.approx(1.0, abs=1e-9)
    assert is_irreducible(routed)
    assert len(route_with_swap_back(qc, physics).data) >= len(routed.data)


def test_cancel_inverses_removes_dag_adjacent_pairs():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 1)
    assert len(cancel_inverses(qc).data) == 0
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.h(0)
    assert len(cancel_inverses(qc2).data) == 3


def test_batched_edit_scores_equal_naive_scores_and_are_charged():
    rng = np.random.default_rng(1)
    ref = random_circuit(4, 25, rng)
    base = ref.copy_empty_like()
    for i, ins in enumerate(ref.data):
        if i != 5:
            base._append(ins)                 # one instruction removed
    oracle = BehaviouralOracle(ref, query_budget=10 ** 6, seed_key="t")
    edits = [Edit("insert", p, HGate(), (q,)) for p in range(len(base.data) + 1) for q in range(4)]
    edits += [Edit("insert", p, CXGate(), (0, 1)) for p in range(len(base.data) + 1)]
    edits += [Edit("delete", p) for p in range(len(base.data))]
    fast = oracle.evaluate_edits(base, edits)
    assert oracle.queries_used == len(edits)
    for e, f in list(zip(edits, fast))[::5]:
        assert oracle.score(apply_edit(base, e)) == pytest.approx(f, abs=1e-10)


def test_budget_is_enforced_and_unevaluated_edits_are_nan():
    ref = QuantumCircuit(2)
    ref.h(0)
    ref.cx(0, 1)
    oracle = BehaviouralOracle(ref, query_budget=3, seed_key="t")
    edits = [Edit("insert", 0, XGate(), (0,)) for _ in range(5)]
    scores = oracle.evaluate_edits(ref, edits)
    assert np.isfinite(scores[:3]).all() and np.isnan(scores[3:]).all()
    assert oracle.exhausted() and oracle.score(ref) == 0.0
