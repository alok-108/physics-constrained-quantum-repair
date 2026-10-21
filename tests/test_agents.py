"""Information boundary and behaviour of the repair agents."""

import inspect
import math
import os

import numpy as np
import pytest
import qiskit.qasm3
from qiskit import QuantumCircuit

from agents import repairer as repairer_module
from agents.diagnostician import DiagnosticianAgent
from agents.oracle import BehaviouralOracle
from agents.orchestrator import Orchestrator
from agents.repairer import RepairerAgent
from src.routing import circuit_key

from conftest import read_pair


# ---------------------------------------------------------------- boundary
def test_repair_interface_has_no_reference_or_label_channel():
    params = set(inspect.signature(RepairerAgent.repair).parameters)
    assert not params & {"ground_truth", "ground_truth_hint", "reference", "bug_type"}
    assert "ground_truth" not in inspect.signature(DiagnosticianAgent.diagnose).parameters


def test_repairer_never_touches_private_oracle_state():
    src = inspect.getsource(repairer_module)
    for forbidden in ("._ref", "_suite_ref", "_pool_ref", "_hold_ref", "ground_truth"):
        assert forbidden not in src.replace("ground-truth", ""), forbidden


def test_fault_labels_are_stripped_from_the_specification():
    from agents.diagnostician import LEAKY_KEYS
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    spec = {"circuit_type": "ghz", "num_qubits": 2, "bug_type": "syntax", "fault_mode": "x"}
    report = DiagnosticianAgent().diagnose(qiskit.qasm3.dumps(qc), spec)
    assert report["bug_type"] != "syntax"           # label was not used
    assert {"bug_type", "fault_mode"} <= set(LEAKY_KEYS)


# ---------------------------------------------------------------- oracle
def test_counterexample_extends_suite_and_consumes_budget():
    ref = QuantumCircuit(3)
    ref.h(0)
    ref.cx(0, 1)
    ref.rz(0.4, 2)
    oracle = BehaviouralOracle(ref, query_budget=50, suite_size=1, seed_key="cx")
    wrong = QuantumCircuit(3)
    wrong.h(0)
    wrong.cx(0, 1)
    before = oracle.score(wrong)
    agreement = oracle.counterexample(wrong)
    after = oracle.score(wrong)
    assert oracle.counterexamples_used == 1 and oracle.queries_used == 3
    assert agreement is not None and agreement < 1.0
    assert after < before                           # the new test exposes the missing rz


def test_holdout_gap_reports_overfitting_of_weak_suite():
    ref = QuantumCircuit(3)
    ref.h(0)
    ref.cx(0, 1)
    ref.rz(2.0, 2)
    oracle = BehaviouralOracle(ref, suite_size=1, seed_key="gap")
    candidate = QuantumCircuit(3)
    candidate.h(0)
    candidate.cx(0, 1)                              # rz missing: invisible on |000>
    gap = oracle.holdout_gap(candidate)
    assert gap["suite"] == pytest.approx(1.0) and gap["process"] < 0.5 and gap["gap"] > 0.4


# ---------------------------------------------------------------- search
def test_angle_solver_recovers_a_perturbed_rotation_in_few_queries():
    ref = QuantumCircuit(3)
    ref.h(0)
    ref.cx(0, 1)
    ref.rx(0.83, 2)
    ref.cp(1.1, 1, 2)
    bad = QuantumCircuit(3)
    bad.h(0)
    bad.cx(0, 1)
    bad.rx(0.83 + math.pi / 3, 2)
    bad.cp(1.1, 1, 2)
    oracle = BehaviouralOracle(ref, query_budget=500, seed_key="ang")
    agent = RepairerAgent()
    fixed = agent._search(bad, oracle, set())
    ang = [i.operation.params[0] for i in fixed.data if i.operation.name == "rx"][0]
    assert abs(ang - 0.83) < 1e-3
    assert oracle.queries_used < 150


def test_tabu_forces_a_different_candidate_on_retry():
    ref = QuantumCircuit(3)
    ref.h(0)
    ref.cx(0, 1)
    ref.cx(1, 2)
    bad = QuantumCircuit(3)
    bad.h(0)
    bad.cx(1, 0)
    bad.cx(1, 2)
    agent = RepairerAgent()
    oracle = BehaviouralOracle(ref, query_budget=5000, seed_key="tabu")
    first = agent._search(bad, oracle, set())
    second = agent._search(first, oracle, {circuit_key(first)})
    assert circuit_key(first) != circuit_key(second)


def test_physics_awareness_gates_the_routing_path():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 3)                                     # non-adjacent on the device
    text = qiskit.qasm3.dumps(qc)
    report = {"bug_type": "physics_violation"}
    with_physics = RepairerAgent(use_physics_context=True).repair(text, report, {}, oracle=None)
    without = RepairerAgent(use_physics_context=False).repair(text, report, {}, oracle=None)
    from src.physics_model import get_default_physics_model
    phys = get_default_physics_model()
    assert phys.validate_circuit_connectivity(qiskit.qasm3.loads(with_physics))[0]
    assert not phys.validate_circuit_connectivity(qiskit.qasm3.loads(without))[0]


# ---------------------------------------------------------------- end to end
def test_orchestrator_repairs_one_circuit_per_fault_class(metadata, root, validator):
    orch = Orchestrator(max_iterations=3, validator=validator)
    for bug in ["syntax", "structural", "gate_redundancy", "semantic_deviation", "physics_violation"]:
        row = metadata[(metadata.bug_type == bug) & (metadata.circuit_type == "vqe_h2")].iloc[0]
        buggy, gt = read_pair(root, row)
        res = orch.run(row["circuit_id"], buggy, gt, {"circuit_type": row["circuit_type"],
                                                      "num_qubits": int(row["num_qubits"])})
        assert res["status"] == "PASS", (bug, [h["failed_stage"] for h in res["history"]])
        assert 1 <= res["iterations"] <= 3


def test_unconstrained_arm_cannot_repair_a_physics_fault(metadata, root, validator):
    row = metadata[(metadata.bug_type == "physics_violation") & (metadata.fault_mode == "unmapped")].iloc[0]
    buggy, gt = read_pair(root, row)
    orch = Orchestrator(max_iterations=2, validator=validator, use_physics_context=False,
                        verbose_feedback=False)
    res = orch.run(row["circuit_id"], buggy, gt, {"circuit_type": row["circuit_type"],
                                                  "num_qubits": int(row["num_qubits"])})
    assert res["status"] == "UNRESOLVED"
    assert res["history"][-1]["failed_stage"] == "connectivity"


def test_run_is_deterministic(metadata, root, validator):
    row = metadata[(metadata.bug_type == "semantic_deviation") & (metadata.circuit_type == "qaoa_maxcut")].iloc[0]
    buggy, gt = read_pair(root, row)
    spec = {"circuit_type": row["circuit_type"], "num_qubits": int(row["num_qubits"])}
    a = Orchestrator(validator=validator).run(row["circuit_id"], buggy, gt, spec)
    b = Orchestrator(validator=validator).run(row["circuit_id"], buggy, gt, spec)
    assert a["final_candidate_qasm"] == b["final_candidate_qasm"]
    assert a["oracle_queries"] == b["oracle_queries"]
