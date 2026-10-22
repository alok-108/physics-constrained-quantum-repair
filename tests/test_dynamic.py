"""Dynamic programs: measurement, reset, classical control, loops (semantics, acceptance, repair, v3 data)."""

import math
import os
import random

import numpy as np
import pandas as pd
import pytest

from src import dynamic as D
from src import generate_v3 as G
from src import product
from src.dynamic import For, Gate, If, Measure, Program, Reset
from src.sim import process_fidelity


def basis(n, idx=0):
    v = np.zeros(2 ** n, dtype=complex)
    v[idx] = 1.0
    return v


# ------------------------------------------------------------------ semantics
def test_ghz_measurement_distribution():
    p = G.ghz(random.Random(1))
    d = D.distribution(p, basis(p.n_qubits))
    assert d == pytest.approx({"0" * p.n_clbits: 0.5, "1" * p.n_clbits: 0.5})


def test_teleportation_moves_the_state_to_the_last_qubit():
    p = G.teleport(random.Random(2))
    theta = p.stmts[0].params[0]
    d = D.distribution(p, basis(3))
    p_c2_is_1 = sum(v for k, v in d.items() if k[0] == "1")          # c[2] is the leftmost character
    assert p_c2_is_1 == pytest.approx(math.sin(theta / 2) ** 2, abs=1e-9)


def test_reset_returns_the_qubit_to_zero():
    prog = Program(1, 1, [Gate("x", [], (0,)), Reset(0), Measure(0, 0)])
    assert D.distribution(prog, basis(1)) == pytest.approx({"0": 1.0})


def test_for_loop_repeats_its_body():
    prog = Program(1, 1, [For(2, [Gate("rx", [math.pi / 2], (0,))]), Measure(0, 0)])      # two quarter turns = X
    assert D.distribution(prog, basis(1)) == pytest.approx({"1": 1.0})


def test_classical_condition_correlates_two_qubits():
    prog = Program(2, 2, [Gate("h", [], (0,)), Measure(0, 0), If(0, True, [Gate("x", [], (1,))]), Measure(1, 1)])
    assert D.distribution(prog, basis(2)) == pytest.approx({"00": 0.5, "11": 0.5})


def test_qasm_round_trip_is_stable():
    for fam in G.FAMILIES.values():
        p = fam(random.Random(5))
        assert D.to_qasm(D.parse(D.to_qasm(p))) == D.to_qasm(p)


# ------------------------------------------------------------------ acceptance
def test_validate_stages():
    ref = G.teleport(random.Random(3))
    text = D.to_qasm(ref)
    assert D.validate(text, ref)["status"] == "PASS"
    assert D.validate("OPENQASM 3.0; garbage", ref)["failed_stage"] == "syntax"
    fewer = D.Program(ref.n_qubits, ref.n_clbits - 1, [])
    assert D.validate(D.to_qasm(fewer), ref)["failed_stage"] == "structural"
    far = D.parse(text)
    far.stmts.insert(-1, If(0, True, [Gate("cx", [], (0, 2))]))      # non-adjacent gate hidden in a block
    assert D.validate(D.to_qasm(far), ref)["failed_stage"] == "connectivity"
    bad = D.parse(text)
    bad.stmts = [s for s in bad.stmts if not isinstance(s, If)]
    assert D.validate(D.to_qasm(bad), ref)["failed_stage"] == "semantic"


# ------------------------------------------------------------------ the v3 benchmark
@pytest.fixture(scope="module")
def v3(root):
    path = os.path.join(root, "data_v3", "metadata.csv")
    assert os.path.exists(path), "run `python src/generate_v3.py`"
    return pd.read_csv(path)


def test_v3_design_is_balanced(v3):
    assert len(v3) == 600 and (v3.groupby("bug_class").size() == 100).all()
    assert set(v3.bug_class) == set(G.CLASSES)
    assert (v3[v3.bug_class == "multi_fault"].n_faults >= 2).all()


def test_v3_references_accept_themselves_and_every_fault_is_observable(v3, root):
    sample = pd.concat([g.sample(5, random_state=0) for _, g in v3.groupby("bug_class")])
    for m in sample.to_dict("records"):
        ref_t = open(os.path.join(root, m["reference_file"]), encoding="utf-8").read()
        bug_t = open(os.path.join(root, m["buggy_file"]), encoding="utf-8").read()
        ref = D.parse(ref_t)
        assert D.validate(ref_t, ref)["status"] == "PASS"
        assert D.validate(bug_t, ref)["failed_stage"] == "semantic", m["circuit_id"]


def test_some_v3_faults_are_invisible_to_a_unitary_check(v3, root):
    m = v3[v3.fault_kinds == "swap_measure_bits"].iloc[0]
    ref = D.parse(open(os.path.join(root, m["reference_file"]), encoding="utf-8").read())
    bug = D.parse(open(os.path.join(root, m["buggy_file"]), encoding="utf-8").read())
    assert process_fidelity(D.unitary_view(ref), D.unitary_view(bug)) > 0.999      # unitary check is blind
    assert D.validate(D.to_qasm(bug), ref)["status"] == "FAIL"                      # the distribution check is not


@pytest.mark.parametrize("kind", ["swap_measure_bits", "delete_if", "delete_reset", "loop_count"])
def test_repair_fixes_one_fault_of_each_new_kind(v3, root, kind):
    m = v3[(v3.fault_kinds == kind)].iloc[0]
    ref_t = open(os.path.join(root, m["reference_file"]), encoding="utf-8").read()
    bug_t = open(os.path.join(root, m["buggy_file"]), encoding="utf-8").read()
    res = D.repair(bug_t, ref_t, budget=3000, seed_key=m["circuit_id"])
    assert res["success"], (kind, res["history"])
    assert D.validate(res["qasm"], D.parse(ref_t))["status"] == "PASS"


def test_guided_search_beats_random_search_on_a_small_slice(v3, root):
    sl = pd.concat([g.head(3) for _, g in v3.groupby("bug_class")])
    wins = {"guided": 0, "random": 0}
    for m in sl.to_dict("records"):
        ref_t = open(os.path.join(root, m["reference_file"]), encoding="utf-8").read()
        bug_t = open(os.path.join(root, m["buggy_file"]), encoding="utf-8").read()
        wins["guided"] += D.repair(bug_t, ref_t, budget=3000, seed_key=m["circuit_id"])["success"]
        wins["random"] += D.random_repair(bug_t, ref_t, budget=3000, seed=1)["success"]
    assert wins["guided"] > wins["random"], wins


# ------------------------------------------------------------------ product surface
def test_cli_layer_handles_dynamic_programs(v3, root):
    m = v3[v3.fault_kinds == "delete_reset"].iloc[0]
    ref_t = open(os.path.join(root, m["reference_file"]), encoding="utf-8").read()
    bug_t = open(os.path.join(root, m["buggy_file"]), encoding="utf-8").read()
    assert any(f.code == "I302" for f in product.lint(bug_t))
    assert product.verify(bug_t, reference=ref_t)["failed_stage"] == "semantic"
    rep = product.repair(bug_t, reference=ref_t)
    assert rep.status == "PASS" and "reset" in rep.diff
    with pytest.raises(ValueError, match="reference"):
        product.verify(bug_t, tests={"num_qubits": 1, "tests": [{"input": "0", "expect": {"0": 1}}]})


def test_dynamic_lint_flags_a_condition_that_reads_an_unwritten_bit():
    qasm = ('OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nbit[2] c;\n'
            "if (c[0] == true) {\n  x q[1];\n}\nc[1] = measure q[1];\n")
    assert any(f.code == "W401" for f in product.lint(qasm))
