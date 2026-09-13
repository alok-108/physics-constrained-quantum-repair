"""Tests for multi-agent system (tests/test_agents.py).

Verifies feedback loop, Diagnostician, Repairer, Validator, and Orchestrator
across 5 representative sample circuits representing each bug class.
"""

from __future__ import annotations

import os
import sys
import json
import pytest
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")
for p in [SRC_DIR, AGENTS_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from agents.orchestrator import Orchestrator
from agents.diagnostician import DiagnosticianAgent
from agents.repairer import RepairerAgent
from agents.validator import ValidatorAgent


@pytest.fixture(scope="session")
def metadata():
    csv_path = os.path.join(PROJECT_ROOT, "data", "metadata.csv")
    assert os.path.exists(csv_path), "metadata.csv required for test_agents"
    return pd.read_csv(csv_path)


def test_individual_agents():
    """Unit test individual agent interfaces."""
    diag = DiagnosticianAgent()
    repair = RepairerAgent()
    val = ValidatorAgent()

    # Syntax test
    broken_qasm = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\ncx q[0], q[1]\n'  # missing semicolon
    gt_qasm = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\ncx q[0], q[1];\n'

    report = diag.diagnose(broken_qasm)
    assert report["bug_type"] == "syntax"
    assert "suggested_fix" in report

    repaired = repair.repair(broken_qasm, report)
    assert repaired.strip().endswith(";")

    res = val.validate(repaired, gt_qasm)
    assert res["status"] == "PASS"
    assert res["metrics"]["parsable"] is True


def test_orchestrator_feedback_loop_on_5_samples(metadata):
    """Run Orchestrator feedback loop on 5 sample circuits spanning all 5 bug types."""
    bug_types = ["syntax", "structural", "gate_redundancy", "semantic_deviation", "physics_violation"]
    orchestrator = Orchestrator(max_iterations=3)

    sampled_circuits = []
    for b_type in bug_types:
        row = metadata[metadata["bug_type"] == b_type].iloc[0]
        sampled_circuits.append(row)

    assert len(sampled_circuits) == 5

    for row in sampled_circuits:
        c_id = row["circuit_id"]
        bug_type = row["bug_type"]

        buggy_path = os.path.join(PROJECT_ROOT, row["buggy_file"])
        gt_path = os.path.join(PROJECT_ROOT, row["ground_truth_file"])

        with open(buggy_path, "r", encoding="utf-8") as f:
            buggy_qasm = f.read()
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_qasm = f.read()

        spec = {
            "circuit_type": row["circuit_type"],
            "num_qubits": int(row["num_qubits"]),
            "bug_type": bug_type
        }

        result = orchestrator.run(
            circuit_id=c_id,
            buggy_qasm=buggy_qasm,
            ground_truth_qasm=gt_qasm,
            circuit_spec=spec
        )

        assert result["circuit_id"] == c_id
        assert result["status"] == "PASS", f"Failed repair for {c_id} ({bug_type})"
        assert 1 <= result["iterations"] <= 3

        # Verify log file
        log_file = result["log_file"]
        assert os.path.isfile(log_file), f"Log file missing: {log_file}"

        with open(log_file, "r", encoding="utf-8") as f:
            logged_data = json.load(f)

        assert logged_data["circuit_id"] == c_id
        assert len(logged_data["history"]) == result["iterations"]
        for step in logged_data["history"]:
            assert "diagnosis" in step
            assert "repaired_qasm" in step
            assert "validation" in step
