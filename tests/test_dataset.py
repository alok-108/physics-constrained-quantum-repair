"""Comprehensive verification test suite for dataset generation, bug injection,
and physical hardware model constraints.
"""

from __future__ import annotations

import os
import sys
import pytest
import pandas as pd
import numpy as np

import qiskit
import qiskit.qasm3
from qiskit.quantum_info import Statevector

# Add src/ to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from physics_model import (
    PhysicsModel,
    get_default_physics_model,
    is_connected,
    get_gate_error,
    get_gate_duration,
    is_depth_valid,
    is_coherence_valid,
)


@pytest.fixture(scope="session")
def metadata_df():
    csv_path = os.path.join(PROJECT_ROOT, "data", "metadata.csv")
    assert os.path.exists(csv_path), f"metadata.csv not found at {csv_path}. Run generate_dataset.py first."
    df = pd.read_csv(csv_path)
    return df


@pytest.fixture(scope="session")
def physics_model():
    return get_default_physics_model()


def test_metadata_complete(metadata_df):
    """Verify metadata.csv has exactly 1000 circuits, correct columns, and balanced categories."""
    assert len(metadata_df) == 1000, f"Expected 1000 records, got {len(metadata_df)}"

    expected_columns = {
        "circuit_id",
        "circuit_type",
        "num_qubits",
        "bug_type",
        "valid_file",
        "buggy_file",
        "ground_truth_file",
        "bug_description",
    }
    assert set(metadata_df.columns) == expected_columns, f"Columns mismatch: {metadata_df.columns}"

    # Assert no null values
    assert metadata_df.isnull().sum().sum() == 0, "Found null values in metadata.csv"

    # Assert circuit types
    assert set(metadata_df["circuit_type"].unique()) == {"ghz", "qaoa_maxcut", "vqe_h2", "qft"}

    # Assert bug types
    expected_bug_types = {"syntax", "structural", "gate_redundancy", "semantic_deviation", "physics_violation"}
    assert set(metadata_df["bug_type"].unique()) == expected_bug_types

    # Assert bug types are evenly represented (200 each)
    for b_type in expected_bug_types:
        count = (metadata_df["bug_type"] == b_type).sum()
        assert count == 200, f"Expected 200 circuits for bug type {b_type}, got {count}"


def test_all_circuit_files_exist(metadata_df):
    """Verify that all generated files in metadata.csv exist on disk."""
    for _, row in metadata_df.iterrows():
        valid_path = os.path.join(PROJECT_ROOT, row["valid_file"])
        buggy_path = os.path.join(PROJECT_ROOT, row["buggy_file"])
        gt_path = os.path.join(PROJECT_ROOT, row["ground_truth_file"])

        assert os.path.isfile(valid_path), f"Valid file missing: {valid_path}"
        assert os.path.isfile(buggy_path), f"Buggy file missing: {buggy_path}"
        assert os.path.isfile(gt_path), f"Ground truth file missing: {gt_path}"


def test_valid_circuits_run(metadata_df):
    """Verify that valid circuits parse cleanly as OpenQASM 3.0 and simulate successfully."""
    # Sample 40 valid circuits (10 per family)
    sample_df = metadata_df.groupby("circuit_type").apply(lambda g: g.head(10)).reset_index(drop=True)

    for _, row in sample_df.iterrows():
        file_path = os.path.join(PROJECT_ROOT, row["valid_file"])
        with open(file_path, "r", encoding="utf-8") as f:
            qasm_str = f.read()

        # Parse QASM 3.0
        qc = qiskit.qasm3.loads(qasm_str)
        assert isinstance(qc, qiskit.QuantumCircuit)
        assert qc.num_qubits == row["num_qubits"]

        # Run statevector simulation
        sv = Statevector.from_instruction(qc)
        norm = np.linalg.norm(sv.data)
        assert np.isclose(norm, 1.0, atol=1e-5), f"Statevector norm not 1: {norm}"


def test_buggy_circuits_syntax_fail(metadata_df):
    """Verify that circuits with syntax bugs fail parsing."""
    syntax_df = metadata_df[metadata_df["bug_type"] == "syntax"].head(20)

    for _, row in syntax_df.iterrows():
        file_path = os.path.join(PROJECT_ROOT, row["buggy_file"])
        with open(file_path, "r", encoding="utf-8") as f:
            qasm_str = f.read()

        with pytest.raises(Exception):
            qiskit.qasm3.loads(qasm_str)


def test_buggy_circuits_physics_fail(metadata_df, physics_model):
    """Verify that physics violation circuits fail hardware constraint validation."""
    phys_df = metadata_df[metadata_df["bug_type"] == "physics_violation"].head(20)

    for _, row in phys_df.iterrows():
        file_path = os.path.join(PROJECT_ROOT, row["buggy_file"])
        with open(file_path, "r", encoding="utf-8") as f:
            qasm_str = f.read()

        qc = qiskit.qasm3.loads(qasm_str)
        is_conn, violations = physics_model.validate_circuit_connectivity(qc)
        coherence_ok = physics_model.is_coherence_valid(qc)

        # Must fail at least one physical constraint (connectivity or coherence)
        assert (not is_conn) or (not coherence_ok), (
            f"Circuit {row['circuit_id']} expected physics violation, but passed: {row['bug_description']}"
        )


def test_buggy_circuits_gate_redundancy(metadata_df):
    """Verify that gate redundancy buggy circuits contain duplicate operations compared to ground truth."""
    red_df = metadata_df[metadata_df["bug_type"] == "gate_redundancy"].head(20)

    for _, row in red_df.iterrows():
        buggy_path = os.path.join(PROJECT_ROOT, row["buggy_file"])
        gt_path = os.path.join(PROJECT_ROOT, row["ground_truth_file"])

        buggy_qc = qiskit.qasm3.load(buggy_path)
        gt_qc = qiskit.qasm3.load(gt_path)

        # Buggy circuit must have strictly more operations than ground truth
        assert len(buggy_qc.data) > len(gt_qc.data), (
            f"Circuit {row['circuit_id']} buggy ops ({len(buggy_qc.data)}) not > gt ({len(gt_qc.data)})"
        )


def test_buggy_circuits_semantic_deviation(metadata_df):
    """Verify that semantic deviation circuits differ in simulation output from ground truth."""
    sem_df = metadata_df[metadata_df["bug_type"] == "semantic_deviation"].head(15)

    for _, row in sem_df.iterrows():
        buggy_path = os.path.join(PROJECT_ROOT, row["buggy_file"])
        gt_path = os.path.join(PROJECT_ROOT, row["ground_truth_file"])

        buggy_qc = qiskit.qasm3.load(buggy_path)
        gt_qc = qiskit.qasm3.load(gt_path)

        sv_buggy = Statevector.from_instruction(buggy_qc)
        sv_gt = Statevector.from_instruction(gt_qc)

        # Statevectors should diverge (fidelity < 0.9999 or difference in probabilities)
        fidelity = np.abs(np.vdot(sv_buggy.data, sv_gt.data)) ** 2
        assert fidelity < 0.9999, f"Circuit {row['circuit_id']} statevectors identical despite semantic change"


def test_physics_model_methods(physics_model):
    """Verify core physics model APIs."""
    # FakeSherbrooke has 127 qubits
    assert physics_model.num_qubits == 127

    # Qubit 0 and 1 are connected on Sherbrooke
    assert is_connected(0, 1) is True
    # Qubits 0 and 100 are not directly connected
    assert is_connected(0, 100) is False

    # Error rates
    err_cx = get_gate_error("cx", (0, 1))
    assert 0.0 < err_cx < 0.1

    err_x = get_gate_error("x", 0)
    assert 0.0 < err_x < 0.01

    # Durations
    dur_x = get_gate_duration("x", 0)
    assert 1e-9 < dur_x < 1e-6

    # Depth check
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    assert is_depth_valid(qc, max_depth=10) is True
    assert is_depth_valid(qc, max_depth=1) is False

    # Coherence check
    assert is_coherence_valid(qc) is True
    qc_long_delay = qiskit.QuantumCircuit(1)
    qc_long_delay.delay(2000, 0, unit="us")  # 2ms > T2 ~ 150us
    assert is_coherence_valid(qc_long_delay) is False
