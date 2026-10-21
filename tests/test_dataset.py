"""Benchmark v2 invariants: admissible references, observable faults, balanced design."""

import os

import pytest

from src.audit import audit_dataset, summarise

EXPECTED_COLUMNS = {
    "circuit_id", "circuit_type", "num_qubits", "bug_type", "fault_mode", "mode_family",
    "buggy_file", "ground_truth_file", "bug_description", "ref_gates",
}


def test_metadata_schema_and_balance(metadata):
    assert len(metadata) == 1000
    assert set(metadata.columns) == EXPECTED_COLUMNS
    assert metadata.isnull().sum().sum() == 0
    assert set(metadata.circuit_type) == {"ghz", "qaoa_maxcut", "vqe_h2", "qft"}
    assert (metadata.groupby("bug_type").size() == 200).all()
    assert (metadata.groupby(["circuit_type", "bug_type"]).size() == 50).all()
    assert set(metadata.mode_family) == {"legacy", "extended"}


def test_all_files_exist(metadata, root):
    for col in ("buggy_file", "ground_truth_file"):
        assert all(os.path.isfile(os.path.join(root, p)) for p in metadata[col])


def test_references_admissible_and_faults_observable_on_a_stratified_sample(root):
    res = audit_dataset(root, sample=150)
    s = summarise(res)
    assert s["reference_not_admissible"] == 0, s
    assert s["reference_reducible"] == 0, s
    assert s["faults_not_observable"] == 0, s


def test_v1_failure_mode_is_detected_by_the_audit(root):
    """The audit must flag the defect that motivated v2: QFT references violate connectivity."""
    legacy = os.path.join(root, "legacy_v1")
    if not os.path.exists(os.path.join(legacy, "data", "metadata.csv")):
        pytest.skip("legacy_v1 data not present")
    res = audit_dataset(legacy, sample=100)
    qft = res[res.circuit_type == "qft"]
    assert len(qft) and (~qft.ref_connectivity_ok).all()
