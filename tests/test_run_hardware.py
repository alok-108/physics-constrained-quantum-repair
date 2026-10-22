"""Hardware-run script: metric maths and the noisy-simulator path (no account needed)."""

import pytest
from qiskit import QuantumCircuit

from src import run_hardware as H


def test_tvd_and_ideal_distribution():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    ideal = H.ideal_distribution(qc)
    assert ideal == pytest.approx({"00": 0.5, "11": 0.5})
    assert H.tvd({"00": 50, "11": 50}, ideal) == pytest.approx(0.0)
    assert H.tvd({"01": 100}, ideal) == pytest.approx(1.0)
    assert H.tvd({"00": 100}, ideal) == pytest.approx(0.5)


def test_real_device_mode_refuses_without_a_token(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="IBM_QUANTUM_TOKEN"):
        H.run(["circuit_0009"], "ibm_sherbrooke", dry_run=False, shots=10)


def test_simulator_run_shows_the_repair_restores_the_output():
    df = H.run(["circuit_0009"], "unused", dry_run=True, shots=500)
    t = df.set_index("variant").tvd_to_ideal
    assert t["repaired"] == pytest.approx(t["reference"], abs=0.05)
    assert t["buggy"] > t["reference"] + 0.2
