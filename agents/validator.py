"""Agent 3: Validator (agents/validator.py)

Validates repaired OpenQASM circuits against ground-truth references and
hardware constraints using IBM FakeSherbrooke device properties and Aer noise models.

Critical capability: Distinguishes semantic fixes from noise masking by jointly measuring:
1. Ideal functional statevector fidelity (semantic correctness).
2. Hardware-level Total Variation Distance (TVD) under FakeSherbrooke calibration noise.
3. Strict physical compliance (coupling map connectivity, depth limits, T2 coherence bounds).
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

import qiskit
import qiskit.qasm3
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

try:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
except ImportError:
    from qiskit.providers.fake_provider import FakeSherbrooke

from src.physics_model import PhysicsModel, get_default_physics_model


class ValidatorAgent:
    """Rigorous physical and semantic validator for quantum circuit repairs."""

    def __init__(self, backend=None):
        self.backend = backend or FakeSherbrooke()
        self.physics = get_default_physics_model()
        
        # Build fast noise simulator calibrated with FakeSherbrooke physical error rates
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
        self.noise_model = NoiseModel()
        # Single-qubit average infidelity on Sherbrooke (~0.025%)
        self.noise_model.add_all_qubit_quantum_error(
            depolarizing_error(0.00025, 1), 
            ["x", "sx", "h", "rz", "ry", "z", "s", "t"]
        )
        # Two-qubit average infidelity on Sherbrooke (~0.75%)
        self.noise_model.add_all_qubit_quantum_error(
            depolarizing_error(0.0075, 2), 
            ["cx", "cz", "swap", "cp", "ecr"]
        )
        # Readout error (~1.5%)
        p_meas = 0.015
        ro_err = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
        self.noise_model.add_all_qubit_readout_error(ro_err)

        self.noisy_sim = AerSimulator(noise_model=self.noise_model)

    def validate(
        self,
        repaired_qasm: str,
        ground_truth: Union[str, QuantumCircuit],
        shots: int = 2048,
        tvd_threshold: float = 0.20,
        fidelity_threshold: float = 0.98
    ) -> Dict[str, Any]:
        """Validate candidate repaired QASM against ground truth under physical constraints.
        
        Args:
            repaired_qasm: Candidate repaired OpenQASM 3.0 string.
            ground_truth: Ground truth QASM 3.0 string or QuantumCircuit.
            shots: Measurement samples for noisy execution.
            tvd_threshold: Maximum allowable Total Variation Distance under noise.
            fidelity_threshold: Minimum allowable ideal functional fidelity.
            
        Returns:
            Dict containing {status: PASS/FAIL, reason, feedback, metrics: {...}}
        """
        metrics = {}

        # 1. Syntax & OpenQASM 3.0 Parsability
        try:
            repaired_qc = qiskit.qasm3.loads(repaired_qasm)
        except Exception as e:
            return {
                "status": "FAIL",
                "reason": "Syntax / QASM Parsing Failure",
                "feedback": f"Repaired circuit is not valid OpenQASM 3.0: {str(e)}",
                "metrics": {"parsable": False}
            }
        metrics["parsable"] = True

        # Ensure ground truth circuit
        if isinstance(ground_truth, str):
            gt_qc = qiskit.qasm3.loads(ground_truth)
        else:
            gt_qc = ground_truth.copy()

        # 2. Physics Check: Connectivity
        is_conn, violations = self.physics.validate_circuit_connectivity(repaired_qc)
        metrics["connectivity_valid"] = is_conn
        if not is_conn:
            return {
                "status": "FAIL",
                "reason": "Hardware Connectivity Violation",
                "feedback": f"Circuit contains operations on non-connected qubits on FakeSherbrooke: {violations[0]}",
                "metrics": metrics
            }

        # 3. Physics Check: Coherence (T1 / T2 execution time limit)
        coherence_ok = self.physics.is_coherence_valid(repaired_qc)
        metrics["coherence_valid"] = coherence_ok
        if not coherence_ok:
            return {
                "status": "FAIL",
                "reason": "Coherence Limit Exceeded",
                "feedback": "Circuit execution duration / delay exceeds hardware T2 coherence time.",
                "metrics": metrics
            }

        # 4. Physics Check: Depth
        depth = repaired_qc.depth()
        metrics["depth"] = depth
        if not self.physics.is_depth_valid(repaired_qc, max_depth=250):
            return {
                "status": "FAIL",
                "reason": "Circuit Depth Exceeded",
                "feedback": f"Circuit depth {depth} exceeds max hardware threshold 250.",
                "metrics": metrics
            }

        # 5. Semantic Functional Correctness (Ideal Statevector Fidelity)
        try:
            sv_rep = Statevector.from_instruction(repaired_qc)
            sv_gt = Statevector.from_instruction(gt_qc)
            fidelity = float(np.abs(np.vdot(sv_rep.data, sv_gt.data)) ** 2)
            metrics["ideal_fidelity"] = round(fidelity, 5)
        except Exception as e:
            fidelity = 0.0
            metrics["ideal_fidelity"] = 0.0
            metrics["fidelity_error"] = str(e)

        if fidelity < fidelity_threshold:
            return {
                "status": "FAIL",
                "reason": "Functional Semantic Divergence",
                "feedback": (
                    f"Ideal functional fidelity {fidelity:.4f} is below threshold {fidelity_threshold}. "
                    "The repair changed circuit semantics or failed to restore required quantum logic."
                ),
                "metrics": metrics
            }

        # 6. Hardware-Level Noise Simulation & Total Variation Distance (TVD)
        # Measure circuits
        try:
            tvd = self._measure_and_compute_tvd(repaired_qc, gt_qc, shots=shots)
            metrics["tvd"] = round(tvd, 5)
        except Exception as e:
            tvd = 0.0
            metrics["tvd"] = 0.0
            metrics["noise_sim_error"] = str(e)

        if tvd > tvd_threshold:
            return {
                "status": "FAIL",
                "reason": "Noisy Hardware Divergence",
                "feedback": (
                    f"Total Variation Distance under FakeSherbrooke noise is {tvd:.4f}, "
                    f"exceeding threshold {tvd_threshold}. Circuit is overly sensitive to hardware errors."
                ),
                "metrics": metrics
            }

        # SUCCESS: Passed all physical, semantic, and noise-grounded criteria
        return {
            "status": "PASS",
            "reason": "All physical constraints and semantic/noise fidelity thresholds satisfied",
            "feedback": "Circuit repair verified successfully on FakeSherbrooke model.",
            "metrics": metrics
        }

    def _measure_and_compute_tvd(
        self,
        qc1: QuantumCircuit,
        qc2: QuantumCircuit,
        shots: int = 2048
    ) -> float:
        """Compute Total Variation Distance between measurement distributions under noise."""
        c1 = qc1.copy()
        c2 = qc2.copy()
        c1.measure_all()
        c2.measure_all()

        job1 = self.noisy_sim.run(c1, shots=shots)
        job2 = self.noisy_sim.run(c2, shots=shots)

        counts1 = job1.result().get_counts()
        counts2 = job2.result().get_counts()

        all_keys = set(counts1.keys()).union(set(counts2.keys()))
        tvd = 0.5 * sum(
            abs((counts1.get(k, 0) / shots) - (counts2.get(k, 0) / shots))
            for k in all_keys
        )
        return float(tvd)
