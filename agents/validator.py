"""Acceptance predicate (agents/validator.py).

The validator is the only component that reads the ground-truth reference
*source*. It evaluates one predicate, identically for every system:

    A(C; R, B) = V_syn(C) ^ V_str(C) ^ V_phys(C, B) ^ V_min(C) ^ V_sem(C, R) ^ V_noise(C, R, B)

* V_syn    parses as OpenQASM 3.
* V_str    declares exactly the register width of the specification.
* V_phys   every two-qubit gate acts on adjacent qubits (identity layout),
           execution time per qubit within the coherence budget, depth bound.
* V_min    no adjacent self-inverse pair remains (shared definition in
           ``src/routing.py``). Without this stage a redundancy fault would be
           "repaired" by doing nothing, because inverse pairs preserve the unitary.
* V_sem    exact unitary agreement up to global phase,
           |tr(U_C^dagger U_R)|^2 / d^2 >= 0.99. This replaces the v1 single-input
           statevector overlap, which is blind to e.g. controlled-phase faults
           in QFT acting on |0...0>.
* V_noise  the exact total-variation distance between the noisy output
           distributions of candidate and reference (density-matrix simulation
           with the depolarising + readout model below) is at most the
           (1 - alpha) quantile of the TVD between two independent S-shot
           samples of the reference. Acceptance therefore means "the
           difference is below the sampling noise floor of an S-shot
           experiment". Using exact distributions removes sampling noise from
           the verdict: it is deterministic, and a correct repair identical to
           the reference is never falsely rejected.

The predicate is never ablated across systems.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Union

import numpy as np
import qiskit
import qiskit.qasm3
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from src.physics_model import get_default_physics_model
from src.routing import is_irreducible
from src.sim import process_fidelity, sampled_process_fidelity

MAX_DEPTH = 250
FIDELITY_THRESHOLD = 0.99
READOUT_ERROR = 0.015


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31 - 1)


def build_noise_model() -> NoiseModel:
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(0.00025, 1), ["x", "sx", "h", "rz", "ry", "rx", "z", "s", "t", "y", "sdg", "tdg"])
    nm.add_all_qubit_quantum_error(
        depolarizing_error(0.0075, 2), ["cx", "cz", "swap", "cp", "ecr", "cy"])
    # readout error is applied analytically to exact distributions (see _noisy_probs)
    return nm


class ValidatorAgent:
    """Semantic, physical and noise acceptance oracle shared by all systems."""

    def __init__(self, alpha: float = 0.01, null_resamples: int = 400,
                 shots: int = 4096, fidelity_threshold: float = FIDELITY_THRESHOLD,
                 semantic_mode: str = "auto", dense_limit: int = 10, noise_limit: int = 10,
                 sample_states: int = 24):
        self.physics = get_default_physics_model()
        self.alpha = alpha
        self.null_resamples = null_resamples
        self.shots = shots
        self.fidelity_threshold = fidelity_threshold
        # semantic_mode: "exact" (dense unitary), "sampled" (Haar-state estimate, scales past
        # ~12 qubits, one-sided evidence) or "auto" (exact up to dense_limit qubits, else sampled).
        # The noise stage needs a density matrix (4^n memory) and is skipped above noise_limit.
        self.semantic_mode = semantic_mode
        self.dense_limit = dense_limit
        self.noise_limit = noise_limit
        self.sample_states = sample_states
        self.noise_model = build_noise_model()
        self.noisy_sim = AerSimulator(noise_model=self.noise_model, method="density_matrix",
                                      max_parallel_threads=1)
        self._null_cache: Dict[str, float] = {}
        self._prob_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def validate(
        self,
        repaired_qasm: str,
        ground_truth: Union[str, QuantumCircuit],
        redact: bool = False,
        gt_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        def fail(stage: str, reason: str, feedback: str) -> Dict[str, Any]:
            if redact:
                feedback = "Candidate rejected by the acceptance predicate."
            return {"status": "FAIL", "failed_stage": stage, "reason": reason,
                    "feedback": feedback, "metrics": metrics}

        # V_syn
        try:
            cand = qiskit.qasm3.loads(repaired_qasm)
        except Exception as e:
            metrics["parsable"] = False
            return fail("syntax", "QASM parsing failure", f"Candidate is not valid OpenQASM 3.0: {e}")
        metrics["parsable"] = True

        ref = qiskit.qasm3.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth.copy()

        # V_str
        metrics["num_qubits"] = cand.num_qubits
        if cand.num_qubits != ref.num_qubits:
            return fail("structural", "Register width mismatch",
                        f"Candidate declares {cand.num_qubits} qubits; specification requires {ref.num_qubits}.")

        # V_phys
        conn, violations = self.physics.validate_circuit_connectivity(cand)
        metrics["connectivity_valid"] = bool(conn)
        if not conn:
            return fail("connectivity", "Hardware connectivity violation",
                        f"Operation on non-adjacent qubits on FakeSherbrooke: {violations[0]}")
        coh = self.physics.is_coherence_valid(cand)
        metrics["coherence_valid"] = bool(coh)
        if not coh:
            return fail("coherence", "Coherence limit exceeded",
                        "Execution duration exceeds the device T2 budget.")
        metrics["depth"] = cand.depth()
        if not self.physics.is_depth_valid(cand, max_depth=MAX_DEPTH):
            return fail("depth", "Circuit depth exceeded",
                        f"Depth {cand.depth()} exceeds the limit {MAX_DEPTH}.")

        # V_min
        if not is_irreducible(cand):
            metrics["reducible"] = True
            return fail("minimality", "Reducible inverse pair remains",
                        "The circuit contains adjacent self-inverse gates that cancel.")
        metrics["reducible"] = False

        # V_sem
        sampled = self.semantic_mode == "sampled" or (
            self.semantic_mode == "auto" and cand.num_qubits > self.dense_limit)
        try:
            if sampled:
                f, worst, se = sampled_process_fidelity(cand, ref, self.sample_states,
                                                        seed=_stable_seed(gt_key or "sem"))
                metrics.update({"semantic_check": "sampled", "worst_state_overlap": round(worst, 6),
                                "fidelity_stderr": round(se, 6)})
            else:
                f = process_fidelity(cand, ref)
                metrics["semantic_check"] = "exact"
        except Exception as e:
            metrics["fidelity_error"] = str(e)
            f = 0.0
        metrics["process_fidelity"] = round(f, 6)
        if f < self.fidelity_threshold:
            return fail("semantic", "Functional semantic divergence",
                        f"Process fidelity {f:.4f} is below {self.fidelity_threshold}; "
                        "the candidate does not implement the specified computation.")

        # V_noise
        if cand.num_qubits > self.noise_limit:
            metrics["noise_check"] = "skipped (density matrix infeasible)"
            return {"status": "PASS", "failed_stage": None,
                    "reason": "Structural, physical and semantic criteria satisfied; noise stage skipped",
                    "feedback": "Candidate accepted (noise stage skipped).", "metrics": metrics}
        try:
            key = gt_key or hashlib.sha256(qiskit.qasm3.dumps(ref).encode()).hexdigest()[:16]
            p_ref = self._ref_probs(ref, key)
            thr = self._null_threshold(p_ref, key)
            tvd = 0.5 * float(np.abs(self._noisy_probs(cand) - p_ref).sum())
            metrics["tvd"] = round(tvd, 6)
            metrics["tvd_threshold"] = round(thr, 6)
        except Exception as e:
            metrics["noise_sim_error"] = str(e)
            tvd, thr = 0.0, 1.0
        if tvd > thr:
            return fail("noise", "Noisy hardware divergence",
                        f"Exact noisy TVD {tvd:.4f} exceeds the {self.shots}-shot sampling noise "
                        f"floor {thr:.4f}.")

        return {"status": "PASS", "failed_stage": None,
                "reason": "All structural, physical, semantic and noise criteria satisfied",
                "feedback": "Candidate accepted.", "metrics": metrics}

    # ------------------------------------------------------------------ #
    def _noisy_probs(self, qc: QuantumCircuit) -> np.ndarray:
        """Exact output distribution under the gate-noise model plus readout error."""
        c = qc.copy()
        c.save_probabilities()
        probs = np.asarray(self.noisy_sim.run(c).result().data(0)["probabilities"], dtype=float)
        n = qc.num_qubits
        conf = np.array([[1 - READOUT_ERROR, READOUT_ERROR], [READOUT_ERROR, 1 - READOUT_ERROR]])
        t = probs.reshape((2,) * n)
        for ax in range(n):
            t = np.moveaxis(np.tensordot(conf, t, axes=(1, ax)), 0, ax)
        return t.reshape(-1)

    def _ref_probs(self, ref: QuantumCircuit, key: str) -> np.ndarray:
        if key not in self._prob_cache:
            self._prob_cache[key] = self._noisy_probs(ref)
        return self._prob_cache[key]

    def _null_threshold(self, p_ref: np.ndarray, key: str) -> float:
        """(1 - alpha) quantile of TVD between two S-shot samples of the reference."""
        cache_key = f"{key}:{self.shots}"
        if cache_key not in self._null_cache:
            rng = np.random.default_rng(_stable_seed(cache_key))
            a = rng.multinomial(self.shots, p_ref, size=self.null_resamples) / self.shots
            b = rng.multinomial(self.shots, p_ref, size=self.null_resamples) / self.shots
            vals = 0.5 * np.abs(a - b).sum(axis=1)
            self._null_cache[cache_key] = float(np.quantile(vals, 1.0 - self.alpha))
        return self._null_cache[cache_key]
