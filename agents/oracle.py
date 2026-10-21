"""Behavioural specification oracle (agents/oracle.py).

The repair agent is never given the reference circuit. It is given a *test
suite*: a small set of input states on which the intended behaviour is known,
exposed only through a query-counted scalar score. This is the quantum analogue
of test-suite-guided program repair.

Three disjoint state sets are held privately:

* ``suite``     -- guides the search. |0...0> plus a few Haar-random product
                   states. Deliberately *weaker* than the acceptance predicate.
* ``pool``      -- candidate counterexamples. When the acceptance predicate
                   rejects a candidate on semantic grounds, the orchestrator may
                   (diagnostic-feedback arms only) ask the oracle for the pool
                   state on which the candidate disagrees most; that state joins
                   the suite (counterexample-guided repair).
* ``holdout``   -- never used by the search or the feedback. ``holdout_gap``
                   compares suite agreement with agreement on these states and
                   with exact process fidelity, quantifying overfitting.

Every ``score`` / ``counterexample`` call consumes budget, so search cost is a
reported property of the method rather than a hidden subsidy.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit

from src.edits import Edit
from src.sim import (_apply, _cached_matrix, backward_states, evolve, forward_states,
                     mean_overlap, ops_from_circuit, ops_with_index, overlaps,
                     process_fidelity, random_product_states, random_states)


def _seed_from(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


class BehaviouralOracle:
    """Query-counted behavioural specification. Exposes scores, never source."""

    def __init__(self, reference: QuantumCircuit, query_budget: int = 2000,
                 suite_size: int = 4, pool_size: int = 16, holdout_size: int = 8,
                 seed_key: str = "oracle"):
        self._ref = reference.copy()                  # private; never returned
        self._n = self._ref.num_qubits
        rng = np.random.default_rng(_seed_from(seed_key))

        zero = np.zeros((2 ** self._n, 1), dtype=complex)
        zero[0, 0] = 1.0
        suite = np.concatenate(
            [zero, random_product_states(self._n, max(suite_size - 1, 0), rng)], axis=1)
        pool = random_states(self._n, pool_size, rng)
        holdout = random_states(self._n, holdout_size, rng)

        ref_ops = ops_from_circuit(self._ref)
        self._suite_in = suite
        self._suite_ref = evolve(ref_ops, suite, self._n)
        self._pool_in = pool
        self._pool_ref = evolve(ref_ops, pool, self._n)
        self._hold_in = holdout
        self._hold_ref = evolve(ref_ops, holdout, self._n)

        self.query_budget = query_budget
        self.queries_used = 0
        self.counterexamples_used = 0

    @classmethod
    def from_tests(cls, inputs: np.ndarray, expected: np.ndarray, query_budget: int = 2000,
                   seed_key: str = "tests") -> "BehaviouralOracle":
        """Oracle backed by user-supplied (input state, expected output state) pairs.

        ``inputs`` and ``expected`` have shape (2**n, m). There is no reference circuit, so
        there is no counterexample pool, no held-out set and no exact process fidelity.
        """
        inputs = np.asarray(inputs, dtype=complex)
        expected = np.asarray(expected, dtype=complex)
        if inputs.shape != expected.shape or inputs.ndim != 2:
            raise ValueError("inputs and expected must both have shape (2**n, m)")
        n = int(round(np.log2(inputs.shape[0])))
        if 2 ** n != inputs.shape[0]:
            raise ValueError("state length must be a power of two")
        self = cls.__new__(cls)
        self._ref = None
        self._n = n
        self._suite_in, self._suite_ref = inputs, expected
        empty = np.zeros((2 ** n, 0), dtype=complex)
        self._pool_in = self._pool_ref = self._hold_in = self._hold_ref = empty
        self.query_budget = query_budget
        self.queries_used = 0
        self.counterexamples_used = 0
        return self

    # ------------------------------------------------------------------ #
    @property
    def num_qubits(self) -> int:
        """Register width is part of the declared specification."""
        return self._n

    def exhausted(self) -> bool:
        return self.queries_used >= self.query_budget

    def remaining(self) -> int:
        return max(0, self.query_budget - self.queries_used)

    def _agreement(self, ops, inputs: np.ndarray, refs: np.ndarray) -> np.ndarray:
        return overlaps(refs, evolve(ops, inputs, self._n))

    # ------------------------------------------------------------------ #
    def score_ops(self, ops) -> float:
        """Mean agreement on the current suite from pre-flattened operations."""
        if self.exhausted():
            return 0.0
        self.queries_used += 1
        try:
            return float(self._agreement(ops, self._suite_in, self._suite_ref).mean())
        except Exception:
            return 0.0

    def score(self, candidate: QuantumCircuit) -> float:
        """Agreement of a candidate with the intended behaviour, in [0, 1]."""
        if candidate.num_qubits != self._n:
            self.queries_used += 1
            return 0.0
        try:
            ops = ops_from_circuit(candidate)
        except Exception:
            self.queries_used += 1
            return 0.0
        return self.score_ops(ops)

    def counterexample(self, candidate: QuantumCircuit) -> Optional[float]:
        """Add the pool state with the worst candidate agreement to the suite.

        Returns the candidate's agreement on that state, or ``None`` when the
        budget is spent or the candidate cannot be evaluated.
        """
        if self.exhausted() or candidate.num_qubits != self._n or self._pool_in.shape[1] == 0:
            return None
        self.queries_used += 1
        try:
            ag = self._agreement(ops_from_circuit(candidate), self._pool_in, self._pool_ref)
        except Exception:
            return None
        worst = int(np.argmin(ag))
        self._suite_in = np.concatenate([self._suite_in, self._pool_in[:, worst:worst + 1]], axis=1)
        self._suite_ref = np.concatenate([self._suite_ref, self._pool_ref[:, worst:worst + 1]], axis=1)
        # a state can only serve as a counterexample once
        self._pool_in = np.delete(self._pool_in, worst, axis=1)
        self._pool_ref = np.delete(self._pool_ref, worst, axis=1)
        self.counterexamples_used += 1
        return float(ag[worst])

    def evaluate_edits(self, base: QuantumCircuit, edits: Sequence[Edit]) -> np.ndarray:
        """Score many single-instruction edits of ``base`` in one call.

        Equivalent to building each edited circuit and calling ``score`` on it
        (same values, same budget accounting: one query per edit), but computed
        by meet-in-the-middle from cached prefix/suffix states so that a full
        neighbourhood costs milliseconds. Edits beyond the remaining budget are
        not evaluated and come back as NaN. Only scalar scores leave the oracle.
        """
        scores = np.full(len(edits), np.nan)
        if not edits or base.num_qubits != self._n:
            return scores
        take = min(len(edits), self.remaining())
        if take <= 0:
            return scores
        self.queries_used += take
        n = self._n
        ops, idx = ops_with_index(base)
        pos_of = {d: k for k, d in enumerate(idx)}
        # number of unitary ops strictly before each data index
        before = np.zeros(len(base.data) + 1, dtype=int)
        c = 0
        for d in range(len(base.data)):
            before[d] = c
            c += 1 if d in pos_of else 0
        before[len(base.data)] = c
        fwd = forward_states(ops, self._suite_in, n)
        bwd = backward_states(ops, self._suite_ref, n)
        for e_i in range(take):
            e = edits[e_i]
            try:
                if e.kind == "insert":
                    k = int(before[e.index])
                    mat = _cached_matrix(e.gate.name, tuple(float(p) for p in e.gate.params), e.gate)
                    st = _apply(fwd[k], mat, e.qubits, n)
                    scores[e_i] = mean_overlap(bwd[k], st, n)
                else:
                    if e.index not in pos_of:           # non-unitary instruction: no effect
                        scores[e_i] = mean_overlap(bwd[len(ops)], fwd[len(ops)], n)
                        continue
                    k = pos_of[e.index]
                    if e.kind == "delete":
                        scores[e_i] = mean_overlap(bwd[k + 1], fwd[k], n)
                    else:
                        mat = _cached_matrix(e.gate.name, tuple(float(p) for p in e.gate.params), e.gate)
                        st = _apply(fwd[k], mat, e.qubits, n)
                        scores[e_i] = mean_overlap(bwd[k + 1], st, n)
            except Exception:
                scores[e_i] = 0.0
        return scores

    # ------------------------------------------------------------------ #
    # Evaluation instruments: not an agent affordance, not charged to budget.
    # ------------------------------------------------------------------ #
    def suite_agreements(self, candidate: QuantumCircuit) -> np.ndarray:
        """Per-test agreement of a candidate on the current suite (not charged to the budget)."""
        return self._agreement(ops_from_circuit(candidate), self._suite_in, self._suite_ref)

    def holdout_gap(self, candidate: QuantumCircuit) -> Dict[str, float]:
        """Suite agreement vs. held-out agreement vs. exact process fidelity."""
        try:
            ops = ops_from_circuit(candidate)
            suite = float(self._agreement(ops, self._suite_in, self._suite_ref).mean())
            if self._ref is None:                      # test-suite oracle: nothing held out
                return {"suite": suite, "holdout": None, "process": None, "gap": None}
            hold = float(self._agreement(ops, self._hold_in, self._hold_ref).mean())
            proc = process_fidelity(candidate, self._ref)
        except Exception:
            return {"suite": 0.0, "holdout": 0.0, "process": 0.0, "gap": 0.0}
        return {"suite": suite, "holdout": hold, "process": proc, "gap": suite - proc}
