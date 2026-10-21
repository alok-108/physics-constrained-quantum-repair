"""Batched statevector simulation for small circuits (src/sim.py).

The behavioural oracle is queried hundreds of times per repair, so the search
signal is computed with a lightweight NumPy simulator instead of constructing a
Qiskit ``Statevector`` per query. Gate matrices come from Qiskit
(``Gate.to_matrix``), so the gate set is exactly Qiskit's; only the contraction
is custom. Qubit ordering follows Qiskit (little endian).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

Op = Tuple[np.ndarray, Tuple[int, ...]]

# Instructions with no effect on the ideal unitary.
_NON_UNITARY = {"delay", "barrier", "id", "measure", "snapshot"}


def ops_from_circuit(qc: QuantumCircuit) -> List[Op]:
    """Flatten a circuit into (matrix, qubit-index tuple) pairs."""
    ops: List[Op] = []
    for ins in qc.data:
        op = ins.operation
        if op.name in _NON_UNITARY:
            continue
        qubits = tuple(qc.find_bit(q).index for q in ins.qubits)
        try:
            params = tuple(float(p) for p in op.params)
            mat = _cached_matrix(op.name, params, op)
        except Exception:
            mat = np.asarray(Operator(op).data, dtype=complex)
        ops.append((mat, qubits))
    return ops


_MATRIX_CACHE: dict = {}


def _cached_matrix(name: str, params: Tuple[float, ...], op) -> np.ndarray:
    key = (name, params)
    mat = _MATRIX_CACHE.get(key)
    if mat is None:
        mat = np.asarray(op.to_matrix(), dtype=complex)
        if len(_MATRIX_CACHE) < 50000:
            _MATRIX_CACHE[key] = mat
    return mat


def ops_with_index(qc: QuantumCircuit) -> Tuple[List[Op], List[int]]:
    """Like ``ops_from_circuit`` but also returns each op's index in ``qc.data``."""
    ops: List[Op] = []
    idx: List[int] = []
    for i, ins in enumerate(qc.data):
        op = ins.operation
        if op.name in _NON_UNITARY:
            continue
        qubits = tuple(qc.find_bit(q).index for q in ins.qubits)
        try:
            mat = _cached_matrix(op.name, tuple(float(p) for p in op.params), op)
        except Exception:
            mat = np.asarray(Operator(op).data, dtype=complex)
        ops.append((mat, qubits))
        idx.append(i)
    return ops, idx


def _apply(state: np.ndarray, mat: np.ndarray, qubits: Sequence[int], n: int) -> np.ndarray:
    """Apply a k-qubit matrix to a batch of states shaped (2,)*n + (m,)."""
    k = len(qubits)
    u = mat.reshape((2,) * (2 * k))
    # matrix index bit j corresponds to qubits[j]; C-order axis 0 is the MSB.
    in_axes_u = [k + a for a in range(k)]
    state_axes = [n - 1 - qubits[k - 1 - a] for a in range(k)]
    out = np.tensordot(u, state, axes=(in_axes_u, state_axes))
    # tensordot puts the k output axes first, in U's out-axis order 0..k-1,
    # which correspond to qubits[k-1], ..., qubits[0].
    dest = [n - 1 - qubits[k - 1 - a] for a in range(k)]
    return np.moveaxis(out, list(range(k)), dest)


def evolve(ops: Sequence[Op], init: np.ndarray, n: int) -> np.ndarray:
    """Evolve a batch of states. ``init`` has shape (2**n, m)."""
    m = init.shape[1]
    state = init.reshape((2,) * n + (m,))
    for mat, qubits in ops:
        state = _apply(state, mat, qubits, n)
    return state.reshape(2 ** n, m)


def evolve_circuit(qc: QuantumCircuit, init: np.ndarray) -> np.ndarray:
    return evolve(ops_from_circuit(qc), init, qc.num_qubits)


def overlaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column |<a_j|b_j>|^2."""
    return np.abs(np.sum(a.conj() * b, axis=0)) ** 2


def process_fidelity(a: QuantumCircuit, b: QuantumCircuit) -> float:
    """|tr(U^dagger V)|^2 / d^2: exact unitary agreement up to global phase."""
    if a.num_qubits != b.num_qubits:
        return 0.0
    u = Operator(a).data
    v = Operator(b).data
    d = u.shape[0]
    return float(np.abs(np.trace(u.conj().T @ v)) ** 2 / d ** 2)


def random_product_states(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random single-qubit product states, shape (2**n, count)."""
    out = np.zeros((2 ** n, count), dtype=complex)
    for j in range(count):
        vec = np.array([1.0 + 0j])
        for _ in range(n):
            q = rng.normal(size=2) + 1j * rng.normal(size=2)
            q /= np.linalg.norm(q)
            vec = np.kron(q, vec)        # little endian: newest qubit is MSB
        out[:, j] = vec
    return out


def random_states(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random (generally entangled) states, shape (2**n, count)."""
    z = rng.normal(size=(2 ** n, count)) + 1j * rng.normal(size=(2 ** n, count))
    return z / np.linalg.norm(z, axis=0, keepdims=True)


def forward_states(ops: Sequence[Op], init: np.ndarray, n: int) -> List[np.ndarray]:
    """States after each prefix: element p is the state after the first p ops."""
    m = init.shape[1]
    state = init.reshape((2,) * n + (m,))
    out = [state]
    for mat, qubits in ops:
        state = _apply(state, mat, qubits, n)
        out.append(state)
    return out


def backward_states(ops: Sequence[Op], target: np.ndarray, n: int) -> List[np.ndarray]:
    """Element p is (ops[p] ... ops[-1])^dagger applied to ``target``."""
    m = target.shape[1]
    state = target.reshape((2,) * n + (m,))
    out = [state]
    for mat, qubits in reversed(ops):
        state = _apply(state, mat.conj().T, qubits, n)
        out.append(state)
    out.reverse()
    return out


def mean_overlap(a: np.ndarray, b: np.ndarray, n: int) -> float:
    m = a.shape[-1]
    return float(overlaps(a.reshape(2 ** n, m), b.reshape(2 ** n, m)).mean())


# --------------------------------------------------------------------------- #
# Scalable semantic check (no dense unitary): sampled average fidelity
# --------------------------------------------------------------------------- #
def sampled_process_fidelity(a: QuantumCircuit, b: QuantumCircuit, n_states: int = 24,
                             seed: int = 0, batch: int = 4):
    """Estimate the process fidelity of two circuits without building 2^n x 2^n matrices.

    For Haar-random input states psi the mean of |<psi| U_a^dag U_b |psi>|^2 equals the
    average gate fidelity F_avg, and F_pro = ((d + 1) F_avg - 1) / d. Returns
    ``(F_pro_estimate, worst_state_overlap, standard_error)``.

    One-sided evidence: a state with low overlap *proves* the circuits differ;
    passing only supports equivalence statistically (it can miss faults that act
    non-trivially on a vanishing fraction of states). Cost is O(n_states * 2^n * gates).
    """
    if a.num_qubits != b.num_qubits:
        return 0.0, 0.0, 0.0
    n = a.num_qubits
    d = 2 ** n
    rng = np.random.default_rng(seed)
    ops_a, ops_b = ops_from_circuit(a), ops_from_circuit(b)
    vals = []
    for start in range(0, n_states, batch):
        m = min(batch, n_states - start)
        init = random_states(n, m, rng)
        vals.extend(overlaps(evolve(ops_a, init, n), evolve(ops_b, init, n)).tolist())
    arr = np.asarray(vals)
    f_avg = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    f_pro = ((d + 1) * f_avg - 1.0) / d
    return float(f_pro), float(arr.min()), (d + 1) / d * se
