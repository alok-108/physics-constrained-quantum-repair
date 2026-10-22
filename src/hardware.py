"""Calibration-based hardware estimate (src/hardware.py).

The acceptance predicate checks adjacency under the identity layout, which is a necessary but
weak notion of "runs on the chip". This module adds a more realistic *estimate* using the
FakeSherbrooke calibration snapshot shipped with qiskit-ibm-runtime (per-gate errors, durations,
T1/T2, readout errors): compile to the device's native basis with a chosen layout strategy, then

* estimated success probability (ESP)  = product over native instructions of (1 - error)
                                          x product over measured qubits of (1 - readout error)
* estimated duration                   = longest dependency path using native gate durations
* coherence ratio                      = duration / min(T1, T2) over the physical qubits used

This is a model of one calibration snapshot of a simulated backend. It is **not** a
measurement on hardware and says nothing about drift, crosstalk or compiler randomness beyond
the fixed seed used here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import qiskit.qasm3
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag

try:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
except ImportError:                                            # pragma: no cover
    from qiskit.providers.fake_provider import FakeSherbrooke

_BACKEND = None


def backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = FakeSherbrooke()
    return _BACKEND


@dataclass
class HardwareEstimate:
    layout: str
    native_depth: int
    native_two_qubit_gates: int
    native_gates: int
    estimated_duration_us: float
    coherence_ratio: float
    esp: float
    physical_qubits: List[int]

    def to_dict(self) -> Dict:
        return asdict(self)


def compile_native(qc: QuantumCircuit, layout: str = "identity", seed: int = 0) -> QuantumCircuit:
    """Compile to FakeSherbrooke's native gates. ``identity`` pins virtual qubit i to physical qubit i
    (the layout the acceptance predicate assumes); ``auto`` lets Qiskit pick a noise-aware layout."""
    be = backend()
    kw = dict(backend=be, seed_transpiler=seed)
    if layout == "identity":
        return transpile(qc, optimization_level=1, initial_layout=list(range(qc.num_qubits)), **kw)
    if layout == "auto":
        return transpile(qc, optimization_level=3, **kw)
    raise ValueError("layout must be 'identity' or 'auto'")


def estimate(qc: QuantumCircuit, layout: str = "identity", seed: int = 0) -> HardwareEstimate:
    """Calibration-based estimate for the circuit as compiled with the chosen layout strategy."""
    be = backend()
    target = be.target
    meas = qc.copy()
    if not any(i.operation.name == "measure" for i in meas.data):
        meas.measure_all()                                       # readout error applies to what is read out
    tc = compile_native(meas, layout, seed)
    phys = sorted({tc.find_bit(q).index for ins in tc.data for q in ins.qubits})
    esp = 1.0
    n2 = 0
    for ins in tc.data:
        name = ins.operation.name
        if name in ("barrier", "delay"):
            continue
        qs = tuple(tc.find_bit(q).index for q in ins.qubits)
        err = 0.0
        try:
            props = target[name][qs]
            err = float(props.error) if props is not None and props.error is not None else 0.0
        except KeyError:
            err = 0.0
        esp *= (1.0 - err)
        n2 += 1 if len(qs) == 2 else 0
    dag = circuit_to_dag(tc)
    finish: Dict = {}
    for node in dag.topological_op_nodes():
        qs = tuple(tc.find_bit(q).index for q in node.qargs)
        dur = 0.0
        try:
            p = target[node.op.name][qs]
            dur = float(p.duration) if p is not None and p.duration is not None else 0.0
        except KeyError:
            dur = 0.0
        start = max([finish.get(q, 0.0) for q in node.qargs] + [0.0])
        for q in node.qargs:
            finish[q] = start + dur
    duration = max(finish.values()) if finish else 0.0
    t_lim = min(min(target.qubit_properties[q].t1 or 1e9, target.qubit_properties[q].t2 or 1e9) for q in phys)
    ngates = sum(1 for i in tc.data if i.operation.name not in ("barrier", "delay"))
    return HardwareEstimate(layout, tc.depth(), n2, ngates, duration * 1e6, duration / t_lim, esp, phys)


def estimate_qasm(qasm: str, layout: str = "identity", seed: int = 0) -> HardwareEstimate:
    return estimate(qiskit.qasm3.loads(qasm), layout, seed)
