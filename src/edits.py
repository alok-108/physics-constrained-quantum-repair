"""Single-instruction circuit edits (src/edits.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction


@dataclass(eq=False)
class Edit:
    """A single-instruction edit of a base circuit, addressed by ``qc.data`` index.

    kind: ``replace`` (swap instruction ``index`` for ``gate`` on ``qubits``),
    ``delete`` (drop instruction ``index``), or ``insert`` (place ``gate`` on
    ``qubits`` immediately before instruction ``index``; ``index == len(data)``
    appends).
    """
    kind: str
    index: int
    gate: Any = None
    qubits: Tuple[int, ...] = ()
    tag: str = ""


def apply_edit(qc: QuantumCircuit, edit: Edit) -> QuantumCircuit:
    """Return a new circuit with ``edit`` applied; ``qc`` is not modified."""
    data = list(qc.data)
    if edit.kind == "delete":
        del data[edit.index]
    else:
        inst = CircuitInstruction(edit.gate, tuple(qc.qubits[q] for q in edit.qubits))
        if edit.kind == "replace":
            data[edit.index] = inst
        elif edit.kind == "insert":
            data.insert(edit.index, inst)
        else:
            raise ValueError(f"unknown edit kind {edit.kind!r}")
    out = qc.copy_empty_like()
    for ins in data:
        out._append(ins)
    return out
