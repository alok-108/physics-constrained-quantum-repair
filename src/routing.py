"""Routing and inverse-cancellation utilities (src/routing.py).

``route_with_swap_back`` makes every two-qubit operation act on physically
adjacent qubits by swapping operands together, applying the operation, and
swapping back. The resulting circuit implements *exactly* the same unitary as
the input (no final permutation), so routed and unrouted circuits can be
compared by unitary equivalence without layout bookkeeping.

``cancel_inverses`` removes adjacent self-inverse pairs (DAG adjacency) to a
fixed point. It is used both by the repair agent and by the acceptance
predicate's minimality check, so that "no reducible inverse pair remains" is a
single shared definition.
"""

from __future__ import annotations

from typing import Optional, Tuple

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CXGate, CYGate, CZGate, HGate, SwapGate, XGate, YGate, ZGate,
)
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation

from src.physics_model import PhysicsModel, get_default_physics_model

_SELF_INVERSE = [HGate(), XGate(), YGate(), ZGate(), CXGate(), CYGate(), CZGate(), SwapGate()]


def cancel_inverses(qc: QuantumCircuit) -> QuantumCircuit:
    """Cancel adjacent self-inverse gate pairs until none remain."""
    pm = PassManager([InverseCancellation(_SELF_INVERSE)])
    current = qc
    while True:
        reduced = pm.run(current)
        if len(reduced.data) == len(current.data):
            return reduced
        current = reduced


def is_irreducible(qc: QuantumCircuit) -> bool:
    return len(cancel_inverses(qc).data) == len(qc.data)


def route_with_swap_back(
    qc: QuantumCircuit, physics: Optional[PhysicsModel] = None
) -> QuantumCircuit:
    """Return a unitary-equal circuit whose 2-qubit gates are all adjacent.

    Paths are restricted to the circuit's own register; if the device graph
    offers no such path the operation is left unrouted (and will be rejected by
    the connectivity check rather than silently altered).
    """
    physics = physics or get_default_physics_model()
    n = qc.num_qubits
    sub = physics.graph.subgraph(range(n))
    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    for ins in qc.data:
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if len(qs) == 2 and not physics.is_connected(qs[0], qs[1]):
            try:
                path = nx.shortest_path(sub, qs[0], qs[1])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                out.append(ins.operation, [out.qubits[q] for q in qs], ins.clbits)
                continue
            for k in range(len(path) - 2):
                out.swap(path[k], path[k + 1])
            out.append(ins.operation, [out.qubits[path[-2]], out.qubits[path[-1]]])
            for k in reversed(range(len(path) - 2)):
                out.swap(path[k], path[k + 1])
        else:
            out.append(ins.operation, [out.qubits[q] for q in qs], ins.clbits)
    return out


def route_and_cancel(qc: QuantumCircuit, physics: Optional[PhysicsModel] = None) -> QuantumCircuit:
    """Route, then cancel the redundant swap-in/swap-back pairs routing creates."""
    return cancel_inverses(route_with_swap_back(qc, physics))


def circuit_key(qc: QuantumCircuit) -> Tuple:
    """Hashable structural signature used for tabu bookkeeping."""
    return tuple(
        (ins.operation.name,
         tuple(qc.find_bit(q).index for q in ins.qubits),
         tuple(round(float(p), 6) for p in ins.operation.params
               if isinstance(p, (int, float))))
        for ins in qc.data
    )
