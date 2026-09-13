"""Physics Model for IBM FakeSherbrooke backend.

Extracts device topology, calibration metrics (T1, T2, gate error rates, gate durations),
and validates quantum circuits against physical hardware constraints.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple, Union
import networkx as nx
from qiskit import QuantumCircuit
import qiskit.qasm3

try:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
except ImportError:
    try:
        from qiskit.providers.fake_provider import FakeSherbrooke
    except ImportError:
        # Fallback for alternative packaging
        from qiskit_aer.backends.fake_provider import FakeSherbrooke


class PhysicsModel:
    """Hardware physics constraint engine backed by IBM FakeSherbrooke (127 qubits)."""

    def __init__(self, backend=None):
        self.backend = backend or FakeSherbrooke()
        self.num_qubits = self.backend.num_qubits
        self.target = self.backend.target
        
        # Build coupling graph
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        self.coupling_map: List[Tuple[int, int]] = []
        
        if hasattr(self.backend, "coupling_map") and self.backend.coupling_map is not None:
            cmap = self.backend.coupling_map
            edges = cmap.get_edges() if hasattr(cmap, "get_edges") else list(cmap)
            for src, dst in edges:
                self.graph.add_edge(src, dst)
                self.directed_graph.add_edge(src, dst)
                self.coupling_map.append((src, dst))
        elif hasattr(self.target, "build_coupling_map"):
            cmap = self.target.build_coupling_map()
            for src, dst in cmap.get_edges():
                self.graph.add_edge(src, dst)
                self.directed_graph.add_edge(src, dst)
                self.coupling_map.append((src, dst))

        # Extract T1 and T2 coherence times
        self.t1_times: Dict[int, float] = {}
        self.t2_times: Dict[int, float] = {}
        self._extract_coherence_times()

    def _extract_coherence_times(self) -> None:
        """Extracts T1 and T2 coherence times for all qubits on the backend."""
        for q in range(self.num_qubits):
            t1 = None
            t2 = None
            if self.target.qubit_properties and q < len(self.target.qubit_properties):
                q_prop = self.target.qubit_properties[q]
                if q_prop:
                    t1 = getattr(q_prop, "t1", None)
                    t2 = getattr(q_prop, "t2", None)

            # Fallback values typical for IBM Eagle / Sherbrooke (T1 ~ 250us, T2 ~ 150us)
            self.t1_times[q] = float(t1) if t1 is not None else 2.5e-4
            self.t2_times[q] = float(t2) if t2 is not None else 1.5e-4

    def is_connected(self, q1: int, q2: int) -> bool:
        """Check if physical connectivity exists between qubit q1 and q2 on the device."""
        if q1 == q2:
            return True
        return self.graph.has_edge(q1, q2)

    def get_gate_error(self, gate: str, qubits: Union[int, Tuple[int, ...], List[int]]) -> float:
        """Extract average gate error rate (infidelity) for given gate and qubit(s).
        
        Args:
            gate: Gate operation name (e.g., 'x', 'sx', 'rz', 'cx', 'ecr', 'cz').
            qubits: Qubit index or tuple of qubit indices.
            
        Returns:
            Error rate as a float in [0.0, 1.0].
        """
        gate = gate.lower()
        if isinstance(qubits, int):
            qargs = (qubits,)
        else:
            qargs = tuple(qubits)

        # Standardize 2-qubit gate aliases
        gate_candidates = [gate]
        if gate in ("cx", "cnot"):
            gate_candidates.extend(["cx", "ecr", "cz"])
        elif gate == "ecr":
            gate_candidates.extend(["ecr", "cx"])

        for g in gate_candidates:
            if g in self.target:
                op_map = self.target[g]
                if qargs in op_map and op_map[qargs] is not None:
                    err = op_map[qargs].error
                    if err is not None:
                        return float(err)
                # Check reversed order for two-qubit symmetric queries
                if len(qargs) == 2:
                    rev_qargs = (qargs[1], qargs[0])
                    if rev_qargs in op_map and op_map[rev_qargs] is not None:
                        err = op_map[rev_qargs].error
                        if err is not None:
                            return float(err)

        # Realistic fallback defaults if not recorded in target
        if len(qargs) == 1:
            return 2.5e-4  # ~0.025% single qubit error
        return 7.5e-3      # ~0.75% two-qubit gate error

    def get_gate_duration(self, gate: str, qubits: Union[int, Tuple[int, ...], List[int]]) -> float:
        """Extract physical gate duration in seconds for given gate and qubit(s).
        
        Args:
            gate: Gate operation name.
            qubits: Qubit index or tuple of qubit indices.
            
        Returns:
            Duration in seconds (e.g. 35ns for single qubit, 300-500ns for two-qubit).
        """
        gate = gate.lower()
        if isinstance(qubits, int):
            qargs = (qubits,)
        else:
            qargs = tuple(qubits)

        gate_candidates = [gate]
        if gate in ("cx", "cnot"):
            gate_candidates.extend(["cx", "ecr"])
        elif gate == "ecr":
            gate_candidates.extend(["ecr", "cx"])

        for g in gate_candidates:
            if g in self.target:
                op_map = self.target[g]
                if qargs in op_map and op_map[qargs] is not None:
                    dur = op_map[qargs].duration
                    if dur is not None:
                        return float(dur)
                if len(qargs) == 2:
                    rev_qargs = (qargs[1], qargs[0])
                    if rev_qargs in op_map and op_map[rev_qargs] is not None:
                        dur = op_map[rev_qargs].duration
                        if dur is not None:
                            return float(dur)

        # Realistic hardware defaults
        if len(qargs) == 1:
            return 3.5e-8  # 35 ns
        return 3.2e-7      # 320 ns

    def _ensure_circuit(self, circuit: Union[QuantumCircuit, str]) -> QuantumCircuit:
        """Parse OpenQASM 3.0 string or file path if not already a QuantumCircuit."""
        if isinstance(circuit, QuantumCircuit):
            return circuit
        if isinstance(circuit, str):
            if os.path.exists(circuit):
                with open(circuit, "r", encoding="utf-8") as f:
                    content = f.read()
                return qiskit.qasm3.loads(content)
            return qiskit.qasm3.loads(circuit)
        raise TypeError(f"Expected QuantumCircuit or QASM str, got {type(circuit)}")

    def is_depth_valid(self, circuit: Union[QuantumCircuit, str], max_depth: int = 200) -> bool:
        """Check if circuit depth is within the feasible hardware execution threshold.
        
        Args:
            circuit: QuantumCircuit or OpenQASM 3.0 string/file.
            max_depth: Maximum allowable circuit depth.
            
        Returns:
            bool: True if depth <= max_depth, False otherwise.
        """
        try:
            qc = self._ensure_circuit(circuit)
            return qc.depth() <= max_depth
        except Exception:
            return False

    def is_coherence_valid(self, circuit: Union[QuantumCircuit, str], margin: float = 0.5) -> bool:
        """Evaluate if cumulative circuit execution duration on every qubit stays safely within coherence limits.
        
        Args:
            circuit: QuantumCircuit or OpenQASM 3.0 string/file.
            margin: Safety factor fraction of min(T1, T2) allowable for execution.
            
        Returns:
            bool: True if execution time on all qubits < margin * min(T1, T2).
        """
        try:
            qc = self._ensure_circuit(circuit)
        except Exception:
            return False

        qubit_time: Dict[int, float] = {q: 0.0 for q in range(qc.num_qubits)}

        for instruction in qc.data:
            op = instruction.operation
            gate_name = op.name.lower()
            q_indices = [qc.find_bit(q).index for q in instruction.qubits]

            # Special delay handling
            if gate_name == "delay":
                dur = getattr(op, "duration", 0.0)
                unit = getattr(op, "unit", "s")
                if unit == "ns":
                    dur_s = dur * 1e-9
                elif unit == "us":
                    dur_s = dur * 1e-6
                elif unit == "ms":
                    dur_s = dur * 1e-3
                elif unit == "dt":
                    dur_s = dur * (self.target.dt or 2.222e-10)
                else:
                    dur_s = float(dur)
            else:
                dur_s = self.get_gate_duration(gate_name, tuple(q_indices))

            for q in q_indices:
                qubit_time[q] = qubit_time.get(q, 0.0) + dur_s

        for q_idx, total_time in qubit_time.items():
            # Map logical qubit index to physical qubit if available, else identity
            phys_q = q_idx if q_idx < self.num_qubits else (q_idx % self.num_qubits)
            limit = margin * min(self.t1_times.get(phys_q, 2.5e-4), self.t2_times.get(phys_q, 1.5e-4))
            if total_time >= limit:
                return False

        return True

    def validate_circuit_connectivity(self, circuit: Union[QuantumCircuit, str]) -> Tuple[bool, List[str]]:
        """Validate whether all two-qubit operations in the circuit conform to device connectivity.
        
        Returns:
            (is_valid, list_of_violations)
        """
        try:
            qc = self._ensure_circuit(circuit)
        except Exception as e:
            return False, [f"Failed to parse circuit: {e}"]

        violations = []
        for instruction in qc.data:
            if len(instruction.qubits) == 2:
                q1 = qc.find_bit(instruction.qubits[0]).index
                q2 = qc.find_bit(instruction.qubits[1]).index
                if not self.is_connected(q1, q2):
                    violations.append(f"Qubit pair ({q1}, {q2}) for gate '{instruction.operation.name}' not connected on hardware.")

        return len(violations) == 0, violations


# Module-level singleton instance and convenience functions
_DEFAULT_MODEL: Optional[PhysicsModel] = None


def get_default_physics_model() -> PhysicsModel:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = PhysicsModel()
    return _DEFAULT_MODEL


def is_connected(q1: int, q2: int) -> bool:
    return get_default_physics_model().is_connected(q1, q2)


def get_gate_error(gate: str, qubits: Union[int, Tuple[int, ...], List[int]]) -> float:
    return get_default_physics_model().get_gate_error(gate, qubits)


def get_gate_duration(gate: str, qubits: Union[int, Tuple[int, ...], List[int]]) -> float:
    return get_default_physics_model().get_gate_duration(gate, qubits)


def is_depth_valid(circuit: Union[QuantumCircuit, str], max_depth: int = 200) -> bool:
    return get_default_physics_model().is_depth_valid(circuit, max_depth)


def is_coherence_valid(circuit: Union[QuantumCircuit, str], margin: float = 0.5) -> bool:
    return get_default_physics_model().is_coherence_valid(circuit, margin)
