"""Dataset Generation Pipeline for Physics-Constrained Quantum Circuit Repair.

Generates 1,000 valid quantum circuits across 4 families:
  - GHZ state preparation (3-7 qubits)
  - QAOA MaxCut ansatz (4-8 qubits)
  - VQE H2 ground state ansatz (4 qubits)
  - Quantum Fourier Transform (4-6 qubits)

Injects exactly one bug per circuit across 5 categories:
  1. Syntax error
  2. Structural error
  3. Gate redundancy
  4. Semantic deviation
  5. Physics violation

Outputs:
  - data/valid/*.qasm
  - data/buggy/*.qasm
  - data/ground_truth/*.qasm
  - data/metadata.csv
"""

from __future__ import annotations

import os
import random
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from qiskit import QuantumCircuit
import qiskit.qasm3
from qiskit.circuit.library import QFT, TwoLocal

from physics_model import PhysicsModel, get_default_physics_model


# Seed for reproducible benchmark generation
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_ghz_circuit(num_qubits: int, seed: int) -> QuantumCircuit:
    """Generate a GHZ state preparation circuit."""
    qc = QuantumCircuit(num_qubits, name=f"ghz_{num_qubits}q")
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def build_qaoa_maxcut_circuit(num_qubits: int, seed: int) -> QuantumCircuit:
    """Generate a QAOA MaxCut ansatz circuit with random graph topology."""
    rng = random.Random(seed)
    qc = QuantumCircuit(num_qubits, name=f"qaoa_{num_qubits}q")
    
    # Initial Hadamard superposition
    for i in range(num_qubits):
        qc.h(i)

    # Generate connected-like linear / ring edges plus occasional cross edges
    edges = set()
    for i in range(num_qubits - 1):
        edges.add((i, i + 1))
    if num_qubits > 3 and rng.random() > 0.5:
        edges.add((0, num_qubits - 1))

    # Layers of cost and mixer Hamiltonians
    p_layers = rng.choice([1, 2])
    for _ in range(p_layers):
        gamma = round(rng.uniform(0.1, np.pi), 4)
        beta = round(rng.uniform(0.1, np.pi / 2), 4)
        
        # Problem unitary: e^{-i gamma C}
        for u, v in edges:
            qc.cx(u, v)
            qc.rz(gamma, v)
            qc.cx(u, v)
            
        # Mixer unitary: e^{-i beta B}
        for i in range(num_qubits):
            qc.rx(2 * beta, i)

    return qc


def build_vqe_h2_circuit(num_qubits: int, seed: int) -> QuantumCircuit:
    """Generate a parameterized VQE ansatz for molecular hydrogen (H2)."""
    rng = random.Random(seed)
    qc = QuantumCircuit(num_qubits, name=f"vqe_h2_{num_qubits}q")
    
    # Hartree-Fock initial state |1100>
    qc.x(0)
    qc.x(1)

    # Entangling excitation layers (Hardware-efficient / UCCSD-inspired)
    angles = [round(rng.uniform(-np.pi, np.pi), 4) for _ in range(num_qubits * 2)]
    for i in range(num_qubits):
        qc.ry(angles[i], i)
    
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        
    for i in range(num_qubits):
        qc.rz(angles[num_qubits + i], i)
        
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    return qc


def build_qft_circuit(num_qubits: int, seed: int) -> QuantumCircuit:
    """Generate a Quantum Fourier Transform circuit."""
    qc = QuantumCircuit(num_qubits, name=f"qft_{num_qubits}q")
    
    for j in range(num_qubits):
        qc.h(j)
        for k in range(j + 1, num_qubits):
            angle = np.pi / (2 ** (k - j))
            qc.cp(angle, k, j)
            
    # Swap registers at the end
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - 1 - i)
        
    return qc


def inject_bug(
    qc: QuantumCircuit, 
    bug_type: str, 
    physics: PhysicsModel, 
    seed: int
) -> Tuple[str, QuantumCircuit, str]:
    """Inject exactly one bug into a copy of the circuit.
    
    Returns:
        (buggy_qasm_str, ground_truth_qc, bug_description)
    """
    rng = random.Random(seed)
    ground_truth_qc = qc.copy()
    n_qubits = qc.num_qubits

    if bug_type == "syntax":
        # Convert valid circuit to QASM 3.0 string, then inject textual syntax error
        valid_qasm = qiskit.qasm3.dumps(qc)
        lines = valid_qasm.split("\n")
        syntax_modes = ["missing_semicolon", "corrupt_keyword", "illegal_bracket", "invalid_gate"]
        mode = rng.choice(syntax_modes)

        if mode == "missing_semicolon":
            # Remove semicolon from first gate line found
            for idx, line in enumerate(lines):
                if ";" in line and not line.strip().startswith("OPENQASM") and not line.strip().startswith("include"):
                    lines[idx] = line.rstrip().rstrip(";")
                    desc = f"Syntax error: missing semicolon on line {idx + 1}"
                    break
            else:
                lines[-1] = lines[-1].rstrip(";")
                desc = "Syntax error: missing semicolon at end of file"
        elif mode == "corrupt_keyword":
            # Corrupt 'qubit' declaration or 'include'
            for idx, line in enumerate(lines):
                if "qubit[" in line:
                    lines[idx] = line.replace("qubit[", "qbit_invalid[")
                    desc = f"Syntax error: unknown keyword 'qbit_invalid' on line {idx + 1}"
                    break
            else:
                lines[0] = "OPENQASM_BROKEN 3.0;"
                desc = "Syntax error: malformed header declaration"
        elif mode == "illegal_bracket":
            # Remove a closing bracket
            for idx, line in enumerate(lines):
                if "]" in line:
                    lines[idx] = line.replace("]", "", 1)
                    desc = f"Syntax error: unclosed bracket on line {idx + 1}"
                    break
            else:
                lines[-1] += " {"
                desc = "Syntax error: unclosed curly brace"
        else:
            # Insert nonexistent gate
            insert_idx = len(lines) - 2 if len(lines) > 2 else 0
            lines.insert(insert_idx, f"unknown_quantum_op q[{rng.randint(0, n_qubits - 1)}];")
            desc = "Syntax error: call to undefined gate 'unknown_quantum_op'"

        buggy_qasm = "\n".join(lines)
        return buggy_qasm, ground_truth_qc, desc

    elif bug_type == "structural":
        buggy_qc = qc.copy()
        structural_modes = ["remove_gate", "invert_cnot", "wrong_qubit"]
        mode = rng.choice(structural_modes)

        if mode == "remove_gate" and len(buggy_qc.data) > 1:
            # Delete one entangling or rotation gate
            target_idx = rng.randint(0, len(buggy_qc.data) - 1)
            removed_op = buggy_qc.data[target_idx].operation.name
            del buggy_qc.data[target_idx]
            desc = f"Structural error: omitted {removed_op} gate at instruction index {target_idx}"
        elif mode == "invert_cnot":
            # Invert control and target of a two-qubit gate
            cx_indices = [idx for idx, inst in enumerate(buggy_qc.data) if len(inst.qubits) == 2]
            if cx_indices:
                target_idx = rng.choice(cx_indices)
                inst = buggy_qc.data[target_idx]
                q0, q1 = inst.qubits[0], inst.qubits[1]
                # Replace with inverted qubits
                buggy_qc.data[target_idx] = inst.replace(qubits=(q1, q0))
                desc = f"Structural error: reversed control and target qubits on {inst.operation.name} gate"
            else:
                # Fallback: remove first gate
                del buggy_qc.data[0]
                desc = "Structural error: omitted gate at index 0"
        else:
            # Apply single-qubit gate to wrong qubit
            single_indices = [idx for idx, inst in enumerate(buggy_qc.data) if len(inst.qubits) == 1]
            if single_indices and n_qubits > 1:
                target_idx = rng.choice(single_indices)
                inst = buggy_qc.data[target_idx]
                curr_q = buggy_qc.find_bit(inst.qubits[0]).index
                new_q = (curr_q + 1) % n_qubits
                buggy_qc.data[target_idx] = inst.replace(qubits=(buggy_qc.qubits[new_q],))
                desc = f"Structural error: shifted {inst.operation.name} from qubit {curr_q} to qubit {new_q}"
            else:
                del buggy_qc.data[-1]
                desc = "Structural error: omitted final gate"

        buggy_qasm = qiskit.qasm3.dumps(buggy_qc)
        return buggy_qasm, ground_truth_qc, desc

    elif bug_type == "gate_redundancy":
        buggy_qc = qc.copy()
        redundancy_type = rng.choice(["xx", "hh", "cxcx"])
        insert_idx = rng.randint(0, len(buggy_qc.data))
        new_data = list(buggy_qc.data)

        if redundancy_type == "xx":
            q_idx = rng.randint(0, n_qubits - 1)
            temp_qc = QuantumCircuit(n_qubits)
            temp_qc.x(q_idx)
            temp_qc.x(q_idx)
            new_data[insert_idx:insert_idx] = list(temp_qc.data)
            desc = f"Gate redundancy: inserted two consecutive self-canceling X gates on qubit {q_idx}"
        elif redundancy_type == "hh":
            q_idx = rng.randint(0, n_qubits - 1)
            temp_qc = QuantumCircuit(n_qubits)
            temp_qc.h(q_idx)
            temp_qc.h(q_idx)
            new_data[insert_idx:insert_idx] = list(temp_qc.data)
            desc = f"Gate redundancy: inserted two consecutive self-canceling H gates on qubit {q_idx}"
        else:
            q0, q1 = 0, min(1, n_qubits - 1)
            temp_qc = QuantumCircuit(n_qubits)
            temp_qc.cx(q0, q1)
            temp_qc.cx(q0, q1)
            new_data[insert_idx:insert_idx] = list(temp_qc.data)
            desc = f"Gate redundancy: inserted two duplicate consecutive CNOT gates on ({q0}, {q1})"

        buggy_qc.data = new_data
        buggy_qasm = qiskit.qasm3.dumps(buggy_qc)
        return buggy_qasm, ground_truth_qc, desc

    elif bug_type == "semantic_deviation":
        buggy_qc = qc.copy()
        # Find parameterized gate or rotation
        param_indices = [
            idx for idx, inst in enumerate(buggy_qc.data) 
            if hasattr(inst.operation, "params") and len(inst.operation.params) > 0
        ]

        if param_indices:
            target_idx = rng.choice(param_indices)
            inst = buggy_qc.data[target_idx]
            orig_val = float(inst.operation.params[0])
            num_gate_qubits = len(inst.qubits)
            
            if num_gate_qubits == 1:
                mode = rng.choice(["negate", "shift", "axis_change"])
            else:
                mode = rng.choice(["negate", "shift"])

            if mode == "negate":
                new_val = round(-orig_val, 4)
                new_op = inst.operation.copy()
                new_op.params = [new_val] + list(inst.operation.params[1:])
                buggy_qc.data[target_idx] = inst.replace(operation=new_op)
                desc = f"Semantic deviation: inverted rotation angle from {orig_val} to {new_val} on {inst.operation.name}"
            elif mode == "shift":
                new_val = round(orig_val + np.pi / 2, 4)
                new_op = inst.operation.copy()
                new_op.params = [new_val] + list(inst.operation.params[1:])
                buggy_qc.data[target_idx] = inst.replace(operation=new_op)
                desc = f"Semantic deviation: shifted rotation parameter by pi/2 on {inst.operation.name}"
            else:
                q_idx = buggy_qc.find_bit(inst.qubits[0]).index
                gate_name = inst.operation.name
                alt_name = "rz" if gate_name == "rx" else "rx"
                temp = QuantumCircuit(n_qubits)
                if alt_name == "rz":
                    temp.rz(orig_val, q_idx)
                else:
                    temp.rx(orig_val, q_idx)
                buggy_qc.data[target_idx] = inst.replace(operation=temp.data[0].operation)
                desc = f"Semantic deviation: altered gate basis from {gate_name} to {alt_name}"
        else:
            single_indices = [idx for idx, inst in enumerate(buggy_qc.data) if len(inst.qubits) == 1]
            if single_indices:
                target_idx = rng.choice(single_indices)
                inst = buggy_qc.data[target_idx]
                q_idx = buggy_qc.find_bit(inst.qubits[0]).index
                temp = QuantumCircuit(n_qubits)
                temp.z(q_idx) if inst.operation.name == "x" else temp.x(q_idx)
                buggy_qc.data[target_idx] = inst.replace(operation=temp.data[0].operation)
                desc = f"Semantic deviation: replaced {inst.operation.name} with {temp.data[0].operation.name}"
            else:
                two_indices = [idx for idx, inst in enumerate(buggy_qc.data) if len(inst.qubits) == 2]
                target_idx = rng.choice(two_indices)
                inst = buggy_qc.data[target_idx]
                temp = QuantumCircuit(2)
                temp.cz(0, 1)
                buggy_qc.data[target_idx] = inst.replace(operation=temp.data[0].operation)
                desc = f"Semantic deviation: replaced {inst.operation.name} with cz"

        buggy_qasm = qiskit.qasm3.dumps(buggy_qc)
        return buggy_qasm, ground_truth_qc, desc

    elif bug_type == "physics_violation":
        buggy_qc = qc.copy()
        phys_mode = rng.choice(["disconnected_cnot", "coherence_exceeded"])

        if phys_mode == "disconnected_cnot":
            # Place a CNOT between two non-connected physical qubits on Sherbrooke (e.g. 0 and 2)
            if n_qubits >= 3 and not physics.is_connected(0, 2):
                buggy_qc.cx(0, 2)
                desc = "Physics violation: CNOT between non-connected physical qubits (0, 2) on FakeSherbrooke coupling graph"
            elif n_qubits >= 4 and not physics.is_connected(0, 3):
                buggy_qc.cx(0, 3)
                desc = "Physics violation: CNOT between non-connected physical qubits (0, 3) on FakeSherbrooke coupling graph"
            else:
                temp = QuantumCircuit(3)
                for inst in buggy_qc.data:
                    q_indices = [buggy_qc.find_bit(q).index for q in inst.qubits]
                    temp.append(inst.operation, q_indices)
                temp.cx(0, 2)
                buggy_qc = temp
                desc = "Physics violation: CNOT between non-connected physical qubits (0, 2) on FakeSherbrooke coupling graph"
            buggy_qasm = qiskit.qasm3.dumps(buggy_qc)
            return buggy_qasm, ground_truth_qc, desc
        else:
            # Coherence violation: inject extreme delay or massive idle time exceeding T2 (T2 ~ 150us)
            # A 1200us delay severely exceeds coherence time
            buggy_qc.delay(1200, 0, unit="us")
            desc = "Physics violation: uncalibrated gate delay of 1200us exceeds qubit T2 coherence limit (~150us)"
            buggy_qasm = qiskit.qasm3.dumps(buggy_qc)
            return buggy_qasm, ground_truth_qc, desc

    else:
        raise ValueError(f"Unknown bug type: {bug_type}")


def generate_dataset(
    output_base: str = ".", 
    total_circuits: int = 1000
) -> pd.DataFrame:
    """Generate the complete 1,000 circuit benchmark dataset."""
    valid_dir = os.path.join(output_base, "data", "valid")
    buggy_dir = os.path.join(output_base, "data", "buggy")
    gt_dir = os.path.join(output_base, "data", "ground_truth")

    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(buggy_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    physics = get_default_physics_model()

    circuit_types = ["ghz", "qaoa_maxcut", "vqe_h2", "qft"]
    bug_types = ["syntax", "structural", "gate_redundancy", "semantic_deviation", "physics_violation"]

    # Exactly 250 circuits per family, distributed evenly across 5 bug types (50 each)
    records = []
    circuit_idx = 1

    circuits_per_type = total_circuits // len(circuit_types)

    print(f"Generating {total_circuits} quantum circuits...")

    for c_type in circuit_types:
        for b_idx in range(circuits_per_type):
            bug_type = bug_types[b_idx % len(bug_types)]
            seed = RANDOM_SEED + circuit_idx

            # Build valid circuit
            if c_type == "ghz":
                n_qubits = random.choice([3, 4, 5, 6, 7])
                qc = build_ghz_circuit(n_qubits, seed)
            elif c_type == "qaoa_maxcut":
                n_qubits = random.choice([4, 5, 6, 7, 8])
                qc = build_qaoa_maxcut_circuit(n_qubits, seed)
            elif c_type == "vqe_h2":
                n_qubits = 4
                qc = build_vqe_h2_circuit(n_qubits, seed)
            elif c_type == "qft":
                n_qubits = random.choice([4, 5, 6])
                qc = build_qft_circuit(n_qubits, seed)
            else:
                raise ValueError(c_type)

            circuit_id = f"circuit_{circuit_idx:04d}"
            valid_filename = f"{circuit_id}_valid.qasm"
            buggy_filename = f"{circuit_id}_buggy.qasm"
            gt_filename = f"{circuit_id}_ground_truth.qasm"

            valid_path = os.path.join(valid_dir, valid_filename)
            buggy_path = os.path.join(buggy_dir, buggy_filename)
            gt_path = os.path.join(gt_dir, gt_filename)

            # Dump valid OpenQASM 3.0
            valid_qasm = qiskit.qasm3.dumps(qc)
            with open(valid_path, "w", encoding="utf-8") as f:
                f.write(valid_qasm)

            # Inject bug
            buggy_qasm, gt_qc, bug_desc = inject_bug(qc, bug_type, physics, seed)
            with open(buggy_path, "w", encoding="utf-8") as f:
                f.write(buggy_qasm)

            # Dump ground-truth OpenQASM 3.0
            gt_qasm = qiskit.qasm3.dumps(gt_qc)
            with open(gt_path, "w", encoding="utf-8") as f:
                f.write(gt_qasm)

            records.append({
                "circuit_id": circuit_id,
                "circuit_type": c_type,
                "num_qubits": n_qubits,
                "bug_type": bug_type,
                "valid_file": f"data/valid/{valid_filename}",
                "buggy_file": f"data/buggy/{buggy_filename}",
                "ground_truth_file": f"data/ground_truth/{gt_filename}",
                "bug_description": bug_desc
            })

            circuit_idx += 1

    df = pd.DataFrame(records)
    metadata_csv_path = os.path.join(output_base, "data", "metadata.csv")
    df.to_csv(metadata_csv_path, index=False)
    print(f"Generated {len(df)} circuits. Metadata written to {metadata_csv_path}")
    return df


if __name__ == "__main__":
    import sys
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    generate_dataset(output_base=base_dir, total_circuits=1000)
