"""Benchmark v2 generator (src/generate_dataset.py).

Design rules that distinguish v2 from the v1 benchmark (see docs/BENCHMARK_V2.md):

1. Every ground-truth circuit is *hardware admissible* under the validator's own
   physical checks. Device-agnostic circuits that need routing (QFT, QAOA with a
   ring edge, ...) are routed with swap-in/swap-back and inverse-cancelled, which
   preserves the unitary exactly.
2. Every injected fault is *observable* by the acceptance predicate:
     syntax            -> candidate does not parse
     structural/semantic -> unitary differs from the reference (process
                           fidelity < 0.99) while the circuit stays admissible
     gate_redundancy   -> an adjacent self-inverse pair is present that
                           inverse cancellation removes, restoring the reference
     physics_violation -> connectivity or coherence check fails; semantics are
                           already correct
   Faults that fail this test are re-sampled, never emitted.
3. Each record carries ``fault_mode`` and ``mode_family`` ("legacy" = modes
   inherited from the v1 injector, "extended" = added in v2) so that results can
   be reported per mode rather than only per class.

Outputs: data/buggy/*.qasm, data/ground_truth/*.qasm, data/metadata.csv
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import Callable, Dict, List

import pandas as pd
import qiskit.qasm3
from qiskit import QuantumCircuit

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.physics_model import PhysicsModel, get_default_physics_model
from src.routing import cancel_inverses, route_and_cancel
from src.sim import process_fidelity

RANDOM_SEED = 42
OBSERVABLE_BELOW = 0.99          # process fidelity under which a fault is observable
CIRCUIT_TYPES = ["ghz", "qaoa_maxcut", "vqe_h2", "qft"]
BUG_TYPES = ["syntax", "structural", "gate_redundancy", "semantic_deviation", "physics_violation"]


# --------------------------------------------------------------------------- #
# Device-agnostic ("logical") circuit families
# --------------------------------------------------------------------------- #
def build_ghz(n: int, rng: random.Random, long_range: bool) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"ghz_{n}q")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(0, i + 1) if long_range else qc.cx(i, i + 1)
    return qc


def build_qaoa(n: int, rng: random.Random, long_range: bool) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"qaoa_{n}q")
    for i in range(n):
        qc.h(i)
    edges = {(i, i + 1) for i in range(n - 1)}
    if long_range or (n > 3 and rng.random() > 0.5):
        edges.add((0, n - 1))
    for _ in range(rng.choice([1, 2])):
        gamma = round(rng.uniform(0.1, math.pi), 4)
        beta = round(rng.uniform(0.1, math.pi / 2), 4)
        for u, v in sorted(edges):
            qc.cx(u, v)
            qc.rz(gamma, v)
            qc.cx(u, v)
        for i in range(n):
            qc.rx(2 * beta, i)
    return qc


def build_vqe(n: int, rng: random.Random, long_range: bool) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"vqe_h2_{n}q")
    qc.x(0)
    qc.x(1)
    angles = [round(rng.uniform(-math.pi, math.pi), 4) for _ in range(2 * n)]
    for i in range(n):
        qc.ry(angles[i], i)
    pairs = [(0, i) for i in range(1, n)] if long_range else [(i, i + 1) for i in range(n - 1)]
    for a, b in pairs:
        qc.cx(a, b)
    for i in range(n):
        qc.rz(angles[n + i], i)
    for a, b in pairs:
        qc.cx(a, b)
    return qc


def build_qft(n: int, rng: random.Random, long_range: bool) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"qft_{n}q")
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(math.pi / (2 ** (k - j)), k, j)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


BUILDERS: Dict[str, Callable] = {
    "ghz": build_ghz, "qaoa_maxcut": build_qaoa, "vqe_h2": build_vqe, "qft": build_qft,
}


def sample_qubits(c_type: str, rng: random.Random) -> int:
    return {"ghz": lambda: rng.choice([3, 4, 5, 6, 7]),
            "qaoa_maxcut": lambda: rng.choice([4, 5, 6, 7, 8]),
            "vqe_h2": lambda: 4,
            "qft": lambda: rng.choice([4, 5, 6])}[c_type]()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _copy_with(qc: QuantumCircuit, index: int, operation=None, qubits=None) -> QuantumCircuit:
    out = qc.copy()
    inst = out.data[index]
    out.data[index] = inst.replace(
        operation=operation if operation is not None else inst.operation,
        qubits=qubits if qubits is not None else inst.qubits,
    )
    return out


def _qidx(qc: QuantumCircuit, inst) -> List[int]:
    return [qc.find_bit(q).index for q in inst.qubits]


def _param_gates(qc: QuantumCircuit, max_qubits: int = 2) -> List[int]:
    return [i for i, ins in enumerate(qc.data)
            if len(ins.operation.params) > 0 and len(ins.qubits) <= max_qubits
            and ins.operation.name not in ("delay",)]


def _with_param(op, value: float):
    new = op.copy()
    new.params = [round(value, 4)] + list(op.params[1:])
    return new


# --------------------------------------------------------------------------- #
# Fault injectors. Each returns (buggy_circuit_or_qasm, description, fault_mode,
# mode_family) or None when the drawn fault is not applicable to this circuit.
# --------------------------------------------------------------------------- #
def _semantic_single(ref: QuantumCircuit, rng: random.Random, mode: str):
    idxs = _param_gates(ref)
    if not idxs:
        # parameter-free circuits (GHZ): replace a single-qubit Clifford
        singles = [i for i, ins in enumerate(ref.data) if len(ins.qubits) == 1]
        if not singles:
            return None
        i = rng.choice(singles)
        ins = ref.data[i]
        q = _qidx(ref, ins)[0]
        tmp = QuantumCircuit(ref.num_qubits)
        (tmp.z if ins.operation.name == "x" else tmp.x)(q)
        buggy = _copy_with(ref, i, operation=tmp.data[0].operation)
        return buggy, (f"Semantic deviation: replaced {ins.operation.name} with "
                       f"{tmp.data[0].operation.name} on qubit {q}"), "gate_replace", "legacy"

    i = rng.choice(idxs)
    ins = ref.data[i]
    name = ins.operation.name
    orig = float(ins.operation.params[0])
    single = len(ins.qubits) == 1
    if mode == "negate":
        new_op = _with_param(ins.operation, -orig)
        desc, fam = f"Semantic deviation: inverted angle {orig} -> {-orig:.4f} on {name}", "legacy"
    elif mode == "shift_pi2":
        new_op = _with_param(ins.operation, orig + math.pi / 2)
        desc, fam = f"Semantic deviation: shifted angle by pi/2 on {name}", "legacy"
    elif mode == "axis_change" and single and name in ("rx", "ry", "rz"):
        alt = {"rx": "rz", "ry": "rx", "rz": "rx"}[name]
        tmp = QuantumCircuit(ref.num_qubits)
        getattr(tmp, alt)(orig, _qidx(ref, ins)[0])
        new_op = tmp.data[0].operation
        desc, fam = f"Semantic deviation: altered rotation axis {name} -> {alt}", "legacy"
    elif mode == "scale":
        new_op = _with_param(ins.operation, 1.5 * orig)
        desc, fam = f"Semantic deviation: scaled angle by 1.5 on {name}", "extended"
    elif mode == "shift_pi3":
        new_op = _with_param(ins.operation, orig + math.pi / 3)
        desc, fam = f"Semantic deviation: shifted angle by pi/3 on {name}", "extended"
    else:
        return None
    return _copy_with(ref, i, operation=new_op), desc, mode, fam


def _semantic_compound(ref: QuantumCircuit, rng: random.Random):
    idxs = _param_gates(ref)
    if len(idxs) < 2:
        singles = [i for i, ins in enumerate(ref.data) if len(ins.qubits) == 1]
        if len(singles) < 2:
            return None
        a, b = rng.sample(singles, 2)
        out = ref.copy()
        for i in (a, b):
            ins = out.data[i]
            tmp = QuantumCircuit(ref.num_qubits)
            q = _qidx(ref, ins)[0]
            (tmp.z if ins.operation.name == "x" else tmp.x)(q)
            out.data[i] = ins.replace(operation=tmp.data[0].operation)
        return out, "Semantic deviation: two simultaneous gate replacements", "compound", "extended"
    a, b = rng.sample(idxs, 2)
    out = ref.copy()
    for i in (a, b):
        ins = out.data[i]
        orig = float(ins.operation.params[0])
        new_val = -orig if rng.random() < 0.5 else orig + math.pi / 2
        out.data[i] = ins.replace(operation=_with_param(ins.operation, new_val))
    return out, "Semantic deviation: two simultaneous angle faults", "compound", "extended"


def inject_semantic(ref: QuantumCircuit, rng: random.Random):
    mode = rng.choice(["negate", "shift_pi2", "axis_change", "scale", "shift_pi3", "compound"])
    if mode == "compound":
        return _semantic_compound(ref, rng)
    return _semantic_single(ref, rng, mode)


def inject_structural(ref: QuantumCircuit, rng: random.Random):
    mode = rng.choice(["remove_gate", "invert_cnot", "wrong_qubit", "twoqubit_replace"])
    n = ref.num_qubits
    if mode == "remove_gate":
        i = rng.randrange(len(ref.data))
        name = ref.data[i].operation.name
        out = ref.copy()
        del out.data[i]
        return out, f"Structural error: omitted {name} gate at index {i}", mode, "legacy"
    if mode == "invert_cnot":
        idx = [i for i, ins in enumerate(ref.data) if ins.operation.name == "cx"]
        if not idx:
            return None
        i = rng.choice(idx)
        ins = ref.data[i]
        out = _copy_with(ref, i, qubits=(ins.qubits[1], ins.qubits[0]))
        return out, "Structural error: reversed control and target on cx", mode, "legacy"
    if mode == "wrong_qubit":
        idx = [i for i, ins in enumerate(ref.data) if len(ins.qubits) == 1]
        if not idx or n < 2:
            return None
        i = rng.choice(idx)
        ins = ref.data[i]
        cur = _qidx(ref, ins)[0]
        new = (cur + 1) % n
        out = _copy_with(ref, i, qubits=(ref.qubits[new],))
        return out, f"Structural error: moved {ins.operation.name} from qubit {cur} to {new}", mode, "legacy"
    # extended: replace a cx by cz (a two-qubit substitution)
    idx = [i for i, ins in enumerate(ref.data) if ins.operation.name == "cx"]
    if not idx:
        return None
    i = rng.choice(idx)
    tmp = QuantumCircuit(2)
    tmp.cz(0, 1)
    out = _copy_with(ref, i, operation=tmp.data[0].operation)
    return out, "Structural error: cx replaced by cz", mode, "extended"


def inject_redundancy(ref: QuantumCircuit, rng: random.Random):
    kind = rng.choice(["xx", "hh", "cxcx"])
    n = ref.num_qubits
    pos = rng.randint(0, len(ref.data))
    block = QuantumCircuit(n)
    if kind == "xx":
        q = rng.randrange(n)
        block.x(q); block.x(q)
        desc = f"Gate redundancy: two consecutive X on qubit {q}"
    elif kind == "hh":
        q = rng.randrange(n)
        block.h(q); block.h(q)
        desc = f"Gate redundancy: two consecutive H on qubit {q}"
    else:
        block.cx(0, 1); block.cx(0, 1)
        desc = "Gate redundancy: two duplicate cx on (0, 1)"
    out = ref.copy()
    data = list(out.data)
    data[pos:pos] = list(block.data)
    out.data = data
    return out, desc, kind, "legacy"


def inject_physics(logical: QuantumCircuit, ref: QuantumCircuit, rng: random.Random):
    """Returns (buggy, description, mode, family, ground_truth)."""
    mode = rng.choice(["unmapped", "coherence_delay"])
    if mode == "unmapped":
        return logical, "Physics violation: device-agnostic circuit needs routing", mode, "legacy", ref
    out = ref.copy()
    out.delay(1200, rng.randrange(ref.num_qubits), unit="us")
    return out, "Physics violation: 1200us idle delay exceeds coherence budget", mode, "legacy", ref


def _syntax(ref: QuantumCircuit, rng: random.Random):
    text = qiskit.qasm3.dumps(ref)
    lines = text.split("\n")
    mode = rng.choice(["missing_semicolon", "corrupt_keyword", "illegal_bracket", "invalid_gate"])
    n = ref.num_qubits
    if mode == "missing_semicolon":
        for i, line in enumerate(lines):
            if ";" in line and not line.startswith(("OPENQASM", "include")):
                lines[i] = line.rstrip().rstrip(";")
                desc = f"Syntax error: missing semicolon on line {i + 1}"
                break
        else:
            return None
    elif mode == "corrupt_keyword":
        for i, line in enumerate(lines):
            if "qubit[" in line:
                lines[i] = line.replace("qubit[", "qbit_invalid[")
                desc = f"Syntax error: unknown keyword on line {i + 1}"
                break
        else:
            return None
    elif mode == "illegal_bracket":
        for i, line in enumerate(lines):
            if "]" in line:
                lines[i] = line.replace("]", "", 1)
                desc = f"Syntax error: unclosed bracket on line {i + 1}"
                break
        else:
            return None
    else:
        pos = len(lines) - 2 if len(lines) > 2 else 0
        lines.insert(pos, f"unknown_quantum_op q[{rng.randrange(n)}];")
        desc = "Syntax error: undefined gate 'unknown_quantum_op'"
    return "\n".join(lines), desc, mode, "legacy"


# --------------------------------------------------------------------------- #
# Record construction with observability guarantees
# --------------------------------------------------------------------------- #
def _parses(text: str) -> bool:
    try:
        qiskit.qasm3.loads(text)
        return True
    except Exception:
        return False


def is_observable(bug_type: str, buggy, ref: QuantumCircuit, physics: PhysicsModel) -> bool:
    """The acceptance-predicate view of 'this fault actually exists'."""
    if bug_type == "syntax":
        return isinstance(buggy, str) and not _parses(buggy)
    if not isinstance(buggy, QuantumCircuit):
        return False
    conn, _ = physics.validate_circuit_connectivity(buggy)
    coh = physics.is_coherence_valid(buggy)
    if bug_type == "physics_violation":
        return (not conn) or (not coh)
    if not (conn and coh):
        return False                      # logical faults must stay admissible
    if bug_type == "gate_redundancy":
        cancelled = cancel_inverses(buggy)
        return (len(cancelled.data) == len(ref.data)
                and process_fidelity(cancelled, ref) > 0.999)
    # structural / semantic
    if buggy.num_qubits != ref.num_qubits:
        return False
    return process_fidelity(buggy, ref) < OBSERVABLE_BELOW


def make_record(c_type: str, bug_type: str, seed: int, physics: PhysicsModel,
                max_tries: int = 60):
    rng = random.Random(seed)
    n = sample_qubits(c_type, rng)
    builder = BUILDERS[c_type]
    for _ in range(max_tries):
        if bug_type == "physics_violation":
            logical = builder(n, rng, True)
        else:
            logical = builder(n, rng, False)
        ref = route_and_cancel(logical, physics)
        gt = ref
        if bug_type == "syntax":
            res = _syntax(ref, rng)
        elif bug_type == "structural":
            res = inject_structural(ref, rng)
        elif bug_type == "gate_redundancy":
            res = inject_redundancy(ref, rng)
        elif bug_type == "semantic_deviation":
            res = inject_semantic(ref, rng)
        else:
            res = inject_physics(logical, ref, rng)
            if res is not None:
                *res, gt = res
        if res is None:
            continue
        buggy, desc, mode, family = res
        if is_observable(bug_type, buggy, gt, physics):
            return n, buggy, gt, desc, mode, family
    raise RuntimeError(f"could not produce an observable {bug_type} fault for {c_type} (seed {seed})")


def generate_dataset(output_base: str = PROJECT_ROOT, total_circuits: int = 1000,
                     seed: int = RANDOM_SEED) -> pd.DataFrame:
    dirs = {k: os.path.join(output_base, "data", k) for k in ("buggy", "ground_truth")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    physics = get_default_physics_model()
    per_family = total_circuits // len(CIRCUIT_TYPES)
    records, idx = [], 1
    for c_type in CIRCUIT_TYPES:
        for b in range(per_family):
            bug_type = BUG_TYPES[b % len(BUG_TYPES)]
            cid = f"circuit_{idx:04d}"
            n, buggy, gt, desc, mode, family = make_record(
                c_type, bug_type, seed * 100003 + idx, physics)
            bpath, gpath = f"data/buggy/{cid}_buggy.qasm", f"data/ground_truth/{cid}_ground_truth.qasm"
            btxt = buggy if isinstance(buggy, str) else qiskit.qasm3.dumps(buggy)
            with open(os.path.join(output_base, bpath), "w", encoding="utf-8") as f:
                f.write(btxt)
            with open(os.path.join(output_base, gpath), "w", encoding="utf-8") as f:
                f.write(qiskit.qasm3.dumps(gt))
            records.append({
                "circuit_id": cid, "circuit_type": c_type, "num_qubits": n,
                "bug_type": bug_type, "fault_mode": mode, "mode_family": family,
                "buggy_file": bpath, "ground_truth_file": gpath,
                "bug_description": desc, "ref_gates": len(gt.data),
            })
            idx += 1
        print(f"  {c_type}: done ({per_family} circuits)")
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_base, "data", "metadata.csv"), index=False)
    print(f"Generated {len(df)} circuits -> {os.path.join(output_base, 'data', 'metadata.csv')}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=PROJECT_ROOT, help="project root to write data/ into")
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args()
    generate_dataset(a.out, a.total, a.seed)
