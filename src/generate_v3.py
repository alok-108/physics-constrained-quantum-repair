"""Benchmark v3 generator: dynamic circuits and richer faults (src/generate_v3.py).

v2 faults are single-instruction edits of unitary circuits. v3 adds what real programs have:
mid-circuit measurement, reset, classical feed-forward and loops, and faults that live there
(measurement wiring, wrong classical conditions, missing resets, wrong loop bounds), plus
programs with two or three simultaneous faults. These faults are invisible to a unitary check
and are observable only through the classical output distribution.

Families (all device-adjacent on a line):  ghz (3-5 qubits, terminal measurement),
teleport (3 qubits, two feed-forward corrections), parity (3 qubits, one ancilla reused with
reset), loop (3 qubits, a constant-body for loop).

Every fault is verified observable: the buggy program's exact output distribution differs from
the reference's on at least one of 16 fixed inputs (TVD > 1e-3).

    python src/generate_v3.py            # writes data_v3/ (600 programs, deterministic)
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from typing import Callable, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import dynamic as D
from src.dynamic import For, Gate, If, Measure, Program, Reset

SEED = 42
OBSERVABLE_TVD = 1e-3
CLASSES = ["wiring", "control_flow", "reset_fault", "loop_fault", "gate_fault", "multi_fault"]
PER_CLASS = {
    "wiring": [("ghz", 34), ("teleport", 33), ("parity", 33)],
    "control_flow": [("teleport", 100)],
    "reset_fault": [("parity", 100)],
    "loop_fault": [("loop", 100)],
    "gate_fault": [("ghz", 50), ("loop", 50)],
    "multi_fault": [("ghz", 25), ("teleport", 25), ("parity", 25), ("loop", 25)],
}


# --------------------------------------------------------------------------- #
# reference programs
# --------------------------------------------------------------------------- #
def ghz(rng: random.Random) -> Program:
    n = rng.choice([3, 4, 5])
    st = [Gate("h", [], (0,))] + [Gate("cx", [], (i, i + 1)) for i in range(n - 1)]
    st += [Measure(i, i) for i in range(n)]
    return Program(n, n, st)


def teleport(rng: random.Random) -> Program:
    th = round(rng.uniform(0.3, 2.8), 4)
    st = [Gate("ry", [th], (0,)), Gate("h", [], (1,)), Gate("cx", [], (1, 2)), Gate("cx", [], (0, 1)),
          Gate("h", [], (0,)), Measure(0, 0), Measure(1, 1),
          If(1, True, [Gate("x", [], (2,))]), If(0, True, [Gate("z", [], (2,))]), Measure(2, 2)]
    return Program(3, 3, st)


def parity(rng: random.Random) -> Program:
    a, b = round(rng.uniform(0.4, 2.6), 4), round(rng.uniform(0.4, 2.6), 4)
    st = [Gate("ry", [a], (0,)), Gate("ry", [b], (2,)),
          Gate("cx", [], (0, 1)), Gate("cx", [], (2, 1)), Measure(1, 0), Reset(1),
          Gate("cx", [], (0, 1)), Gate("cx", [], (2, 1)), Measure(1, 1),
          Measure(0, 2), Measure(2, 3)]
    return Program(3, 4, st)


def loop(rng: random.Random) -> Program:
    a, b = round(rng.uniform(0.2, 1.2), 4), round(rng.uniform(0.2, 1.2), 4)
    r = rng.choice([2, 3, 4])
    st = [Gate("h", [], (0,)), For(r, [Gate("rz", [a], (0,)), Gate("cx", [], (0, 1)), Gate("rx", [b], (2,))]),
          Gate("cx", [], (1, 2)), Measure(0, 0), Measure(1, 1), Measure(2, 2)]
    return Program(3, 3, st)


FAMILIES: Dict[str, Callable[[random.Random], Program]] = {"ghz": ghz, "teleport": teleport, "parity": parity, "loop": loop}


# --------------------------------------------------------------------------- #
# fault generators: (program, rng) -> (program, description, kind) or None
# --------------------------------------------------------------------------- #
def _nodes(prog: Program, kind):
    return [(path, ) for path in D.enumerate_paths(prog.stmts)
            if isinstance(D._container(prog, path)[0][D._container(prog, path)[1]], kind)]


def _pick(prog: Program, kind, rng: random.Random):
    paths = [p for (p,) in _nodes(prog, kind)]
    return rng.choice(paths) if paths else None


def _at(prog: Program, path):
    lst, i = D._container(prog, path)
    return lst, i


def f_swap_measure_bits(prog, rng):
    ms = [p for (p,) in _nodes(prog, Measure)]
    if len(ms) < 2:
        return None
    p1, p2 = rng.sample(ms, 2)
    q = copy.deepcopy(prog)
    (l1, i1), (l2, i2) = _at(q, p1), _at(q, p2)
    l1[i1].c, l2[i2].c = l2[i2].c, l1[i1].c
    return q, "two measurements write to each other's classical bit", "swap_measure_bits"


def f_swap_measure_qubits(prog, rng):
    ms = [p for (p,) in _nodes(prog, Measure)]
    if len(ms) < 2:
        return None
    p1, p2 = rng.sample(ms, 2)
    q = copy.deepcopy(prog)
    (l1, i1), (l2, i2) = _at(q, p1), _at(q, p2)
    l1[i1].q, l2[i2].q = l2[i2].q, l1[i1].q
    return q, "two measurements read each other's qubit", "swap_measure_qubits"


def f_flip_condition(prog, rng):
    p = _pick(prog, If, rng)
    if p is None:
        return None
    q = copy.deepcopy(prog)
    l, i = _at(q, p)
    l[i].value = not l[i].value
    return q, "classical condition inverted", "flip_condition"


def f_condition_bit(prog, rng):
    p = _pick(prog, If, rng)
    if p is None or prog.n_clbits < 2:
        return None
    q = copy.deepcopy(prog)
    l, i = _at(q, p)
    l[i].c = rng.choice([c for c in range(prog.n_clbits) if c != l[i].c])
    return q, "condition reads the wrong classical bit", "condition_bit"


def f_delete_if(prog, rng):
    p = _pick(prog, If, rng)
    if p is None:
        return None
    q = copy.deepcopy(prog)
    l, i = _at(q, p)
    l.pop(i)
    return q, "classical correction missing", "delete_if"


def f_body_gate(prog, rng):
    p = _pick(prog, If, rng)
    if p is None:
        return None
    q = copy.deepcopy(prog)
    l, i = _at(q, p)
    g = l[i].then[0]
    if not isinstance(g, Gate) or g.name not in ("x", "z"):
        return None
    g.name = "z" if g.name == "x" else "x"
    return q, "correction applies the wrong gate", "body_gate"


def f_delete_reset(prog, rng):
    p = _pick(prog, Reset, rng)
    if p is None:
        return None
    q = copy.deepcopy(prog)
    l, i = _at(q, p)
    l.pop(i)
    return q, "reset missing", "delete_reset"


def f_reset_qubit(prog, rng):
    p = _pick(prog, Reset, rng)
    if p is None:
        return None
    q = copy.deepcopy(prog)
    l, i = _at(q, p)
    l[i].q = rng.choice([x for x in range(prog.n_qubits) if x != l[i].q])
    return q, "reset applied to the wrong qubit", "reset_qubit"


def f_loop_count(prog, rng):
    p = _pick(prog, For, rng)
    if p is None:
        return None
    q = copy.deepcopy(prog)
    l, i = _at(q, p)
    d = rng.choice([-1, 1]) if l[i].count > 1 else 1
    l[i].count += d
    return q, f"loop runs {d:+d} iteration", "loop_count"


def f_angle(prog, rng):
    cands = [p for p in D.enumerate_paths(prog.stmts)
             if isinstance(D._container(prog, p)[0][D._container(prog, p)[1]], Gate)
             and D._container(prog, p)[0][D._container(prog, p)[1]].params]
    if not cands:
        return None
    q = copy.deepcopy(prog)
    l, i = _at(q, rng.choice(cands))
    a = l[i].params[0]
    if rng.random() < 0.5:
        l[i].params[0] = round(-a, 4)
        return q, "rotation angle negated", "angle_negate"
    l[i].params[0] = round(a + math.pi / 2, 4)
    return q, "rotation angle shifted by pi/2", "angle_shift"


def f_reverse_cx(prog, rng):
    cx = [p for p in D.enumerate_paths(prog.stmts)
          if isinstance(D._container(prog, p)[0][D._container(prog, p)[1]], Gate)
          and D._container(prog, p)[0][D._container(prog, p)[1]].name == "cx"]
    if not cx:
        return None
    q = copy.deepcopy(prog)
    l, i = _at(q, rng.choice(cx))
    l[i].qubits = (l[i].qubits[1], l[i].qubits[0])
    return q, "CX control and target swapped", "reverse_cx"


def f_substitute(prog, rng):
    single = [p for p in D.enumerate_paths(prog.stmts)
              if isinstance(D._container(prog, p)[0][D._container(prog, p)[1]], Gate)
              and D._container(prog, p)[0][D._container(prog, p)[1]].name in D.SINGLE]
    if not single:
        return None
    q = copy.deepcopy(prog)
    l, i = _at(q, rng.choice(single))
    l[i].name = rng.choice([g for g in D.SINGLE if g != l[i].name])
    return q, "single-qubit gate replaced", "substitute"


CLASS_FAULTS: Dict[str, List[Callable]] = {
    "wiring": [f_swap_measure_bits, f_swap_measure_qubits],
    "control_flow": [f_flip_condition, f_condition_bit, f_delete_if, f_body_gate],
    "reset_fault": [f_delete_reset, f_reset_qubit],
    "loop_fault": [f_loop_count, f_angle],
    "gate_fault": [f_angle, f_reverse_cx, f_substitute],
}
ALL_FAULTS = sorted({f for fs in CLASS_FAULTS.values() for f in fs}, key=lambda f: f.__name__)


# --------------------------------------------------------------------------- #
def observable(buggy: Program, ref: Program, seed: int = 7) -> bool:
    try:
        inputs = D.input_states(ref.n_qubits, 16, seed)
        return any(D.tvd(D.distribution(buggy, inputs[:, j]), D.distribution(ref, inputs[:, j])) > OBSERVABLE_TVD
                   for j in range(inputs.shape[1]))
    except Exception:
        return False


def make(family: str, cls: str, seed: int, tries: int = 80) -> Tuple[Program, Program, str, str, int]:
    rng = random.Random(seed)
    ref = FAMILIES[family](rng)
    for _ in range(tries):
        if cls == "multi_fault":
            k = rng.choice([2, 3])
            cur, descs, kinds, used = copy.deepcopy(ref), [], [], set()
            for _ in range(k):
                fs = [f for f in ALL_FAULTS if f.__name__ not in used]
                rng.shuffle(fs)
                for f in fs:
                    r = f(cur, rng)
                    if r is not None:
                        cur, d, kind = r
                        descs.append(d); kinds.append(kind); used.add(f.__name__)
                        break
            if len(kinds) >= 2 and observable(cur, ref):
                return ref, cur, "; ".join(descs), "+".join(kinds), len(kinds)
        else:
            f = rng.choice(CLASS_FAULTS[cls])
            r = f(ref, rng)
            if r is not None and observable(r[0], ref):
                return ref, r[0], r[1], r[2], 1
    raise RuntimeError(f"no observable {cls} fault for {family} (seed {seed})")


def generate(root: str = PROJECT_ROOT, out_dir: str = "data_v3") -> pd.DataFrame:
    base = os.path.join(root, out_dir)
    for sub in ("buggy", "reference"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    rows, idx = [], 1
    for cls in CLASSES:
        for family, count in PER_CLASS[cls]:
            for _ in range(count):
                cid = f"dyn_{idx:04d}"
                ref, bug, desc, kind, nf = make(family, cls, SEED * 100003 + idx)
                bp, rp = f"{out_dir}/buggy/{cid}_buggy.qasm", f"{out_dir}/reference/{cid}_reference.qasm"
                open(os.path.join(root, bp), "w", encoding="utf-8").write(D.to_qasm(bug))
                open(os.path.join(root, rp), "w", encoding="utf-8").write(D.to_qasm(ref))
                rows.append({"circuit_id": cid, "family": family, "num_qubits": ref.n_qubits,
                             "num_clbits": ref.n_clbits, "bug_class": cls, "fault_kinds": kind, "n_faults": nf,
                             "buggy_file": bp, "reference_file": rp, "description": desc})
                idx += 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base, "metadata.csv"), index=False)
    print(f"wrote {len(df)} programs to {base}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=PROJECT_ROOT)
    ap.add_argument("--out", default="data_v3")
    a = ap.parse_args()
    generate(a.root, a.out)
