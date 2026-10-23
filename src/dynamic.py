"""Dynamic-circuit support (src/dynamic.py).

The v1/v2 pipeline compares unitaries, so it cannot say anything about programs with
measurement, reset, classical control or loops. This module adds a small, exact semantics for
that subset of OpenQASM 3:

    gates (standard library), measure q -> c, reset q,
    if (c[i] == true|false) { ... } [else { ... }],  for int k in [0:n-1] { ... } (constant body)

* ``parse`` / ``to_qasm``    OpenQASM 3 <-> a tiny IR that is easy to edit.
* ``distribution``           exact distribution of the classical register for one input state,
                             by branching on measurement outcomes (no sampling).
* ``DynamicOracle``         query-counted score = mean (1 - TVD) over a few test inputs, with a
                             counterexample pool and held-out inputs (same protocol as the
                             unitary oracle).
* ``validate``              acceptance predicate: parse, register sizes, device adjacency (also
                             inside blocks), exact output-distribution equality on held-out inputs.
* ``DynamicRepairer``       greedy search over an edit grammar that includes measurement wiring,
                             classical conditions, resets and loop bounds.

What it does not cover: while loops, multi-bit conditions, expressions in loop bodies, timing
beyond ``delay``, and the noise stage (a density matrix over branches is not implemented).
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import qiskit.qasm3
from qiskit.circuit.library import get_standard_gate_name_mapping

from src.physics_model import get_default_physics_model
from src.sim import _apply, random_product_states


# --------------------------------------------------------------------------- #
# IR
# --------------------------------------------------------------------------- #
@dataclass
class Gate:
    name: str
    params: List[float]
    qubits: Tuple[int, ...]


@dataclass
class Measure:
    q: int
    c: int


@dataclass
class Reset:
    q: int


@dataclass
class If:
    c: int
    value: bool
    then: List["Stmt"]
    els: List["Stmt"] = field(default_factory=list)


@dataclass
class For:
    count: int
    body: List["Stmt"]


Stmt = Union[Gate, Measure, Reset, If, For]


@dataclass
class Program:
    n_qubits: int
    n_clbits: int
    stmts: List[Stmt]


_MAP = get_standard_gate_name_mapping()
SINGLE = ["h", "x", "y", "z", "s", "sdg", "t", "tdg"]
ROT = ["rx", "ry", "rz"]
_MAT_CACHE: Dict[Tuple, np.ndarray] = {}


def gate_matrix(name: str, params: Sequence[float]) -> np.ndarray:
    key = (name, tuple(round(float(p), 12) for p in params))
    m = _MAT_CACHE.get(key)
    if m is None:
        if name not in _MAP:
            raise ValueError(f"unsupported gate {name!r}")
        g = _MAP[name]
        m = np.asarray(type(g)(*params).to_matrix() if g.params else type(g)().to_matrix(), dtype=complex)
        _MAT_CACHE[key] = m
    return m


# --------------------------------------------------------------------------- #
# Qiskit circuit -> IR
# --------------------------------------------------------------------------- #
def _convert(circ, qmap: Sequence[int], cmap: Sequence[int]) -> List[Stmt]:
    out: List[Stmt] = []
    for ins in circ.data:
        op = ins.operation
        name = op.name
        qs = tuple(qmap[circ.find_bit(q).index] for q in ins.qubits)
        cs = [cmap[circ.find_bit(c).index] for c in ins.clbits]
        if name in ("barrier", "delay"):
            continue
        if name == "measure":
            out.append(Measure(qs[0], cs[0]))
        elif name == "reset":
            out.append(Reset(qs[0]))
        elif name == "if_else":
            tgt, val = op.condition
            if hasattr(tgt, "size") and tgt.size != 1:
                raise ValueError("multi-bit conditions are not supported")
            bit = circ.find_bit(tgt[0] if hasattr(tgt, "size") else tgt).index
            # condition bit index in the *outer* register: resolve through this circuit's own clbits
            cbit = cmap[bit]
            then_b, else_b = op.params
            # map block-local indices through the instruction's own qubit/clbit lists
            def conv(block):
                if block is None:
                    return []
                bq = [qmap[circ.find_bit(q).index] for q in ins.qubits]
                bc = [cmap[circ.find_bit(c).index] for c in ins.clbits]
                return _convert(block, bq, bc)
            out.append(If(cbit, bool(val), conv(then_b), conv(else_b)))
        elif name == "for_loop":
            indexset, _param, body = op.params
            bq = [qmap[circ.find_bit(q).index] for q in ins.qubits]
            bc = [cmap[circ.find_bit(c).index] for c in ins.clbits]
            out.append(For(len(indexset), _convert(body, bq, bc)))
        else:
            params = [float(p) for p in op.params]
            gate_matrix(name, params)               # raises for unsupported gates
            out.append(Gate(name, params, qs))
    return out


def parse(qasm: str) -> Program:
    qc = qiskit.qasm3.loads(qasm)
    prog = Program(qc.num_qubits, qc.num_clbits, [])
    prog.stmts = _convert(qc, list(range(qc.num_qubits)), list(range(qc.num_clbits)))
    return prog


# --------------------------------------------------------------------------- #
# IR -> OpenQASM 3
# --------------------------------------------------------------------------- #
def _fmt(p: float) -> str:
    return repr(round(float(p), 10))


def _emit(stmts: Sequence[Stmt], ind: str, lines: List[str]) -> None:
    for s in stmts:
        if isinstance(s, Gate):
            args = f"({', '.join(_fmt(p) for p in s.params)})" if s.params else ""
            lines.append(f"{ind}{s.name}{args} {', '.join(f'q[{q}]' for q in s.qubits)};")
        elif isinstance(s, Measure):
            lines.append(f"{ind}c[{s.c}] = measure q[{s.q}];")
        elif isinstance(s, Reset):
            lines.append(f"{ind}reset q[{s.q}];")
        elif isinstance(s, If):
            lines.append(f"{ind}if (c[{s.c}] == {'true' if s.value else 'false'}) {{")
            _emit(s.then, ind + "  ", lines)
            if s.els:
                lines.append(f"{ind}}} else {{")
                _emit(s.els, ind + "  ", lines)
            lines.append(f"{ind}}}")
        elif isinstance(s, For):
            lines.append(f"{ind}for int k in [0:{max(s.count, 1) - 1}] {{")
            _emit(s.body, ind + "  ", lines)
            lines.append(f"{ind}}}")


def to_qasm(prog: Program) -> str:
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{prog.n_qubits}] q;"]
    if prog.n_clbits:
        lines.append(f"bit[{prog.n_clbits}] c;")
    _emit(prog.stmts, "", lines)
    return "\n".join(lines) + "\n"


def key(prog: Program) -> str:
    return hashlib.sha256(to_qasm(prog).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# exact semantics
# --------------------------------------------------------------------------- #
Branch = Tuple[np.ndarray, Tuple[int, ...]]          # (unnormalised state shaped (2,)*n+(1,), clbits)


def _project(state: np.ndarray, q: int, n: int, b: int) -> np.ndarray:
    ax = n - 1 - q
    out = np.zeros_like(state)
    sl = [slice(None)] * state.ndim
    sl[ax] = b
    out[tuple(sl)] = state[tuple(sl)]
    return out


def _flip(state: np.ndarray, q: int, n: int) -> np.ndarray:
    return np.flip(state, axis=n - 1 - q).copy()


def _run(stmts: Sequence[Stmt], branches: List[Branch], n: int) -> List[Branch]:
    for s in stmts:
        if isinstance(s, Gate):
            mat = gate_matrix(s.name, s.params)
            branches = [(_apply(st, mat, s.qubits, n), cb) for st, cb in branches]
        elif isinstance(s, Measure):
            nxt: List[Branch] = []
            for st, cb in branches:
                for b in (0, 1):
                    p = _project(st, s.q, n, b)
                    if np.vdot(p, p).real > 1e-15:
                        c2 = list(cb)
                        c2[s.c] = b
                        nxt.append((p, tuple(c2)))
            branches = nxt
        elif isinstance(s, Reset):
            nxt = []
            for st, cb in branches:
                for b in (0, 1):
                    p = _project(st, s.q, n, b)
                    if np.vdot(p, p).real > 1e-15:
                        nxt.append((_flip(p, s.q, n) if b else p, cb))
            branches = nxt
        elif isinstance(s, If):
            yes = [br for br in branches if bool(br[1][s.c]) == s.value]
            no = [br for br in branches if bool(br[1][s.c]) != s.value]
            branches = _run(s.then, yes, n) + _run(s.els, no, n)
        elif isinstance(s, For):
            for _ in range(max(s.count, 0)):
                branches = _run(s.body, branches, n)
    return branches


def distribution(prog: Program, init: np.ndarray) -> Dict[str, float]:
    """Exact distribution of the classical register (bit c[m-1] ... c[0] as a string)."""
    n = prog.n_qubits
    st = init.reshape((2,) * n + (1,)).astype(complex)
    out: Dict[str, float] = {}
    for state, cb in _run(prog.stmts, [(st, (0,) * prog.n_clbits)], n):
        p = float(np.vdot(state, state).real)
        k = "".join(str(b) for b in reversed(cb))
        out[k] = out.get(k, 0.0) + p
    return out


def tvd(p: Dict[str, float], q: Dict[str, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def input_states(n: int, count: int, seed: int) -> np.ndarray:
    """|0..0> followed by Haar-random product states, shape (2**n, count)."""
    rng = np.random.default_rng(seed)
    zero = np.zeros((2 ** n, 1), dtype=complex)
    zero[0, 0] = 1.0
    return np.concatenate([zero, random_product_states(n, max(count - 1, 0), rng)], axis=1)


# --------------------------------------------------------------------------- #
# structure helpers
# --------------------------------------------------------------------------- #
def walk(stmts: Sequence[Stmt], depth: int = 0) -> Iterator[Tuple[Stmt, int]]:
    for s in stmts:
        yield s, depth
        if isinstance(s, If):
            yield from walk(s.then, depth + 1)
            yield from walk(s.els, depth + 1)
        elif isinstance(s, For):
            yield from walk(s.body, depth + 1)


def two_qubit_pairs(prog: Program) -> List[Tuple[int, int]]:
    out = []
    for s, _ in walk(prog.stmts):
        if isinstance(s, Gate) and len(s.qubits) == 2:
            out.append((s.qubits[0], s.qubits[1]))
    return out


def has_dynamic(qasm: str) -> bool:
    return any(w in qasm for w in ("measure", "reset", "if (", "for ", "while"))


# --------------------------------------------------------------------------- #
# oracle and acceptance
# --------------------------------------------------------------------------- #
class DynamicOracle:
    """Query-counted behavioural specification for dynamic programs."""

    def __init__(self, reference: Program, query_budget: int = 3000, suite_size: int = 4,
                 pool_size: int = 8, holdout_size: int = 12, seed_key: str = "dyn"):
        seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
        self.n = reference.n_qubits
        self._ref = reference
        self._suite = input_states(self.n, suite_size, seed)
        self._pool = input_states(self.n, pool_size + 1, seed + 1)[:, 1:]
        self._hold = input_states(self.n, holdout_size, seed + 2)
        self._suite_exp = [distribution(reference, self._suite[:, j]) for j in range(self._suite.shape[1])]
        self._pool_exp = [distribution(reference, self._pool[:, j]) for j in range(self._pool.shape[1])]
        self._hold_exp = [distribution(reference, self._hold[:, j]) for j in range(self._hold.shape[1])]
        self.query_budget, self.queries_used, self.counterexamples_used = query_budget, 0, 0

    def exhausted(self) -> bool:
        return self.queries_used >= self.query_budget

    def remaining(self) -> int:
        return max(0, self.query_budget - self.queries_used)

    def _mean_score(self, prog: Program, inputs, exp) -> float:
        return float(np.mean([1.0 - tvd(distribution(prog, inputs[:, j]), exp[j]) for j in range(inputs.shape[1])]))

    def score(self, prog: Program) -> float:
        if self.exhausted():
            return 0.0
        self.queries_used += 1
        try:
            return self._mean_score(prog, self._suite, self._suite_exp)
        except Exception:
            return 0.0

    def counterexample(self, prog: Program) -> Optional[float]:
        if self.exhausted() or self._pool.shape[1] == 0:
            return None
        self.queries_used += 1
        try:
            gaps = [tvd(distribution(prog, self._pool[:, j]), self._pool_exp[j]) for j in range(self._pool.shape[1])]
        except Exception:
            return None
        w = int(np.argmax(gaps))
        self._suite = np.concatenate([self._suite, self._pool[:, w:w + 1]], axis=1)
        self._suite_exp.append(self._pool_exp[w])
        self._pool = np.delete(self._pool, w, axis=1)
        del self._pool_exp[w]
        self.counterexamples_used += 1
        return float(1.0 - gaps[w])

    def holdout_gap(self, prog: Program) -> Dict[str, float]:
        try:
            suite = self._mean_score(prog, self._suite, self._suite_exp)
            hold = self._mean_score(prog, self._hold, self._hold_exp)
        except Exception:
            return {"suite": 0.0, "holdout": 0.0, "gap": 0.0}
        return {"suite": suite, "holdout": hold, "gap": suite - hold}


def validate(qasm: str, reference: Program, tol: float = 1e-6, n_inputs: int = 16, seed: int = 7,
             device: bool = True) -> Dict:
    """Acceptance for dynamic programs: parse, register sizes, adjacency, exact distributions.
    ``device=False`` drops the adjacency stage (logic-only acceptance, for programs whose own reference is not device-ready)."""
    metrics: Dict = {}

    def fail(stage, why):
        return {"status": "FAIL", "failed_stage": stage, "feedback": why, "metrics": metrics}

    try:
        prog = parse(qasm)
    except Exception as e:
        return fail("syntax", f"not valid OpenQASM 3 (or unsupported construct): {e}")
    if prog.n_qubits != reference.n_qubits or prog.n_clbits != reference.n_clbits:
        return fail("structural", f"expected {reference.n_qubits} qubits and {reference.n_clbits} classical bits, "
                                  f"got {prog.n_qubits} and {prog.n_clbits}")
    physics = get_default_physics_model()
    for a, b in (two_qubit_pairs(prog) if device else []):
        if not physics.is_connected(a, b):
            return fail("connectivity", f"two-qubit gate on non-adjacent qubits ({a}, {b}), also checked inside blocks")
    inputs = input_states(prog.n_qubits, n_inputs, seed)
    worst = 0.0
    for j in range(inputs.shape[1]):
        worst = max(worst, tvd(distribution(prog, inputs[:, j]), distribution(reference, inputs[:, j])))
    metrics["max_tvd"] = worst
    if worst > tol:
        return fail("semantic", f"output distribution differs on held-out inputs (max TVD {worst:.4f})")
    return {"status": "PASS", "failed_stage": None, "feedback": "accepted", "metrics": metrics}


# --------------------------------------------------------------------------- #
# repair: edit grammar and greedy search
# --------------------------------------------------------------------------- #
def enumerate_paths(stmts: List[Stmt]) -> List[Tuple[Tuple[str, int], ...]]:
    out: List[Tuple[Tuple[str, int], ...]] = []

    def rec(lst, pre):
        for i, s in enumerate(lst):
            here = pre + (("*", i),)
            out.append(here)
            if isinstance(s, If):
                rec(s.then, pre + (("then", i),))
                rec(s.els, pre + (("els", i),))
            elif isinstance(s, For):
                rec(s.body, pre + (("body", i),))
    rec(stmts, ())
    return out


def _container(prog: Program, path):
    """List holding the statement addressed by ``path`` and its index."""
    lst = prog.stmts
    for kind, idx in path[:-1]:
        lst = getattr(lst[idx], kind)
    return lst, path[-1][1]


class DynamicRepairer:
    """Greedy edit search over the dynamic-program edit grammar, ranked by a DynamicOracle."""

    def __init__(self, max_steps: int = 5, build_per_step: int = 4, accept_at: float = 0.999999):
        self.max_steps, self.build_per_step, self.accept_at = max_steps, build_per_step, accept_at

    # -- neighbourhood ------------------------------------------------------
    def edits(self, prog: Program) -> Iterator[Tuple[str, Program]]:
        n, m = prog.n_qubits, prog.n_clbits
        for path in enumerate_paths(prog.stmts):
            lst0, i0 = _container(prog, path)
            s = lst0[i0]

            def variant(fn):
                p = copy.deepcopy(prog)
                lst, i = _container(p, path)
                fn(lst, i)
                return p

            yield "delete", variant(lambda l, i: l.pop(i))
            if isinstance(s, Gate):
                if len(s.qubits) == 1 and not s.params and s.name in SINGLE:
                    for g in SINGLE:
                        if g != s.name:
                            yield "substitute", variant(lambda l, i, g=g: setattr(l[i], "name", g))
                if len(s.qubits) == 1 and s.name in ROT:
                    for g in ROT:
                        if g != s.name:
                            yield "axis", variant(lambda l, i, g=g: setattr(l[i], "name", g))
                if s.params:
                    for tag, f in (("negate", lambda a: -a), ("plus_pi2", lambda a: a + math.pi / 2),
                                   ("minus_pi2", lambda a: a - math.pi / 2), ("plus_pi", lambda a: a + math.pi)):
                        yield tag, variant(lambda l, i, f=f: l[i].params.__setitem__(0, f(l[i].params[0])))
                if s.name == "cx":
                    yield "reverse", variant(lambda l, i: setattr(l[i], "qubits", (l[i].qubits[1], l[i].qubits[0])))
                if len(s.qubits) == 1:
                    for q2 in range(n):
                        if q2 != s.qubits[0]:
                            yield "retarget", variant(lambda l, i, q2=q2: setattr(l[i], "qubits", (q2,)))
            elif isinstance(s, Measure):
                for c2 in range(m):
                    if c2 != s.c:
                        yield "measure_to_bit", variant(lambda l, i, c2=c2: setattr(l[i], "c", c2))
                for q2 in range(n):
                    if q2 != s.q:
                        yield "measure_qubit", variant(lambda l, i, q2=q2: setattr(l[i], "q", q2))
            elif isinstance(s, Reset):
                for q2 in range(n):
                    if q2 != s.q:
                        yield "reset_qubit", variant(lambda l, i, q2=q2: setattr(l[i], "q", q2))
            elif isinstance(s, If):
                yield "flip_condition", variant(lambda l, i: setattr(l[i], "value", not l[i].value))
                for c2 in range(m):
                    if c2 != s.c:
                        yield "condition_bit", variant(lambda l, i, c2=c2: setattr(l[i], "c", c2))
                if s.els:
                    yield "swap_branches", variant(lambda l, i: (setattr(l[i], "then", l[i].els),
                                                                setattr(l[i], "els", l[i].then)))
            elif isinstance(s, For):
                for d in (-1, 1):
                    if s.count + d >= 1:
                        yield "loop_count", variant(lambda l, i, d=d: setattr(l[i], "count", l[i].count + d))
        # coordinated pair edits: a swap of two measurements cannot be undone one edit at a time,
        # because the intermediate program (two measurements writing one bit) scores worse
        meas = [p for p in enumerate_paths(prog.stmts) if isinstance(_container(prog, p)[0][_container(prog, p)[1]], Measure)]
        for a in range(len(meas)):
            for b in range(a + 1, len(meas)):
                for attr, tag in (("c", "swap_measure_bits"), ("q", "swap_measure_qubits")):
                    p = copy.deepcopy(prog)
                    (la, ia), (lb, ib) = _container(p, meas[a]), _container(p, meas[b])
                    va, vb = getattr(la[ia], attr), getattr(lb[ib], attr)
                    if va != vb:
                        setattr(la[ia], attr, vb)
                        setattr(lb[ib], attr, va)
                        yield tag, p
        # insertions: reset after a measure; classical correction before the last measure of a qubit
        for path in enumerate_paths(prog.stmts):
            lst0, i0 = _container(prog, path)
            s = lst0[i0]
            if isinstance(s, Measure) and not (i0 + 1 < len(lst0) and isinstance(lst0[i0 + 1], Reset)):
                p = copy.deepcopy(prog)
                lst, i = _container(p, path)
                lst.insert(i + 1, Reset(s.q))
                yield "insert_reset", p
        last_measure: Dict[int, Tuple] = {}
        for path in enumerate_paths(prog.stmts):
            lst0, i0 = _container(prog, path)
            if isinstance(lst0[i0], Measure):
                last_measure[lst0[i0].q] = path
        for q, path in last_measure.items():
            for c in range(m):
                for g in ("x", "z"):
                    for val in (True,):
                        p = copy.deepcopy(prog)
                        lst, i = _container(p, path)
                        lst.insert(i, If(c, val, [Gate(g, [], (q,))]))
                        yield "insert_correction", p

    def _angle_search(self, prog: Program, oracle: DynamicOracle, base: float) -> List[Tuple[float, str, Program]]:
        """1-D scan + refinement of every rotation angle (the score is smooth in a single angle)."""
        out = []
        for path in enumerate_paths(prog.stmts):
            lst0, i0 = _container(prog, path)
            s = lst0[i0]
            if not (isinstance(s, Gate) and s.params and s.name in ROT + ["p", "cp"]):
                continue

            def at(theta):
                p = copy.deepcopy(prog)
                lst, i = _container(p, path)
                lst[i].params[0] = theta
                return p
            grid = [2 * math.pi * k / 36 for k in range(36)]   # a loop body repeats the rotation: multi-modal
            vals = []
            for th in grid:
                if oracle.exhausted():
                    break
                vals.append(oracle.score(at(th)))
            if not vals:
                continue
            k = int(np.argmax(vals))
            lo, hi = grid[k] - math.pi / 36, grid[k] + math.pi / 36
            for _ in range(8):                           # golden-section refinement
                if oracle.exhausted():
                    break
                a = lo + 0.382 * (hi - lo)
                b = lo + 0.618 * (hi - lo)
                if oracle.score(at(a)) > oracle.score(at(b)):
                    hi = b
                else:
                    lo = a
            best = (lo + hi) / 2
            sc = oracle.score(at(best))
            if sc > base + 1e-9:
                out.append((sc, "angle_solve", at(round(best, 4))))
        return out

    # -- search ---------------------------------------------------------------
    def search(self, prog: Program, oracle: DynamicOracle, tabu: Set[str]) -> Program:
        cur, cur_score = prog, oracle.score(prog)
        visited = {key(cur)}
        best: Optional[Tuple[float, Program]] = None if key(cur) in tabu else (cur_score, cur)
        if best and cur_score >= self.accept_at:
            return cur
        for _ in range(self.max_steps):
            if oracle.exhausted():
                break
            cands: List[Tuple[float, str, Program]] = []
            for tag, p in self.edits(cur):
                if oracle.exhausted():
                    break
                cands.append((oracle.score(p), tag, p))
            cands += self._angle_search(cur, oracle, cur_score)
            cands.sort(key=lambda t: -t[0])
            nxt, built = None, 0
            for sc, tag, p in cands[:30]:
                k = key(p)
                if k in visited or k in tabu:
                    continue
                visited.add(k)
                built += 1
                if best is None or sc > best[0]:
                    best = (sc, p)
                if nxt is None and sc > cur_score + 1e-9:
                    nxt = (sc, p)
                if built >= self.build_per_step:
                    break
            if nxt is None:
                break
            cur_score, cur = nxt
            if cur_score >= self.accept_at:
                break
        return best[1] if best else cur


def repair(qasm: str, reference_qasm: str, attempts: int = 3, budget: int = 3000,
           feedback: bool = True, seed_key: str = "dyn", device: bool = True) -> Dict:
    """Bounded repair loop for dynamic programs (same protocol as the unitary pipeline)."""
    ref = parse(reference_qasm)
    oracle = DynamicOracle(ref, query_budget=budget, seed_key=seed_key)
    rep = DynamicRepairer()
    cur_q = qasm
    tabu: Set[str] = set()
    history = []
    final = qasm
    for it in range(1, attempts + 1):
        try:
            prog = parse(cur_q)
        except Exception:
            history.append({"attempt": it, "status": "FAIL", "failed_stage": "syntax"})
            break
        cand = rep.search(prog, oracle, tabu)
        tabu.add(key(cand))
        final = to_qasm(cand)
        v = validate(final, ref, device=device)
        history.append({"attempt": it, "status": v["status"], "failed_stage": v["failed_stage"],
                        "queries": oracle.queries_used})
        if v["status"] == "PASS":
            return {"success": True, "qasm": final, "history": history, "queries": oracle.queries_used,
                    "gap": oracle.holdout_gap(cand)}
        if feedback and v["failed_stage"] == "semantic":
            oracle.counterexample(cand)
        cur_q = final
    return {"success": False, "qasm": final, "history": history, "queries": oracle.queries_used, "gap": None}


def random_repair(qasm: str, reference_qasm: str, attempts: int = 3, budget: int = 3000,
                  seed: int = 0, device: bool = True) -> Dict:
    """Baseline without guidance: random-mutation hill climbing over the same edit grammar and
    the same oracle budget (an edit is kept only if it raises the suite score)."""
    import random as _random
    rng = _random.Random(seed)
    ref = parse(reference_qasm)
    oracle = DynamicOracle(ref, query_budget=budget, seed_key=f"rand{seed}")
    history, final = [], qasm
    per_attempt = max(budget // attempts, 1)
    try:
        cur = parse(qasm)
    except Exception:
        return {"success": False, "qasm": qasm, "history": [{"attempt": 1, "status": "FAIL", "failed_stage": "syntax"}],
                "queries": 0, "gap": None}
    for it in range(1, attempts + 1):
        spent_to = oracle.queries_used + per_attempt
        cur_score = oracle.score(cur)
        pool = [p for _, p in DynamicRepairer().edits(cur)]
        rng.shuffle(pool)
        while pool and oracle.queries_used < spent_to and not oracle.exhausted():
            cand = pool.pop()
            sc = oracle.score(cand)
            if sc > cur_score + 1e-9:
                cur, cur_score = cand, sc
                pool = [p for _, p in DynamicRepairer().edits(cur)]
                rng.shuffle(pool)
                if cur_score >= 0.999999:
                    break
        final = to_qasm(cur)
        v = validate(final, ref, device=device)
        history.append({"attempt": it, "status": v["status"], "failed_stage": v["failed_stage"],
                        "queries": oracle.queries_used})
        if v["status"] == "PASS":
            return {"success": True, "qasm": final, "history": history, "queries": oracle.queries_used,
                    "gap": oracle.holdout_gap(cur)}
    return {"success": False, "qasm": final, "history": history, "queries": oracle.queries_used, "gap": None}


def unitary_view(prog: Program):
    """Gate-only circuit (loops expanded, measurements/resets/conditionals dropped): what a
    unitary-equivalence check would see."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(prog.n_qubits)

    def emit(stmts):
        for s in stmts:
            if isinstance(s, Gate):
                qc.append(type(_MAP[s.name])(*s.params) if _MAP[s.name].params else type(_MAP[s.name])(), s.qubits)
            elif isinstance(s, For):
                for _ in range(s.count):
                    emit(s.body)
    emit(prog.stmts)
    return qc
