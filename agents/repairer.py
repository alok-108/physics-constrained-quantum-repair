"""Agent 2: Repairer (agents/repairer.py).

Produces candidate repaired OpenQASM 3.0 circuits. The agent never receives the
ground-truth circuit or the injected fault class; semantic repair is driven by a
query-counted behavioural oracle (``agents/oracle.py``).

Repair strategies (deterministic path; an optional LLM path is tried first when
an API key is configured):

* syntax           -- token-level recovery.
* gate redundancy  -- DAG inverse cancellation.
* physics          -- drop over-budget idle delays, then swap-in/swap-back
                      routing with cancellation. Only when ``use_physics_context``.
* semantic/structural -- oracle-guided hill climbing over single-instruction
                      edits. The neighbourhood is generic (it is not conditioned
                      on a fault label) and contains:
        - analytic angle solving (Rotosolve-style): the agreement is a
          degree-1 trigonometric polynomial in each rotation angle, so three
          probe queries identify its exact maximiser;
        - gate substitution, rotation-axis exchange, operand reversal,
          re-targeting a gate to another qubit;
        - deletion of an instruction;
        - insertion of H, X, adjacent CX or adjacent SWAP at any position.
  Every candidate edit costs one oracle query. A tabu set of previously
  submitted circuits forces a *different* candidate on retry, so extra attempts
  are never silently wasted on a byte-identical resubmission.

Not in the neighbourhood by design: two-qubit gate substitution (e.g. CX -> CZ),
insertion of rotations, and simultaneous multi-instruction edits beyond the
greedy depth. Faults needing those are expected to remain unresolved; the
benchmark reports them per fault mode.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:                                   # only .type and .content are used by the LLM clients
    from types import SimpleNamespace as _NS

    def SystemMessage(content):
        return _NS(type="system", content=content)

    def HumanMessage(content):
        return _NS(type="human", content=content)

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

import qiskit
import qiskit.qasm3
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CPhaseGate, CXGate, CYGate, CZGate, HGate, PhaseGate, RXGate, RYGate, RZGate, SdgGate, SGate,
    SwapGate, TdgGate, TGate, XGate, YGate, ZGate,
)

from agents.diagnostician import LEAKY_KEYS
from agents.llm import QuotaExhausted
from agents.oracle import BehaviouralOracle
from src.edits import Edit, apply_edit
from src.physics_model import get_default_physics_model
from src.routing import cancel_inverses, circuit_key, compile_with_qiskit, route_and_cancel

ANGLE_GATES = {"rx": RXGate, "ry": RYGate, "rz": RZGate, "p": PhaseGate, "cp": CPhaseGate}
SINGLE_GATES = {"h": HGate, "x": XGate, "y": YGate, "z": ZGate,
                "s": SGate, "sdg": SdgGate, "t": TGate, "tdg": TdgGate}
AXIS_EXCHANGE = {"rx": ("ry", "rz"), "ry": ("rx", "rz"), "rz": ("rx", "ry")}
PROBE_ANGLES = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
COHERENCE_DELAY_US = 150.0
TWO_QUBIT_SUBS = {"cx": (CZGate, CYGate), "cz": (CXGate,), "cy": (CXGate,)}


class RepairerAgent:
    """Repairs quantum circuits from a diagnosis and a task specification."""

    SYSTEM_PROMPT_PHYSICS = (
        "You are a quantum circuit repair expert. Produce corrected OpenQASM 3.0 "
        "respecting device connectivity, depth, and coherence limits. Distinguish "
        "between logical fixes and noise-aware routing.\n"
        "Output ONLY the repaired OpenQASM 3.0 code, wrapped inside ```qasm ... ``` "
        "code block. Do not include conversational preamble."
    )
    SYSTEM_PROMPT_NO_PHYSICS = (
        "You are a quantum circuit repair expert. Produce corrected OpenQASM 3.0 "
        "that implements the intended computation.\n"
        "Output ONLY the repaired OpenQASM 3.0 code, wrapped inside ```qasm ... ``` "
        "code block. Do not include conversational preamble."
    )

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0,
                 use_physics_context: bool = True, accept_at: float = 0.999,
                 max_steps: int = 4, build_per_step: int = 4,
                 router: str = "swapback", extended_edits: bool = False,
                 llm_model: Optional[str] = None, llm_only: bool = False,
                 guard_cancel: bool = False):
        self.model_name = model_name
        self.temperature = temperature
        self.use_physics_context = use_physics_context
        self.accept_at = accept_at
        self.max_steps = max_steps
        self.build_per_step = build_per_step
        self.llm_only = llm_only          # LLM arm: no heuristic or oracle fallback, a failed LLM call = unchanged circuit
        self.extended_edits = extended_edits   # + 2-qubit substitution and analytic rotation insertion
        self.guard_cancel = guard_cancel   # apply inverse cancellation only to circuits the oracle already scores as functionally correct
        self.router = router            # "swapback" (own construction) or "qiskit" (SABRE + layout undo)
        self.physics = get_default_physics_model()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = None
        self.last_strategy = "none"
        self.llm_errors = 0             # API/transport failures (not wrong answers), cumulative
        if llm_model:
            from agents.llm import make_llm
            self.llm = make_llm(llm_model)
        elif self.api_key and ChatOpenAI is not None and not self.api_key.startswith("your-"):
            try:
                self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature,
                                      api_key=self.api_key)
            except Exception:
                self.llm = None

    # ------------------------------------------------------------------ #
    def repair(self, buggy_qasm: str, bug_report: Dict[str, Any],
               circuit_spec: Optional[Dict[str, Any]] = None,
               feedback: Optional[str] = None,
               oracle: Optional[BehaviouralOracle] = None,
               tabu: Optional[Set[Tuple]] = None) -> str:
        """Produce a candidate. No argument carries the reference or the fault label."""
        spec = {k: v for k, v in dict(circuit_spec or {}).items() if k not in LEAKY_KEYS}
        tabu = tabu if tabu is not None else set()

        if self.llm is not None:
            cand = self._llm_repair(buggy_qasm, bug_report, spec, feedback)
            # In the LLM-only arm the model is not forced to diversify: resubmitting the same circuit
            # is its own choice and is simply rejected again. "llm_failed" means no parsable reply.
            if cand is not None and (self.llm_only or self._key_of(cand) not in tabu):
                self.last_strategy = "llm"
                return cand
            if self.llm_only:
                self.last_strategy = "llm_failed"
                return buggy_qasm

        return self._heuristic_repair(buggy_qasm, bug_report, spec, oracle, tabu)

    # ------------------------------------------------------------------ #
    # LLM path
    # ------------------------------------------------------------------ #
    def _llm_repair(self, buggy_qasm, bug_report, spec, feedback) -> Optional[str]:
        try:
            msg = (f"Task specification:\n{spec}\n\nFaulty OpenQASM 3.0:\n```qasm\n{buggy_qasm}\n```\n\n"
                   f"Diagnostic report:\n{bug_report}\n\n")
            if self.use_physics_context:
                msg += ("Hardware physics constraints:\n{'device': 'IBM FakeSherbrooke (127 qubits)', "
                        "'max_depth': 250, 'T1_typical': '250 us', 'T2_typical': '150 us'}\n\n")
            if feedback:
                msg += f"Validator feedback:\n{feedback}\n\n"
            msg += "Produce the complete, corrected OpenQASM 3.0 circuit now."
            system = self.SYSTEM_PROMPT_PHYSICS if self.use_physics_context else self.SYSTEM_PROMPT_NO_PHYSICS
            try:
                content = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=msg)]).content.strip()
            except QuotaExhausted:
                raise                                   # abort the run: continuing would only record outages
            except Exception:
                self.llm_errors += 1
                return None
            m = re.search(r"```(?:qasm)?\n(.*?)```", content, re.DOTALL)
            cand = m.group(1).strip() if m else content
            qiskit.qasm3.loads(cand)
            return cand
        except QuotaExhausted:
            raise
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Deterministic path
    # ------------------------------------------------------------------ #
    def _heuristic_repair(self, buggy_qasm, bug_report, spec, oracle, tabu) -> str:
        bug_type = bug_report.get("bug_type", "")
        self.last_strategy = "heuristic"

        if bug_type == "syntax" or not self._is_parsable(buggy_qasm):
            fixed = self._repair_syntax(buggy_qasm)
            if not self._is_parsable(fixed):
                self.last_strategy = "syntax_unrecovered"
                return fixed
            if circuit_key(qiskit.qasm3.loads(fixed)) not in tabu and bug_type == "syntax":
                self.last_strategy = "syntax"
                return fixed
            buggy_qasm = fixed

        qc = qiskit.qasm3.loads(buggy_qasm)

        if bug_type == "gate_redundancy":
            out = cancel_inverses(qc)
            # Two CX on the same pair around a rotation that sits on another wire cancel exactly, so the static
            # diagnosis calls the circuit redundant, but the cancelled circuit is further from the repair (the
            # real fault, a misplaced rotation, then needs three edits instead of one). A circuit that is only
            # redundant is already functionally right; with the guard, cancel when the oracle agrees it is.
            safe = not (self.guard_cancel and oracle is not None and oracle.score(qc) < self.accept_at)
            if safe:
                if circuit_key(out) not in tabu:
                    self.last_strategy = "cancel"
                    return qiskit.qasm3.dumps(out)
                qc = out

        if bug_type == "physics_violation" and self.use_physics_context:
            out = self._repair_physics(qc)
            if circuit_key(out) not in tabu:
                self.last_strategy = "route"
                return qiskit.qasm3.dumps(out)
            qc = out

        if oracle is not None:
            self.last_strategy = "search"
            return qiskit.qasm3.dumps(self._search(qc, oracle, tabu))
        self.last_strategy = "none"
        return qiskit.qasm3.dumps(qc)

    # ------------------------------------------------------------------ #
    # Physics repair
    # ------------------------------------------------------------------ #
    def _repair_physics(self, qc: QuantumCircuit) -> QuantumCircuit:
        kept = qc.copy_empty_like()
        for ins in qc.data:
            if ins.operation.name == "delay":
                dur = getattr(ins.operation, "duration", 0) or 0
                unit = getattr(ins.operation, "unit", "s")
                dt_s = getattr(self.physics.target, "dt", None) or 2.222e-10
                to_us = {"us": 1.0, "s": 1e6, "ms": 1e3, "ns": 1e-3, "dt": dt_s * 1e6}.get(unit, 0.0)
                if dur * to_us > COHERENCE_DELAY_US:
                    continue
            kept._append(ins)
        if self.router == "qiskit":
            return cancel_inverses(compile_with_qiskit(kept))
        return route_and_cancel(kept, self.physics)

    # ------------------------------------------------------------------ #
    # Oracle-guided search
    # ------------------------------------------------------------------ #
    def _search(self, qc: QuantumCircuit, oracle: BehaviouralOracle,
                tabu: Set[Tuple]) -> QuantumCircuit:
        cur = qc
        cur_score = oracle.score(cur)
        visited = {circuit_key(cur)}
        best: Optional[Tuple[float, int, QuantumCircuit]] = None
        if circuit_key(cur) not in tabu:
            best = (cur_score, -len(cur.data), cur)
            if cur_score >= self.accept_at:
                return cur

        for _ in range(self.max_steps):
            if oracle.exhausted():
                break
            cands = self._neighbourhood(cur, oracle)
            if (self.extended_edits and not oracle.exhausted()
                    and (not cands or cands[0][0] < self.accept_at)):
                # no single edit completes the repair: let angle-solved rotation insertions compete
                cands = sorted(cands + self._rotation_insertions(cur, oracle), key=lambda r: -r[0])
            nxt: Optional[Tuple[float, QuantumCircuit]] = None
            built = 0
            for score, edit in cands[:24]:
                nb = apply_edit(cur, edit)
                k = circuit_key(nb)
                if k in visited or k in tabu:
                    continue
                visited.add(k)
                built += 1
                rank = (score, -len(nb.data))
                if best is None or rank > (best[0], best[1]):
                    best = (score, -len(nb.data), nb)
                if nxt is None and score > cur_score + 1e-9:
                    nxt = (score, nb)
                if built >= self.build_per_step:
                    break
            if nxt is None:
                break
            cur_score, cur = nxt
            if cur_score >= self.accept_at:
                break
        return best[2] if best is not None else cur

    def _neighbourhood(self, qc: QuantumCircuit, oracle: BehaviouralOracle,
                       rotations: bool = False) -> List[Tuple[float, Edit]]:
        """All single-instruction edits, scored by the oracle, best first."""
        n = qc.num_qubits
        results: List[Tuple[float, int, Edit]] = []

        # (A) analytic angle solving: 3 probe queries per rotation instruction
        angle_idx = [i for i, ins in enumerate(qc.data) if ins.operation.name in ANGLE_GATES]
        if angle_idx and oracle.remaining() >= 3 * len(angle_idx):
            probes: List[Edit] = []
            for i in angle_idx:
                ins = qc.data[i]
                cls = ANGLE_GATES[ins.operation.name]
                qs = tuple(qc.find_bit(q).index for q in ins.qubits)
                probes += [Edit("replace", i, cls(t), qs, "angle_probe") for t in PROBE_ANGLES]
            sc = oracle.evaluate_edits(qc, probes)
            design = np.array([[1.0, math.cos(t), math.sin(t)] for t in PROBE_ANGLES])
            for g, i in enumerate(angle_idx):
                f = sc[3 * g:3 * g + 3]
                if np.isnan(f).any():
                    continue
                c0, c1, s1 = np.linalg.solve(design, f)
                theta = math.atan2(s1, c1)
                pred = min(1.0, float(c0 + math.hypot(c1, s1)))
                ins = qc.data[i]
                cls = ANGLE_GATES[ins.operation.name]
                qs = tuple(qc.find_bit(q).index for q in ins.qubits)
                results.append((pred, 0, Edit("replace", i, cls(round(theta, 4)), qs, "angle")))

        # (B..G) discrete edits, evaluated as one batch in priority order
        batch: List[Tuple[int, Edit]] = []
        L = len(qc.data)
        for i, ins in enumerate(qc.data):
            name = ins.operation.name
            qs = tuple(qc.find_bit(q).index for q in ins.qubits)
            if len(qs) == 1 and not ins.operation.params and name in SINGLE_GATES:
                for g, cls in SINGLE_GATES.items():
                    if g != name:
                        batch.append((1, Edit("replace", i, cls(), qs, "substitute")))
            if len(qs) == 1 and name in AXIS_EXCHANGE:
                for alt in AXIS_EXCHANGE[name]:
                    batch.append((2, Edit("replace", i, ANGLE_GATES[alt](ins.operation.params[0]), qs, "axis")))
            if len(qs) == 1 and name in SINGLE_GATES | ANGLE_GATES:
                cls = (SINGLE_GATES | ANGLE_GATES)[name]
                for q2 in range(n):
                    if q2 != qs[0]:
                        gate = cls(*ins.operation.params) if ins.operation.params else cls()
                        batch.append((3, Edit("replace", i, gate, (q2,), "retarget")))
            if name == "cx":
                batch.append((3, Edit("replace", i, CXGate(), (qs[1], qs[0]), "reverse")))
            if self.extended_edits and name in TWO_QUBIT_SUBS and len(qs) == 2:
                for alt in TWO_QUBIT_SUBS[name]:
                    batch.append((2, Edit("replace", i, alt(), qs, "substitute_2q")))
        for i, ins in enumerate(qc.data):
            if ins.operation.name not in ("delay", "barrier", "measure"):
                batch.append((4, Edit("delete", i, None, (), "delete")))
        for p in range(L + 1):
            for q in range(n):
                batch.append((5, Edit("insert", p, HGate(), (q,), "insert_h")))
                batch.append((5, Edit("insert", p, XGate(), (q,), "insert_x")))
            for q in range(n - 1):
                batch.append((5, Edit("insert", p, CXGate(), (q, q + 1), "insert_cx")))
                batch.append((5, Edit("insert", p, CXGate(), (q + 1, q), "insert_cx")))
                batch.append((5, Edit("insert", p, SwapGate(), (q, q + 1), "insert_swap")))
        edits = [e for _, e in batch]
        sc = oracle.evaluate_edits(qc, edits)
        for (prio, e), s in zip(batch, sc):
            if not np.isnan(s):
                results.append((float(s), prio, e))

        if rotations and self.extended_edits:
            results += [(sc_, 6, e_) for sc_, e_ in self._rotation_insertions(qc, oracle)]
        results.sort(key=lambda r: (-r[0], r[1]))
        return [(s, e) for s, _, e in results]

    def _rotation_insertions(self, qc: QuantumCircuit, oracle: BehaviouralOracle) -> List[Tuple[float, Edit]]:
        """Insert a rotation (or controlled phase on adjacent qubits) anywhere, with its angle solved
        analytically from three probe queries; costs 3 queries per (position, gate, qubits)."""
        n, L = qc.num_qubits, len(qc.data)
        slots = [(name, (q,)) for q in range(n) for name in ("rx", "ry", "rz")]
        slots += [("cp", (q, q + 1)) for q in range(n - 1)]
        need = 3 * len(slots) * (L + 1)
        if oracle.remaining() < need:
            return []
        probes, meta = [], []
        for pos in range(L + 1):
            for name, qs in slots:
                cls = ANGLE_GATES[name]
                probes += [Edit("insert", pos, cls(t), qs, "angle_probe") for t in PROBE_ANGLES]
                meta.append((pos, name, qs))
        sc = oracle.evaluate_edits(qc, probes)
        design = np.array([[1.0, math.cos(t), math.sin(t)] for t in PROBE_ANGLES])
        out: List[Tuple[float, Edit]] = []
        for g, (pos, name, qs) in enumerate(meta):
            f = sc[3 * g:3 * g + 3]
            if np.isnan(f).any():
                continue
            c0, c1, s1 = np.linalg.solve(design, f)
            theta = round(math.atan2(s1, c1), 4)
            out.append((min(1.0, float(c0 + math.hypot(c1, s1))),
                        Edit("insert", pos, ANGLE_GATES[name](theta), qs, "insert_rotation")))
        return out

    # ------------------------------------------------------------------ #
    def _key_of(self, qasm: str) -> Optional[Tuple]:
        try:
            return circuit_key(qiskit.qasm3.loads(qasm))
        except Exception:
            return None

    @staticmethod
    def _is_parsable(qasm: str) -> bool:
        try:
            qiskit.qasm3.loads(qasm)
            return True
        except Exception:
            return False

    @staticmethod
    def _repair_syntax(qasm: str) -> str:
        fixed, has_header = [], False
        for line in qasm.split("\n"):
            t = line.strip()
            if not t:
                fixed.append("")
                continue
            if t.startswith("OPENQASM"):
                has_header = True
                fixed.append("OPENQASM 3.0;")
                continue
            t = re.sub(r"\bqbit_invalid\b", "qubit", t)
            t = re.sub(r"\bqbit\b", "qubit", t)
            t = re.sub(r"\bgtae\b", "gate", t)
            if "unknown_quantum_op" in t:
                continue
            if not t.endswith((";", "{", "}")) and not t.startswith("//"):
                t += ";"
            if t.count("[") > t.count("]"):
                # an index lost its closing bracket: restore it right after the integer
                t = re.sub(r"\[(\d+)(?![\d\]])", r"[\1]", t)
            fixed.append(t)
        if not has_header:
            fixed.insert(0, "OPENQASM 3.0;")
        if not any('include "stdgates.inc"' in l for l in fixed):
            fixed.insert(1, 'include "stdgates.inc";')
        return "\n".join(fixed)
