"""Library API behind the ``qrepair`` command line tool (src/product.py).

Four operations on an OpenQASM 3 circuit:

* ``lint``      static checks (parse errors with line, device admissibility, cancellable
                pairs, no-op rotations, idle qubits); no specification needed.
* ``localize``  rank instructions by how much a single edit at that instruction would
                improve agreement with a specification (mutation-based fault localisation;
                the scores come from the same meet-in-the-middle oracle as the repair search).
* ``verify``    run the acceptance predicate against a specification.
* ``repair``    search for a minimal repair, with a unified diff and a machine-readable report.

A *specification* is either a reference circuit (strongest: exact unitary equivalence plus the
noise criterion) or a test file of (input state, expected output state) pairs (weaker: passing
the tests is not an equivalence certificate, and tests that only constrain output
probabilities cannot see phase faults, which is why tests are given as amplitudes).
"""

from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import qiskit.qasm3
from qiskit import QuantumCircuit

from agents.diagnostician import DiagnosticianAgent
from agents.oracle import BehaviouralOracle
from agents.orchestrator import Orchestrator
from agents.repairer import ANGLE_GATES, RepairerAgent
from agents.validator import ValidatorAgent
from src.physics_model import get_default_physics_model
from src.routing import cancel_inverses, circuit_key


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def describe(qc: QuantumCircuit, index: int) -> str:
    ins = qc.data[index]
    qs = ",".join(str(qc.find_bit(q).index) for q in ins.qubits)
    params = ",".join(f"{float(p):.4g}" for p in ins.operation.params if isinstance(p, (int, float)))
    return f"{ins.operation.name}({params}) q[{qs}]" if params else f"{ins.operation.name} q[{qs}]"


def normalised(qasm: str) -> str:
    """Canonical text of a circuit so diffs show semantic changes, not formatting."""
    try:
        return qiskit.qasm3.dumps(qiskit.qasm3.loads(qasm))
    except Exception:
        return qasm


def unified_diff(before: str, after: str) -> str:
    return "".join(difflib.unified_diff(normalised(before).splitlines(True), normalised(after).splitlines(True),
                                        "buggy.qasm", "repaired.qasm"))


# --------------------------------------------------------------------------- #
# lint
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    code: str
    severity: str            # "error" | "warning" | "info"
    message: str
    instruction: Optional[int] = None
    line: Optional[int] = None


_KEYWORDS = {"OPENQASM", "include", "qubit", "bit", "gate", "def", "if", "else", "for", "while", "measure",
             "reset", "barrier", "delay", "input", "output", "const", "int", "float", "angle", "let", "box", "return"}
_GATES = {"id", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz", "p", "u", "u1",
          "u2", "u3", "cx", "cy", "cz", "ch", "cp", "crx", "cry", "crz", "swap", "ccx", "cswap", "cu", "phase",
          "cphase", "gphase", "ecr", "rzz", "rxx", "ryy"}


def syntax_hint(qasm: str) -> Optional[Tuple[int, str]]:
    """Heuristic location of the first suspicious line when the parser gives no useful message."""
    for no, raw in enumerate(qasm.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//") or line.endswith(("{", "}")):
            continue
        if line.count("[") != line.count("]"):
            return no, "unbalanced bracket"
        if not line.endswith(";"):
            return no, "missing ';' at end of statement"
        head = re.split(r"[\s(\[;]", line, maxsplit=1)[0]
        if head and head not in _KEYWORDS and head not in _GATES and not head.startswith("//"):
            return no, f"unknown instruction or keyword '{head}'"
    return None


def lint(qasm: str, max_depth: int = 250) -> List[Finding]:
    """Static checks that need no specification."""
    try:
        qc = qiskit.qasm3.loads(qasm)
    except Exception as e:
        m = re.search(r"line (\d+)", str(e), re.IGNORECASE)
        text = ((str(e).splitlines() or [""])[0]).strip()
        line = int(m.group(1)) if m else None
        hint = syntax_hint(qasm)
        if hint:
            line = line or hint[0]
            text = f"{text + '; ' if text else ''}probably line {hint[0]}: {hint[1]}"
        return [Finding("E001", "error", f"OpenQASM 3 parse error: {text or 'not valid OpenQASM 3'}", line=line)]
    physics = get_default_physics_model()
    out: List[Finding] = []
    used = set()
    for i, ins in enumerate(qc.data):
        qs = [qc.find_bit(q).index for q in ins.qubits]
        used.update(qs)
        if len(qs) == 2 and not physics.is_connected(qs[0], qs[1]):
            out.append(Finding("W101", "warning", f"two-qubit gate on non-adjacent device qubits {qs}: "
                               f"{describe(qc, i)} (identity layout)", instruction=i))
        if ins.operation.name in ANGLE_GATES and ins.operation.params:
            ang = float(ins.operation.params[0])
            if abs(math.remainder(ang, 2 * math.pi)) < 1e-9 and ins.operation.name != "p":
                out.append(Finding("W202", "warning", f"rotation by a multiple of 2*pi has no effect: "
                                   f"{describe(qc, i)}", instruction=i))
    if not physics.is_coherence_valid(qc):
        out.append(Finding("W102", "warning", "estimated execution time exceeds the coherence budget "
                           "(check delay instructions)"))
    if qc.depth() > max_depth:
        out.append(Finding("W103", "warning", f"depth {qc.depth()} exceeds {max_depth}"))
    removed = len(qc.data) - len(cancel_inverses(qc).data)
    if removed:
        out.append(Finding("W201", "warning", f"{removed} instruction(s) cancel as adjacent inverse pairs"))
    idle = [q for q in range(qc.num_qubits) if q not in used]
    if idle:
        out.append(Finding("I301", "info", f"declared but unused qubit(s): {idle}"))
    out.append(Finding("I302", "info", f"{len(qc.data)} instructions, depth {qc.depth()}, "
                       f"{sum(1 for i in qc.data if len(i.qubits) == 2)} two-qubit gates"))
    return out


# --------------------------------------------------------------------------- #
# specifications
# --------------------------------------------------------------------------- #
def _bit_index(bits: str) -> int:
    if not re.fullmatch(r"[01]+", bits):
        raise ValueError(f"not a bitstring: {bits!r}")
    return int(bits, 2)                      # Qiskit convention: leftmost bit is the highest qubit


def _amplitude(v: Any) -> complex:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return complex(float(v[0]), float(v[1]))
    return complex(v)


def _state_from(spec: Any, n: int) -> np.ndarray:
    """A basis bitstring, or a sparse dict {bitstring: amplitude}, as a normalised vector."""
    vec = np.zeros(2 ** n, dtype=complex)
    if isinstance(spec, str):
        vec[_bit_index(spec)] = 1.0
    else:
        for bits, amp in spec.items():
            vec[_bit_index(bits)] = _amplitude(amp)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("zero state in test file")
    return vec / norm


def load_tests(path_or_dict) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parse a test file: {"num_qubits": n, "tests": [{"input": "000", "expect": {"000": 0.7071, "111": 0.7071}}]}."""
    data = path_or_dict
    if not isinstance(data, dict):
        with open(path_or_dict, encoding="utf-8") as f:
            data = json.load(f)
    n = int(data["num_qubits"])
    ins, outs = [], []
    for t in data["tests"]:
        ins.append(_state_from(t["input"], n))
        outs.append(_state_from(t["expect"], n))
    if not ins:
        raise ValueError("no tests")
    return np.stack(ins, axis=1), np.stack(outs, axis=1), n


def tests_oracle(tests, budget: int = 8000) -> BehaviouralOracle:
    inputs, expected, _ = load_tests(tests)
    return BehaviouralOracle.from_tests(inputs, expected, query_budget=budget)


def reference_oracle(reference_qasm: str, budget: int = 8000, key: str = "cli") -> BehaviouralOracle:
    return BehaviouralOracle(qiskit.qasm3.loads(reference_qasm), query_budget=budget, seed_key=key)


# --------------------------------------------------------------------------- #
# localize
# --------------------------------------------------------------------------- #
def localize(qasm: str, oracle: BehaviouralOracle, top: int = 5,
             extended: bool = True) -> List[Dict[str, Any]]:
    """Rank instructions by the best improvement any single edit there would give.

    ``wrong instruction`` entries come from replace/delete edits at that instruction;
    ``missing instruction before`` entries from insertions at that position. Gains are
    differences in mean suite agreement, so they reflect only what the tests can see.
    """
    qc = qiskit.qasm3.loads(qasm)
    base = oracle.score(qc)
    best: Dict[Tuple[str, int], Tuple[float, str]] = {}
    for score, edit in RepairerAgent(extended_edits=extended)._neighbourhood(qc, oracle, rotations=extended):
        gain = score - base
        if gain <= 1e-9:
            continue
        kind = "missing instruction before" if edit.kind == "insert" else "wrong instruction"
        key = (kind, edit.index)
        if key not in best or gain > best[key][0]:
            best[key] = (gain, edit.tag)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:top]
    out = []
    for r, ((kind, idx), (gain, tag)) in enumerate(ranked, 1):
        out.append({"rank": r, "kind": kind, "instruction": idx,
                    "text": describe(qc, idx) if idx < len(qc.data) else "(end of circuit)",
                    "suggested_edit": tag, "agreement_gain": round(gain, 6)})
    return out


# --------------------------------------------------------------------------- #
# verify / repair
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    status: str                          # "PASS" | "UNRESOLVED" | "ALREADY_CORRECT"
    certificate: str                     # what acceptance means for this run
    attempts: int
    repaired_qasm: Optional[str]
    diff: str
    gates_before: Optional[int]
    gates_after: Optional[int]
    oracle_queries: int
    trace: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CERT_REFERENCE = ("exact unitary equivalence to the reference (process fidelity >= 0.99), device "
                  "admissibility, irreducibility and noisy-output agreement below the 4096-shot noise floor")
CERT_TESTS = ("agreement >= 0.999 with every supplied test plus device admissibility and irreducibility; "
              "NOT an equivalence certificate")


def verify(qasm: str, reference: Optional[str] = None, tests=None,
           validator: Optional[ValidatorAgent] = None) -> Dict[str, Any]:
    validator = validator or ValidatorAgent()
    if reference is not None:
        return validator.validate(qasm, reference, gt_key="cli")
    if tests is not None:
        return validator.validate_tests(qasm, tests_oracle(tests))
    raise ValueError("give a reference circuit or a test file")


def _gates(qasm: str) -> Optional[int]:
    try:
        return len(qiskit.qasm3.loads(qasm).data)
    except Exception:
        return None


def repair(qasm: str, reference: Optional[str] = None, tests=None, max_attempts: int = 3,
           oracle_budget: int = 8000, physics_context: bool = True) -> Report:
    """Repair a faulty circuit against a reference circuit or a test file."""
    if (reference is None) == (tests is None):
        raise ValueError("give exactly one of reference / tests")
    validator = ValidatorAgent()
    before = _gates(qasm)
    warnings: List[str] = []
    try:                                     # fail fast and clearly on a register-width mismatch
        width = qiskit.qasm3.loads(qasm).num_qubits
        spec_width = (qiskit.qasm3.loads(reference).num_qubits if reference is not None
                      else load_tests(tests)[2])
        if width != spec_width:
            raise ValueError(f"circuit declares {width} qubits but the specification uses {spec_width}")
    except ValueError:
        raise
    except Exception:
        pass                                 # unparsable input: the syntax repair path handles it

    if reference is not None:
        orch = Orchestrator(max_iterations=max_attempts, validator=validator, oracle_budget=oracle_budget,
                            use_physics_context=physics_context, extended_edits=True)
        n = qiskit.qasm3.loads(reference).num_qubits
        res = orch.run("cli", qasm, reference, {"num_qubits": n})
        final = res["final_candidate_qasm"]
        trace = [{"attempt": h["iteration"], "strategy": h["strategy"], "status": h["status"],
                  "failed_stage": h["failed_stage"]} for h in res["history"]]
        ok = res["success"]
        if ok and res["iterations"] == 1 and normalised(final) == normalised(qasm):
            status = "ALREADY_CORRECT"
        else:
            status = "PASS" if ok else "UNRESOLVED"
        return Report(status, CERT_REFERENCE, res["iterations"], final if ok else None,
                      unified_diff(qasm, final) if ok else "", before, _gates(final) if ok else None,
                      res["oracle_queries"], trace, warnings)

    # ---- test-suite specification: no reference circuit, no counterexample pool, no noise stage
    oracle = tests_oracle(tests, oracle_budget)
    diag, rep = (DiagnosticianAgent(use_physics_checks=physics_context),
                 RepairerAgent(use_physics_context=physics_context, extended_edits=True))
    spec = {"num_qubits": oracle.num_qubits}
    warnings.append("test-suite specification: acceptance is not an equivalence certificate; add more "
                    "tests (including phase-sensitive ones) to reduce the risk of an overfitted repair")
    current, tabu, trace = qasm, set(), []
    for it in range(1, max_attempts + 1):
        report = diag.diagnose(current, spec, None)
        cand = rep.repair(current, report, spec, None, oracle=oracle, tabu=tabu)
        try:
            tabu.add(circuit_key(qiskit.qasm3.loads(cand)))
        except Exception:
            pass
        v = validator.validate_tests(cand, oracle)
        trace.append({"attempt": it, "strategy": rep.last_strategy, "status": v["status"],
                      "failed_stage": v["failed_stage"], "detail": v["feedback"]})
        if v["status"] == "PASS":
            status = "ALREADY_CORRECT" if (it == 1 and normalised(cand) == normalised(qasm)) else "PASS"
            return Report(status, CERT_TESTS, it, cand, unified_diff(qasm, cand), before, _gates(cand),
                          oracle.queries_used, trace, warnings)
        current = cand
    return Report("UNRESOLVED", CERT_TESTS, max_attempts, None, "", before, None, oracle.queries_used,
                  trace, warnings)
