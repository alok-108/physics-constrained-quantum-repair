"""A language model as the *controller* of verified repair tools (agents/llm_tools.py).

The plain LLM arms ask a model to write the repaired circuit. Here the model instead calls tools, one JSON object per
turn, and the tools do the checking:

    diagnose           static diagnosis (no reference involved)
    lint               static findings with probable lines
    localize           instructions ranked by the agreement gain of the best single edit there (queries the oracle)
    suggest_edits      the best-scoring single-instruction edits, each with an id and an oracle score (queries the oracle)
    apply_edit         apply one of the suggested edits
    fix_syntax / cancel_redundancy / route_to_device     the deterministic static repairs
    replace_circuit    the model writes the whole circuit itself (the old behaviour, still available)
    submit             run the acceptance predicate on the current circuit; it names the failing stage

Information boundary and budgets are those of the deterministic agent: the model never sees the reference, only the
behavioural oracle's scores (counted queries) and the validator's stage-named feedback; at most K = 3 submissions.

What this arm can and cannot show. Correctness comes from the verifier and the edit search, not from the model's
knowledge, so a high pass rate here says that a model can *drive* the tools, not that it can repair circuits. The
``TOOLS_GREEDY`` policy takes the same tools with no model at all (localize, apply the top edit, submit) and is the
ablation to compare against: the model has to beat or match it to be worth anything.
"""

from __future__ import annotations

import json
import re
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import qiskit
import qiskit.qasm3

from agents.diagnostician import DiagnosticianAgent, LEAKY_KEYS
from agents.llm import LLMError, QuotaExhausted
from agents.oracle import BehaviouralOracle
from agents.repairer import RepairerAgent
from agents.validator import ValidatorAgent
from src import product
from src.edits import apply_edit
from src.routing import circuit_key

MAX_SUBMISSIONS = 3
MAX_STEPS = 14

TOOL_HELP = """Tools (reply with exactly ONE JSON object per turn, no other text):
{"tool": "diagnose"}                          static diagnosis of the current circuit
{"tool": "lint"}                              static findings (syntax hint, non-adjacent gates, redundant gate pairs, ...)
{"tool": "localize"}                          the instructions where one edit would help most, with oracle gains
{"tool": "suggest_edits", "args": {"top": 5}} best single-instruction edits with ids and oracle scores (1.0 = matches the specification)
{"tool": "apply_edit", "args": {"id": 2}}     apply an edit from the latest suggest_edits list
{"tool": "fix_syntax"}                        repair a program that does not parse
{"tool": "cancel_redundancy"}                 remove cancelling gate pairs
{"tool": "route_to_device"}                   make two-qubit gates respect the device coupling map
{"tool": "replace_circuit", "args": {"qasm": "OPENQASM 3.0; ..."}}   write the whole circuit yourself
{"tool": "submit"}                            ask the validator to accept the current circuit (at most 3 submissions)"""

SYSTEM = ("You repair a faulty OpenQASM 3.0 quantum circuit for IBM FakeSherbrooke by calling tools. "
          "You never see the reference circuit; the oracle scores candidates against it. " + TOOL_HELP)


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """First JSON object in a reply (models wrap it in prose or code fences)."""
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
            args = obj.get("args")
            return {"tool": obj["tool"], "args": args if isinstance(args, dict) else {}}
    return None


def edit_text(qc, e) -> str:
    gate = getattr(e.gate, "name", None)
    params = ",".join(f"{float(p):.4g}" for p in getattr(e.gate, "params", []) if isinstance(p, (int, float)))
    g = f"{gate}({params})" if params else str(gate)
    qs = ",".join(map(str, e.qubits))
    if e.kind == "delete":
        return f"delete instruction {e.index} ({product.describe(qc, e.index)})"
    if e.kind == "replace":
        return f"replace instruction {e.index} ({product.describe(qc, e.index)}) with {g} q[{qs}]"
    return f"insert {g} q[{qs}] before instruction {e.index}"


class ToolSession:
    """State and tools for one repair; every tool returns a JSON-serialisable dict."""

    def __init__(self, buggy_qasm: str, ground_truth_qasm: str, circuit_id: str, spec: Dict[str, Any],
                 oracle_budget: int = 8000, suite_size: int = 4, validator: Optional[ValidatorAgent] = None):
        self.text = buggy_qasm
        self.reference = ground_truth_qasm
        self.circuit_id = circuit_id
        self.spec = {k: v for k, v in dict(spec).items() if k not in LEAKY_KEYS}
        self.validator = validator or ValidatorAgent()
        self.diagnostician = DiagnosticianAgent(use_physics_checks=True)
        self.agent = RepairerAgent(use_physics_context=True, extended_edits=True)
        try:
            self.oracle: Optional[BehaviouralOracle] = BehaviouralOracle(
                qiskit.qasm3.loads(ground_truth_qasm), query_budget=oracle_budget, suite_size=suite_size,
                seed_key=f"{circuit_id}:oracle")
        except Exception:
            self.oracle = None
        self.suggested: List[Any] = []
        self.history: List[Dict[str, Any]] = []
        self.seen: set = set()
        self.tool_counts: Dict[str, int] = {}
        self.submissions = 0
        self.done = False
        self.success = False
        self.feedback: Optional[str] = None
        self.counterexamples = 0

    # -- helpers -------------------------------------------------------------
    def _qc(self):
        try:
            return qiskit.qasm3.loads(self.text)
        except Exception:
            return None

    def _set(self, qc) -> None:
        self.text = qiskit.qasm3.dumps(qc)
        self.suggested = []

    def state(self) -> str:
        return f"Current circuit:\n```qasm\n{self.text}\n```"

    def queries(self) -> int:
        return self.oracle.queries_used if self.oracle is not None else 0

    # -- tools ---------------------------------------------------------------
    def diagnose(self, **_) -> Dict[str, Any]:
        d = self.diagnostician.diagnose(buggy_qasm=self.text, circuit_spec=self.spec, feedback=self.feedback)
        return {k: v for k, v in d.items() if k in ("bug_type", "details", "explanation", "issues", "hints")} or d

    def lint(self, **_) -> Dict[str, Any]:
        return {"findings": [{"code": f.code, "message": f.message, "line": getattr(f, "line", None)}
                             for f in product.lint(self.text)][:10]}

    def localize(self, **_) -> Dict[str, Any]:
        qc = self._qc()
        if qc is None or self.oracle is None:
            return {"error": "program does not parse (use fix_syntax or replace_circuit)" if qc is None else "no oracle"}
        if self.oracle.exhausted():
            return {"error": "oracle budget exhausted"}
        return {"ranked": product.localize(self.text, self.oracle, top=5)}

    def suggest_edits(self, top: int = 5, **_) -> Dict[str, Any]:
        qc = self._qc()
        if qc is None or self.oracle is None:
            return {"error": "program does not parse (use fix_syntax or replace_circuit)" if qc is None else "no oracle"}
        if self.oracle.exhausted():
            return {"error": "oracle budget exhausted"}
        top = max(1, min(int(top), 8))
        ranked = self.agent._neighbourhood(qc, self.oracle, rotations=True)[:top]
        self.suggested = [e for _, e in ranked]
        base = self.oracle.score(qc)
        return {"current_score": round(float(base), 4),
                "edits": [{"id": i, "score": round(float(s), 4), "edit": edit_text(qc, e)}
                          for i, (s, e) in enumerate(ranked)]}

    def apply_edit(self, id: Any = None, **_) -> Dict[str, Any]:
        qc = self._qc()
        try:
            e = self.suggested[int(id)]
        except (TypeError, ValueError, IndexError):
            return {"error": "no such edit id; call suggest_edits first"}
        if qc is None:
            return {"error": "program does not parse"}
        self._set(apply_edit(qc, e))
        return {"applied": edit_text(qc, e)}

    def fix_syntax(self, **_) -> Dict[str, Any]:
        if self._qc() is not None:
            return {"note": "the program already parses"}
        self.text = self.agent._repair_syntax(self.text)
        self.suggested = []
        return {"parses_now": self._qc() is not None}

    def cancel_redundancy(self, **_) -> Dict[str, Any]:
        from src.routing import cancel_inverses
        qc = self._qc()
        if qc is None:
            return {"error": "program does not parse"}
        out = cancel_inverses(qc)
        removed = len(qc.data) - len(out.data)
        self._set(out)
        return {"removed_instructions": removed}

    def route_to_device(self, **_) -> Dict[str, Any]:
        qc = self._qc()
        if qc is None:
            return {"error": "program does not parse"}
        self._set(self.agent._repair_physics(qc))
        return {"done": True}

    def replace_circuit(self, qasm: Any = None, **_) -> Dict[str, Any]:
        if not isinstance(qasm, str):
            return {"error": "args.qasm must be a string"}
        try:
            qiskit.qasm3.loads(qasm)
        except Exception as e:
            return {"error": f"does not parse: {str(e)[:120]}"}
        self.text = qasm
        self.suggested = []
        return {"replaced": True}

    def submit(self, **_) -> Dict[str, Any]:
        if self.submissions >= MAX_SUBMISSIONS:
            self.done = True
            return {"error": "no submissions left"}
        self.submissions += 1
        v = self.validator.validate(repaired_qasm=self.text, ground_truth=self.reference, redact=False, gt_key=self.circuit_id)
        digest = circuit_key(self._qc()) if self._qc() is not None else self.text
        self.history.append({"iteration": self.submissions, "diagnosis_bug_type": None, "strategy": "llm_tools",
                             "candidate_digest": str(hash(digest)), "candidate_repeated": digest in self.seen,
                             "status": v["status"], "failed_stage": v.get("failed_stage"), "metrics": v.get("metrics", {}),
                             "oracle_queries": self.queries()})
        self.seen.add(digest)
        if v["status"] == "PASS":
            self.success = self.done = True
            return {"status": "PASS"}
        self.feedback = f"Submission {self.submissions} rejected at stage '{v.get('failed_stage')}'. {v.get('feedback')}"
        qc = self._qc()
        if v.get("failed_stage") == "semantic" and self.oracle is not None and qc is not None:
            self.oracle.counterexample(qc)
            self.counterexamples += 1
        if self.submissions >= MAX_SUBMISSIONS:
            self.done = True
        return {"status": "FAIL", "failed_stage": v.get("failed_stage"), "feedback": v.get("feedback"),
                "submissions_left": MAX_SUBMISSIONS - self.submissions}

    TOOLS = ("diagnose", "lint", "localize", "suggest_edits", "apply_edit", "fix_syntax", "cancel_redundancy",
             "route_to_device", "replace_circuit", "submit")

    def call(self, action: Dict[str, Any]) -> Dict[str, Any]:
        name = action["tool"]
        if name not in self.TOOLS:
            return {"error": f"unknown tool {name!r}; tools: {', '.join(self.TOOLS)}"}
        self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
        try:
            return getattr(self, name)(**action.get("args", {}))
        except Exception as e:                      # a malformed call from the model is a tool error, not a crash
            return {"error": f"bad arguments or tool failure: {type(e).__name__}: {str(e)[:100]}"}


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
class GreedyPolicy:
    """No model: take the top suggested edit while it improves the oracle score, then submit (the ablation)."""

    name = "tools_greedy"

    def __init__(self):
        self.stage = 0
        self.last_score = -1.0
        self.pending: List[Dict[str, Any]] = []

    def first_message(self, session: ToolSession) -> None:
        self.stage, self.pending = 0, []                  # a policy object is reused across circuits: reset per run

    def act(self, session: ToolSession, last_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if self.pending:
            return self.pending.pop(0)
        if session._qc() is None:
            return {"tool": "fix_syntax", "args": {}}
        if last_result and "error" in last_result:
            return {"tool": "submit", "args": {}}
        if last_result and "edits" in last_result:
            edits = last_result["edits"]
            if edits and edits[0]["score"] > last_result["current_score"] + 1e-9:
                self.pending = [{"tool": "suggest_edits", "args": {"top": 1}}]
                return {"tool": "apply_edit", "args": {"id": edits[0]["id"]}}
            return {"tool": "submit", "args": {}}
        if last_result and "applied" in last_result:
            return {"tool": "suggest_edits", "args": {"top": 1}}
        if last_result and last_result.get("status") == "FAIL":
            return {"tool": "suggest_edits", "args": {"top": 1}}
        if self.stage == 0:
            self.stage = 1
            self.pending = [{"tool": "cancel_redundancy", "args": {}}, {"tool": "route_to_device", "args": {}},
                            {"tool": "suggest_edits", "args": {"top": 1}}]
            return self.pending.pop(0)
        return {"tool": "suggest_edits", "args": {"top": 1}}


class LLMPolicy:
    """A chat model picks the next tool. The transcript is the conversation; results are fed back as user turns."""

    name = "llm_tools"

    def __init__(self, llm):
        self.llm = llm
        self.messages: List[Any] = []
        self.invalid = 0
        self.errors = 0

    def first_message(self, session: ToolSession) -> None:
        self.invalid = self.errors = 0                    # per-run counters (the object is reused across circuits)
        self.messages = [SimpleNamespace(type="system", content=SYSTEM),
                         SimpleNamespace(type="human", content=(
                             f"Task specification:\n{session.spec}\n\n{session.state()}\n\n"
                             "Hardware: IBM FakeSherbrooke (127 qubits), max depth 250. Call a tool."))]

    def act(self, session: ToolSession, last_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if last_result is not None:
            self.messages.append(SimpleNamespace(type="human", content=(
                f"Tool result:\n{json.dumps(last_result, default=str)[:2500]}\n\n{session.state()}\n\nNext tool call (JSON only)?")))
        try:
            reply = self.llm.invoke(self.messages).content.strip()
        except QuotaExhausted:
            raise
        except LLMError:
            self.errors += 1
            return {"tool": "__llm_error__", "args": {}}
        self.messages.append(SimpleNamespace(type="ai", content=reply))
        act = parse_action(reply)
        if act is None:
            self.invalid += 1
            return {"tool": "__invalid__", "args": {}}
        return act


class LLMToolAgent:
    """Runs a policy against a ToolSession and returns the same summary dict as ``Orchestrator.run``."""

    def __init__(self, policy, validator: Optional[ValidatorAgent] = None, oracle_budget: int = 8000,
                 suite_size: int = 4, max_steps: int = MAX_STEPS):
        self.policy, self.validator = policy, validator
        self.oracle_budget, self.suite_size, self.max_steps = oracle_budget, suite_size, max_steps

    def run(self, circuit_id: str, buggy_qasm: str, ground_truth_qasm: str,
            circuit_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        llm = getattr(self.policy, "llm", None)
        t0 = time.time()
        calls0, tokens0 = (getattr(llm, "calls", 0), getattr(llm, "tokens", 0)) if llm is not None else (0, 0)
        s = ToolSession(buggy_qasm, ground_truth_qasm, circuit_id, circuit_spec or {}, self.oracle_budget,
                        self.suite_size, self.validator)
        self.policy.first_message(s)
        last: Optional[Dict[str, Any]] = None
        steps = 0
        while not s.done and steps < self.max_steps:
            steps += 1
            action = self.policy.act(s, last)
            if action["tool"] in ("__invalid__", "__llm_error__"):
                last = {"error": "reply was not a valid JSON tool call" if action["tool"] == "__invalid__" else "temporary model error"}
                if action["tool"] == "__llm_error__" and getattr(self.policy, "errors", 0) >= 2:
                    break
                continue
            last = s.call(action)
        if not s.history:                      # never submitted: judge what the session holds, as one attempt
            s.submit()
        elif not s.done and not s.success:     # ran out of steps with an unsubmitted change
            if s.history[-1]["candidate_digest"] != str(hash(circuit_key(s._qc()) if s._qc() is not None else s.text)):
                s.submit()
        return {
            "circuit_id": circuit_id, "variant": self.policy.name, "success": s.success,
            "status": "PASS" if s.success else "UNRESOLVED", "iterations": len(s.history),
            "distinct_candidates": len(s.seen), "oracle_queries": s.queries(), "counterexamples": s.counterexamples,
            "holdout": None,
            "llm_tokens": (getattr(llm, "tokens", 0) - tokens0) if llm is not None else 0,
            "llm_calls": (getattr(llm, "calls", 0) - calls0) if llm is not None else 0,
            "llm_errors": getattr(self.policy, "errors", 0),
            "llm_model": getattr(llm, "model", None), "tool_steps": steps, "tool_counts": dict(s.tool_counts),
            "invalid_replies": getattr(self.policy, "invalid", 0),
            "total_duration_s": round(time.time() - t0, 3), "final_candidate_qasm": s.text, "history": s.history,
        }
