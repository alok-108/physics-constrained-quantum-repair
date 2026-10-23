"""LLM-as-controller arm: tool session, greedy ablation, and a model driving the tools (stub server)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import qiskit.qasm3
from qiskit import QuantumCircuit

from agents.llm import OllamaChat
from agents.llm_tools import GreedyPolicy, LLMPolicy, LLMToolAgent, ToolSession, parse_action


def ghz():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qiskit.qasm3.dumps(qc)


def buggy():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(1, 0)                         # wrong direction
    qc.cx(1, 2)
    return qiskit.qasm3.dumps(qc)


def test_parse_action_finds_json_in_prose_and_fences():
    assert parse_action('Sure.\n```json\n{"tool": "submit"}\n```') == {"tool": "submit", "args": {}}
    assert parse_action('{"tool": "apply_edit", "args": {"id": 1}}')["args"] == {"id": 1}
    assert parse_action("no json here") is None and parse_action('{"nottool": 1}') is None


def test_session_tools_are_budgeted_and_hide_the_reference(validator):
    s = ToolSession(buggy(), ghz(), "t1", {"num_qubits": 3, "ground_truth_file": "LEAK", "bug_type": "x"}, validator=validator)
    assert set(s.spec) == {"num_qubits"}
    r = s.call({"tool": "suggest_edits", "args": {"top": 3}})
    assert r["edits"] and s.queries() > 0 and r["edits"][0]["score"] >= r["current_score"]
    assert "error" in s.call({"tool": "apply_edit", "args": {"id": 99}})
    assert "error" in s.call({"tool": "nonsense"})
    assert s.call({"tool": "apply_edit", "args": {"id": 0}}).get("applied")
    assert s.call({"tool": "submit"})["status"] == "PASS"


def test_session_allows_at_most_three_submissions(validator):
    s = ToolSession(buggy(), ghz(), "t2", {"num_qubits": 3}, validator=validator)
    for _ in range(3):
        assert s.call({"tool": "submit"})["status"] == "FAIL"
    assert s.done and "error" in s.call({"tool": "submit"})


def test_greedy_policy_repairs_without_any_model(validator):
    res = LLMToolAgent(GreedyPolicy(), validator=validator).run("g1", buggy(), ghz(), {"num_qubits": 3})
    assert res["success"] and res["llm_calls"] == 0 and res["oracle_queries"] > 0 and res["history"][-1]["status"] == "PASS"


class Scripted(BaseHTTPRequestHandler):
    replies = []
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        Scripted.seen.append(body["messages"])
        msg = Scripted.replies.pop(0) if Scripted.replies else '{"tool": "submit"}'
        raw = json.dumps({"message": {"content": msg}, "prompt_eval_count": 10, "eval_count": 5}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture()
def scripted(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Scripted)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Scripted.replies, Scripted.seen = [], []
    monkeypatch.setenv("QREPAIR_OLLAMA_HOST", f"http://127.0.0.1:{srv.server_address[1]}")
    yield Scripted
    srv.shutdown()


def test_model_drives_the_tools_and_invalid_replies_are_counted(scripted, validator):
    scripted.replies = ["I think I should look at the circuit.", '{"tool": "suggest_edits", "args": {"top": 3}}',
                        '{"tool": "apply_edit", "args": {"id": 0}}', '```json\n{"tool": "submit"}\n```']
    pol = LLMPolicy(OllamaChat("stub"))
    res = LLMToolAgent(pol, validator=validator).run("m1", buggy(), ghz(), {"num_qubits": 3})
    assert res["success"] and res["invalid_replies"] == 1 and res["llm_calls"] == 4 and res["llm_tokens"] == 60
    assert res["tool_counts"]["suggest_edits"] == 1 and res["variant"] == "llm_tools"
    first = scripted.seen[0]                                      # the model is told the task, never the reference
    assert "LEAK" not in json.dumps(first) and ghz() not in json.dumps(first)


def test_a_model_that_never_submits_is_still_judged_once(scripted, validator):
    scripted.replies = ['{"tool": "lint"}'] * 20
    res = LLMToolAgent(LLMPolicy(OllamaChat("stub")), validator=validator, max_steps=4).run("m2", buggy(), ghz(), {"num_qubits": 3})
    assert not res["success"] and res["iterations"] == 1 and res["tool_steps"] == 4


def test_policy_state_does_not_leak_between_circuits(validator):
    """One policy object serves every circuit of a run; the second circuit must behave like the first."""
    agent = LLMToolAgent(GreedyPolicy(), validator=validator)
    a = agent.run("p1", buggy(), ghz(), {"num_qubits": 3})
    red = QuantumCircuit(3)
    red.h(0), red.cx(0, 1), red.cx(1, 2), red.x(0), red.x(0)           # a redundant pair on top of a correct circuit
    b = agent.run("p2", qiskit.qasm3.dumps(red), ghz(), {"num_qubits": 3})
    assert a["success"] and b["success"] and b["tool_counts"].get("cancel_redundancy") == 1


def test_llm_counters_are_per_run(scripted, validator):
    scripted.replies = ["nonsense"] * 2 + ['{"tool": "submit"}'] * 3
    agent = LLMToolAgent(LLMPolicy(OllamaChat("stub")), validator=validator, max_steps=3)
    first = agent.run("c1", buggy(), ghz(), {"num_qubits": 3})
    scripted.replies = ['{"tool": "submit"}'] * 3
    second = agent.run("c2", buggy(), ghz(), {"num_qubits": 3})
    assert first["invalid_replies"] == 2 and second["invalid_replies"] == 0
