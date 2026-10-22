"""Calibration-based hardware estimate and the local-LLM plumbing (tested against a stub server)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import qiskit.qasm3
from qiskit import QuantumCircuit

from agents.orchestrator import Orchestrator
from agents.repairer import RepairerAgent
from src import cli, hardware


def ghz():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


# ------------------------------------------------------------------ hardware
def test_estimate_is_sane_and_uses_only_native_gates():
    for layout in ("identity", "auto"):
        e = hardware.estimate(ghz(), layout)
        assert 0.0 < e.esp <= 1.0 and e.estimated_duration_us > 0 and e.native_two_qubit_gates >= 2
        assert 0.0 < e.coherence_ratio < 1.0
    native = set(hardware.backend().target.operation_names)
    tc = hardware.compile_native(ghz(), "identity")
    assert {i.operation.name for i in tc.data} <= native | {"barrier"}


def test_a_longer_circuit_has_lower_esp():
    long_qc = ghz()
    for _ in range(6):
        long_qc.rz(0.3, 1)
        long_qc.cx(0, 1)
        long_qc.rz(0.2, 1)
        long_qc.cx(0, 1)
    assert hardware.estimate(long_qc).esp < hardware.estimate(ghz()).esp


def test_cli_hardware_command(tmp_path, capsys):
    f = tmp_path / "g.qasm"
    f.write_text(qiskit.qasm3.dumps(ghz()))
    assert cli.main(["hardware", str(f), "--layout", "identity", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["layout"] == "identity" and 0 < data[0]["esp"] <= 1


# ------------------------------------------------------------------ LLM plumbing
class Stub(BaseHTTPRequestHandler):
    reply = ""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"message": {"content": Stub.reply}, "prompt_eval_count": 11, "eval_count": 23}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def stub_server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("QREPAIR_OLLAMA_HOST", f"http://127.0.0.1:{srv.server_address[1]}")
    yield Stub
    srv.shutdown()


def buggy_text():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(1, 0)
    qc.cx(1, 2)
    return qiskit.qasm3.dumps(qc)


def test_llm_only_agent_returns_the_llm_circuit_and_counts_tokens(stub_server):
    stub_server.reply = "```qasm\n" + qiskit.qasm3.dumps(ghz()) + "\n```"
    agent = RepairerAgent(llm_model="stub", llm_only=True)
    out = agent.repair(buggy_text(), {"bug_type": "structural"}, {"num_qubits": 3})
    assert agent.last_strategy == "llm" and "cx q[0], q[1]" in out
    assert agent.llm.calls == 1 and agent.llm.tokens == 34


def test_llm_only_agent_does_not_fall_back_to_search_when_the_llm_fails(stub_server):
    stub_server.reply = "I cannot help with that."
    agent = RepairerAgent(llm_model="stub", llm_only=True)
    text = buggy_text()
    assert agent.repair(text, {"bug_type": "structural"}, {"num_qubits": 3}) == text
    assert agent.last_strategy == "llm_failed"


def test_orchestrator_llm_arm_reports_tokens_and_uses_no_oracle(stub_server, validator):
    stub_server.reply = "```qasm\n" + qiskit.qasm3.dumps(ghz()) + "\n```"
    orch = Orchestrator(validator=validator, use_oracle=False, llm_model="stub", llm_only=True)
    res = orch.run("stub_run", buggy_text(), qiskit.qasm3.dumps(ghz()), {"num_qubits": 3})
    assert res["oracle_queries"] == 0 and res["llm_calls"] >= 1 and res["llm_tokens"] >= 34
    assert res["status"] == "PASS"


def test_llm_token_counts_are_per_run_not_cumulative(stub_server, validator):
    stub_server.reply = "```qasm\n" + qiskit.qasm3.dumps(ghz()) + "\n```"
    orch = Orchestrator(validator=validator, use_oracle=False, llm_model="stub", llm_only=True)
    a = orch.run("r1", buggy_text(), qiskit.qasm3.dumps(ghz()), {"num_qubits": 3})
    b = orch.run("r2", buggy_text(), qiskit.qasm3.dumps(ghz()), {"num_qubits": 3})
    assert a["llm_tokens"] == b["llm_tokens"] == 34 and a["llm_calls"] == b["llm_calls"] == 1


def test_llm_only_arm_may_resubmit_the_same_circuit(stub_server):
    """No forced diversification for the LLM: a repeated reply is returned, not turned into a failure."""
    stub_server.reply = "```qasm\n" + buggy_text() + "\n```"
    agent = RepairerAgent(llm_model="stub", llm_only=True)
    first = agent.repair(buggy_text(), {"bug_type": "structural"}, {"num_qubits": 3})
    from src.routing import circuit_key
    tabu = {circuit_key(qiskit.qasm3.loads(first))}
    second = agent.repair(buggy_text(), {"bug_type": "structural"}, {"num_qubits": 3}, tabu=tabu)
    assert agent.last_strategy == "llm" and second == first
