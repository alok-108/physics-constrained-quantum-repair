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


# ------------------------------------------------------------------ Gemini client (stub server)
class GeminiStub(BaseHTTPRequestHandler):
    reply = ""
    fail_first = 0                      # answer this many requests with 429 before succeeding
    delay = "0s"                        # the retryDelay carried by those 429s
    reject_system = False               # behave like Gemma: refuse systemInstruction
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        GeminiStub.seen.append({"path": self.path, "key": self.headers.get("x-goog-api-key"), "body": body})
        if GeminiStub.reject_system and "systemInstruction" in body:
            raw = json.dumps({"error": {"code": 400, "message": "Developer instruction is not enabled for this model"}}).encode()
            self.send_response(400)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if GeminiStub.fail_first > 0:
            GeminiStub.fail_first -= 1
            out, code = {"error": {"message": "quota, key=SECRETKEY", "details": [{"retryDelay": GeminiStub.delay}]}}, 429
        else:
            out = {"candidates": [{"content": {"parts": [{"text": GeminiStub.reply}]}}],
                   "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 25, "thoughtsTokenCount": 7}}
            code = 200
        raw = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture()
def gemini_stub(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), GeminiStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    GeminiStub.seen, GeminiStub.fail_first, GeminiStub.reply, GeminiStub.delay = [], 0, "", "0s"
    GeminiStub.reject_system = False
    monkeypatch.setenv("QREPAIR_GEMINI_BASE", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setenv("GEMINI_API_KEY", "SECRETKEY")
    monkeypatch.setenv("QREPAIR_LLM_MIN_INTERVAL", "0")
    yield GeminiStub
    srv.shutdown()


def _msgs(system="be exact", user="fix it"):
    from types import SimpleNamespace      # the clients only need .type and .content; langchain is not a requirement
    return [SimpleNamespace(type="system", content=system), SimpleNamespace(type="human", content=user)]


def test_gemini_request_shape_key_in_header_and_token_accounting(gemini_stub):
    from agents.llm import GeminiChat
    gemini_stub.reply = "ok"
    llm = GeminiChat("gemini-test")
    assert llm.invoke(_msgs()).content == "ok"
    req = gemini_stub.seen[0]
    assert req["path"] == "/v1beta/models/gemini-test:generateContent" and "SECRETKEY" not in req["path"]
    assert req["key"] == "SECRETKEY"
    assert req["body"]["systemInstruction"]["parts"][0]["text"] == "be exact"
    assert req["body"]["contents"] == [{"role": "user", "parts": [{"text": "fix it"}]}]
    assert req["body"]["generationConfig"]["temperature"] == 0.0
    assert (llm.calls, llm.prompt_tokens, llm.output_tokens, llm.thought_tokens) == (1, 40, 25, 7)
    assert llm.tokens == 72                          # thinking tokens count towards cost


def test_gemini_retries_on_429_then_succeeds(gemini_stub):
    from agents.llm import GeminiChat
    gemini_stub.reply, gemini_stub.fail_first = "ok", 2
    llm = GeminiChat("m", max_retries=3)
    llm.RETRYABLE = (429,)
    import time as _t
    real = _t.sleep
    _t.sleep = lambda s: None                        # do not actually wait in the test
    try:
        assert llm.invoke(_msgs()).content == "ok"
    finally:
        _t.sleep = real
    assert llm.retries == 2 and llm.rate_limited == 2 and llm.calls == 1 and llm.errors == 0


def test_gemini_gives_up_and_never_leaks_the_key(gemini_stub):
    from agents.llm import GeminiChat, LLMError
    gemini_stub.fail_first = 99
    llm = GeminiChat("m", max_retries=1)
    import time as _t
    real = _t.sleep
    _t.sleep = lambda s: None
    try:
        with pytest.raises(LLMError) as ei:
            llm.invoke(_msgs())
    finally:
        _t.sleep = real
    assert "SECRETKEY" not in str(ei.value) and llm.errors == 1


def test_gemini_needs_a_key(monkeypatch):
    from agents.llm import GeminiChat, LLMError
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiChat("m")


def test_gemini_arm_in_the_repair_agent_separates_api_failures_from_wrong_answers(gemini_stub):
    gemini_stub.reply = "```qasm\n" + qiskit.qasm3.dumps(ghz()) + "\n```"
    agent = RepairerAgent(llm_model="gemini:gemini-test", llm_only=True)
    out = agent.repair(buggy_text(), {"bug_type": "structural"}, {"num_qubits": 3})
    assert agent.last_strategy == "llm" and "cx q[0], q[1]" in out and agent.llm_errors == 0
    gemini_stub.fail_first = 99
    agent.llm.max_retries = 0
    assert agent.repair(buggy_text(), {"bug_type": "structural"}, {"num_qubits": 3}) == buggy_text()
    assert agent.last_strategy == "llm_failed" and agent.llm_errors == 1


def test_make_llm_dispatch(monkeypatch):
    from agents.llm import GeminiChat, OllamaChat, make_llm
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(make_llm("gemini:x"), GeminiChat) and make_llm("gemini:x").model == "x"
    assert isinstance(make_llm("ollama:qwen"), OllamaChat) and make_llm("ollama:qwen").model == "qwen"
    assert make_llm("qwen2.5-coder:1.5b").model == "qwen2.5-coder:1.5b"


def test_gemini_daily_quota_fails_fast_instead_of_burning_retries(gemini_stub):
    from agents.llm import GeminiChat, QuotaExhausted
    gemini_stub.fail_first, gemini_stub.delay = 99, "85706s"
    llm = GeminiChat("m", max_retries=6)
    with pytest.raises(QuotaExhausted, match="quota"):
        llm.invoke(_msgs())
    assert len(gemini_stub.seen) == 1 and llm.retries == 0           # no retries, no waiting


def test_quota_exhaustion_aborts_the_agent_run_rather_than_counting_as_a_wrong_answer(gemini_stub):
    from agents.llm import QuotaExhausted
    gemini_stub.fail_first, gemini_stub.delay = 99, "85706s"
    agent = RepairerAgent(llm_model="gemini:m", llm_only=True)
    with pytest.raises(QuotaExhausted):
        agent.repair(buggy_text(), {"bug_type": "structural"}, {"num_qubits": 3})


def test_repairer_works_without_langchain_installed():
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import sys
        sys.modules['langchain_core'] = None; sys.modules['langchain_core.messages'] = None
        sys.modules['langchain_openai'] = None
        from agents import repairer
        m = repairer.SystemMessage(content='x')
        assert (m.type, m.content) == ('system', 'x') and repairer.HumanMessage(content='y').type == 'human'
    """)
    root = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr


def test_gemini_inlines_the_system_prompt_for_models_that_refuse_systeminstruction(gemini_stub):
    from agents.llm import GeminiChat
    gemini_stub.reply, gemini_stub.reject_system = "ok", True
    llm = GeminiChat("gemma-like")
    assert llm.invoke(_msgs(system="be exact", user="fix it")).content == "ok"
    last = gemini_stub.seen[-1]["body"]
    assert "systemInstruction" not in last and last["contents"][0]["parts"][0]["text"] == "be exact\n\nfix it"
