"""Agent 1: Diagnostician (agents/diagnostician.py).

Fault localisation on faulty OpenQASM circuits. Produces a structured report
(bug type, location, description, suggested fix) from parser diagnostics,
structural checks and -- when ``use_physics_checks`` is on -- device checks.

Information boundary: the experimental annotations of the benchmark (injected
fault class, description, reference path) are stripped from the specification
before diagnosis; only algorithm family and qubit count survive.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    ChatOpenAI = None

import qiskit
import qiskit.qasm3

from src.physics_model import get_default_physics_model
from src.routing import cancel_inverses

LEAKY_KEYS = ("bug_type", "bug_description", "fault_mode", "mode_family",
              "ground_truth_file", "valid_file", "buggy_file", "ref_gates")


class DiagnosticianAgent:
    """Diagnoses syntax, structural, redundancy, semantic and physics faults."""

    SYSTEM_PROMPT = (
        "You are a quantum circuit debugger. Analyze the faulty circuit and identify the fault. "
        "Consider both logical errors and physics violations.\n"
        "You must output ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "bug_type": "syntax" | "structural" | "gate_redundancy" | "semantic_deviation" | "physics_violation",\n'
        '  "location": "line or gate index/description",\n'
        '  "description": "clear explanation of the fault",\n'
        '  "suggested_fix": "specific action needed to repair the circuit"\n'
        "}"
    )

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0,
                 use_physics_checks: bool = True):
        self.model_name = model_name
        self.temperature = temperature
        self.use_physics_checks = use_physics_checks
        self.physics = get_default_physics_model()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = None
        if self.api_key and ChatOpenAI is not None and not self.api_key.startswith("your-"):
            try:
                self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature,
                                      api_key=self.api_key)
            except Exception:
                self.llm = None

    # ------------------------------------------------------------------ #
    def diagnose(self, buggy_qasm: str, circuit_spec: Optional[Dict[str, Any]] = None,
                 feedback: Optional[str] = None) -> Dict[str, Any]:
        spec = {k: v for k, v in dict(circuit_spec or {}).items() if k not in LEAKY_KEYS}
        rule = self._rule_based_diagnosis(buggy_qasm, spec, feedback)

        if self.llm is not None:
            try:
                msg = (f"Circuit Specification:\n{json.dumps(spec, indent=2)}\n\n"
                       f"Faulty OpenQASM 3.0:\n```qasm\n{buggy_qasm}\n```\n\n")
                if feedback:
                    msg += f"Previous Validator Feedback:\n{feedback}\n\n"
                if rule:
                    msg += f"Static Analyzer Clues:\n{json.dumps(rule, indent=2)}\n\n"
                msg += "Provide the JSON bug report now."
                content = self.llm.invoke(
                    [SystemMessage(content=self.SYSTEM_PROMPT), HumanMessage(content=msg)]
                ).content.strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\n", "", content)
                    content = re.sub(r"\n```$", "", content)
                parsed = json.loads(content)
                if all(k in parsed for k in ("bug_type", "location", "description", "suggested_fix")):
                    return parsed
            except Exception:
                pass
        return rule

    # ------------------------------------------------------------------ #
    def _rule_based_diagnosis(self, buggy_qasm: str, spec: Dict[str, Any],
                              feedback: Optional[str]) -> Dict[str, Any]:
        # 1. Syntax
        try:
            qc = qiskit.qasm3.loads(buggy_qasm)
        except Exception as e:
            err = str(e)
            m = re.search(r"line (\d+)", err, re.IGNORECASE)
            loc = f"Line {m.group(1)}" if m else "Line unknown"
            return {"bug_type": "syntax", "location": loc,
                    "description": f"OpenQASM 3.0 syntax error: {err}",
                    "suggested_fix": "Fix syntax tokens, restore missing semicolons, "
                                     "correct keyword spelling, and match brackets."}

        # 2. Physics (skipped entirely when physics checks are off)
        if self.use_physics_checks:
            conn, violations = self.physics.validate_circuit_connectivity(qc)
            if not conn:
                return {"bug_type": "physics_violation",
                        "location": "Two-qubit gate operations",
                        "description": f"Hardware coupling violation on FakeSherbrooke: {violations[0]}",
                        "suggested_fix": "Route interactions through connected physical neighbours."}
            if not self.physics.is_coherence_valid(qc):
                return {"bug_type": "physics_violation",
                        "location": "Execution schedule / delay instruction",
                        "description": "Total execution time exceeds the T2 coherence budget.",
                        "suggested_fix": "Remove or shorten the idle delay."}

        # 3. Redundancy: any pair removable by inverse cancellation
        reduced = cancel_inverses(qc)
        if len(reduced.data) < len(qc.data):
            return {"bug_type": "gate_redundancy",
                    "location": f"{len(qc.data) - len(reduced.data)} cancellable instruction(s)",
                    "description": "Adjacent self-inverse gates cancel.",
                    "suggested_fix": "Remove the redundant gate pair."}

        # 4. Semantic / structural
        expected = spec.get("num_qubits", qc.num_qubits)
        if qc.num_qubits != expected:
            return {"bug_type": "structural", "location": "Register declaration",
                    "description": f"Qubit count mismatch: found {qc.num_qubits}, expected {expected}.",
                    "suggested_fix": f"Declare a register of exactly {expected} qubits."}
        if feedback and ("semantic" in feedback.lower() or "fidelity" in feedback.lower()):
            return {"bug_type": "semantic_deviation", "location": "Circuit body",
                    "description": f"Circuit output deviates from the specification: {feedback}",
                    "suggested_fix": "Correct a rotation angle, gate type or operand."}
        return {"bug_type": "structural", "location": "Circuit instruction sequence",
                "description": "Behaviour deviates from the specification.",
                "suggested_fix": "Search single-edit repairs ranked by the behavioural oracle."}
