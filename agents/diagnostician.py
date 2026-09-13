"""Agent 1: Diagnostician (agents/diagnostician.py)

Performs fault localization on buggy OpenQASM circuits.
Produces structured JSON bug reports identifying bug type, code location,
defect description, and suggested fix based on logical analysis and hardware constraints.
"""

from __future__ import annotations

import os
import json
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


class DiagnosticianAgent:
    """Diagnoses syntax, structural, gate redundancy, semantic, and physics bugs in quantum circuits."""

    SYSTEM_PROMPT = (
        "You are a quantum circuit debugger. Analyze buggy circuit and identify bug. "
        "Consider both logical errors and physics violations.\n"
        "You must output ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "bug_type": "syntax" | "structural" | "gate_redundancy" | "semantic_deviation" | "physics_violation",\n'
        '  "location": "line or gate index/description",\n'
        '  "description": "clear explanation of the bug",\n'
        '  "suggested_fix": "specific action needed to repair the circuit"\n'
        "}"
    )

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.physics = get_default_physics_model()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = None
        if self.api_key and ChatOpenAI is not None and not self.api_key.startswith("your-"):
            try:
                self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature, api_key=self.api_key)
            except Exception:
                self.llm = None

    def diagnose(
        self, 
        buggy_qasm: str, 
        circuit_spec: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """Diagnose a buggy OpenQASM circuit and return a structured bug report.
        
        Args:
            buggy_qasm: OpenQASM 3.0 string.
            circuit_spec: Optional metadata describing circuit type, expected qubits, etc.
            feedback: Validation failure feedback from previous repair attempt.
            
        Returns:
            Dict containing {bug_type, location, description, suggested_fix}
        """
        circuit_spec = circuit_spec or {}

        # 1. First run deterministic physics/syntax pre-checker for grounded analysis
        rule_diagnosis = self._rule_based_diagnosis(buggy_qasm, circuit_spec, feedback)

        # 2. If LLM is available, consult LLM with physical context
        if self.llm is not None:
            try:
                user_msg = (
                    f"Circuit Specification:\n{json.dumps(circuit_spec, indent=2)}\n\n"
                    f"Buggy OpenQASM 3.0:\n```qasm\n{buggy_qasm}\n```\n\n"
                )
                if feedback:
                    user_msg += f"Previous Validator Feedback:\n{feedback}\n\n"
                if rule_diagnosis:
                    user_msg += f"Static Analyzer Clues:\n{json.dumps(rule_diagnosis, indent=2)}\n\n"
                user_msg += "Provide the JSON bug report now."

                messages = [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=user_msg)
                ]
                response = self.llm.invoke(messages)
                content = response.content.strip()
                # Clean code blocks if present
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\n", "", content)
                    content = re.sub(r"\n```$", "", content)
                parsed = json.loads(content)
                if all(k in parsed for k in ["bug_type", "location", "description", "suggested_fix"]):
                    return parsed
            except Exception:
                pass  # Fall back to rule_diagnosis

        return rule_diagnosis

    def _rule_based_diagnosis(
        self, 
        buggy_qasm: str, 
        circuit_spec: Dict[str, Any], 
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """Grounded fallback diagnostician analyzing QASM syntax, Sherbrooke topology, and gate redundancy."""
        # 1. Syntax check
        try:
            qc = qiskit.qasm3.loads(buggy_qasm)
        except Exception as e:
            # Locate malformed line
            lines = buggy_qasm.split("\n")
            err_str = str(e)
            loc = "Line unknown"
            match = re.search(r"line (\d+)", err_str, re.IGNORECASE)
            if match:
                loc = f"Line {match.group(1)}"
            else:
                for idx, l in enumerate(lines):
                    if l.strip() and not l.strip().endswith(";") and not l.strip().endswith("{") and not l.strip().startswith("//"):
                        loc = f"Line {idx + 1}"
                        break
            return {
                "bug_type": "syntax",
                "location": loc,
                "description": f"OpenQASM 3.0 syntax error: {err_str}",
                "suggested_fix": "Fix syntax tokens, restore missing semicolons, correct keyword spelling, and match brackets."
            }

        # 2. Physics check: Connectivity & Coherence
        is_conn, violations = self.physics.validate_circuit_connectivity(qc)
        if not is_conn:
            return {
                "bug_type": "physics_violation",
                "location": "Two-qubit gate operations",
                "description": f"Hardware coupling violation on FakeSherbrooke: {violations[0]}",
                "suggested_fix": "Insert SWAP gates or reroute interaction through connected physical neighbors."
            }

        if not self.physics.is_coherence_valid(qc):
            return {
                "bug_type": "physics_violation",
                "location": "Circuit execution schedule / delay instruction",
                "description": "Total execution time exceeds hardware T2 coherence time on qubit.",
                "suggested_fix": "Remove or reduce uncalibrated idle delay to within coherence limits."
            }

        # 3. Gate redundancy check: consecutive self-canceling gates
        for idx in range(len(qc.data) - 1):
            inst1 = qc.data[idx]
            inst2 = qc.data[idx + 1]
            if inst1.operation.name == inst2.operation.name and inst1.qubits == inst2.qubits:
                if inst1.operation.name in ["x", "h", "z", "y", "cx"]:
                    q_indices = [qc.find_bit(q).index for q in inst1.qubits]
                    return {
                        "bug_type": "gate_redundancy",
                        "location": f"Instruction indices {idx} and {idx + 1}",
                        "description": f"Self-canceling duplicate {inst1.operation.name.upper()} gates on qubit(s) {q_indices}.",
                        "suggested_fix": f"Remove redundant consecutive {inst1.operation.name.upper()} pair."
                    }

        # 4. Semantic / Structural analysis
        if feedback and ("semantic" in feedback.lower() or "fidelity" in feedback.lower()):
            return {
                "bug_type": "semantic_deviation",
                "location": "Parameterized rotation gate",
                "description": f"Circuit output deviates from specification: {feedback}",
                "suggested_fix": "Correct rotation angle or gate basis to match specification."
            }

        # Default structural / semantic check
        expected_qubits = circuit_spec.get("num_qubits", qc.num_qubits)
        if qc.num_qubits != expected_qubits:
            return {
                "bug_type": "structural",
                "location": "Register declaration",
                "description": f"Qubit count mismatch: found {qc.num_qubits}, expected {expected_qubits}.",
                "suggested_fix": f"Declare register with exactly {expected_qubits} qubits."
            }

        return {
            "bug_type": "structural",
            "location": "Circuit instruction sequence",
            "description": "Gate order or entangling connectivity deviates from target specification.",
            "suggested_fix": "Reorder gates and ensure correct control/target qubit orientation."
        }
