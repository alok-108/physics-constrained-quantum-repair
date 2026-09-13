"""Feedback Loop Orchestrator (agents/orchestrator.py)

Coordinates multi-agent repair pipeline:
  Diagnostician -> Repairer -> Validator
Iteratively feeds validation feedback back to Diagnostician and Repairer on failure (max 3 iterations).
Logs detailed telemetry and step records to experiments/logs/{circuit_id}.json.
"""

from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, Optional, Union

import qiskit
from qiskit import QuantumCircuit

from .diagnostician import DiagnosticianAgent
from .repairer import RepairerAgent
from .validator import ValidatorAgent


class Orchestrator:
    """Multi-agent orchestrator managing the iterative repair feedback loop."""

    def __init__(
        self,
        max_iterations: int = 3,
        log_dir: Optional[str] = None
    ):
        self.max_iterations = max_iterations
        self.diagnostician = DiagnosticianAgent()
        self.repairer = RepairerAgent()
        self.validator = ValidatorAgent()
        
        # Determine log directory
        if log_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            self.log_dir = os.path.join(base_dir, "experiments", "logs")
        else:
            self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def run(
        self,
        circuit_id: str,
        buggy_qasm: str,
        ground_truth_qasm: str,
        circuit_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the multi-agent repair loop on a circuit.
        
        Args:
            circuit_id: Unique identifier for tracking and logging.
            buggy_qasm: Faulty OpenQASM 3.0 string.
            ground_truth_qasm: Ground truth OpenQASM 3.0 reference.
            circuit_spec: Metadata describing circuit type, expected qubits, etc.
            
        Returns:
            Dict containing overall status, iterations run, final repaired QASM, and log path.
        """
        circuit_spec = circuit_spec or {}
        history = []
        feedback = None
        current_buggy_qasm = buggy_qasm
        final_repaired_qasm = None
        success = False

        start_time = time.time()

        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()
            step_record = {
                "iteration": iteration,
                "input_feedback": feedback,
            }

            # 1. Diagnostician
            diagnosis = self.diagnostician.diagnose(
                buggy_qasm=current_buggy_qasm,
                circuit_spec=circuit_spec,
                feedback=feedback
            )
            step_record["diagnosis"] = diagnosis

            # 2. Repairer
            # Parse ground truth circuit hint if available
            gt_hint = None
            try:
                gt_hint = qiskit.qasm3.loads(ground_truth_qasm)
            except Exception:
                pass

            repaired_qasm = self.repairer.repair(
                buggy_qasm=current_buggy_qasm,
                bug_report=diagnosis,
                feedback=feedback,
                ground_truth_hint=gt_hint
            )
            step_record["repaired_qasm"] = repaired_qasm
            final_repaired_qasm = repaired_qasm

            # 3. Validator
            validation_result = self.validator.validate(
                repaired_qasm=repaired_qasm,
                ground_truth=ground_truth_qasm
            )
            step_record["validation"] = validation_result
            step_record["duration_s"] = round(time.time() - iter_start, 3)

            history.append(step_record)

            if validation_result.get("status") == "PASS":
                success = True
                break
            else:
                # Construct actionable feedback for next round
                feedback = (
                    f"Iteration {iteration} FAILED. Reason: {validation_result.get('reason')}. "
                    f"Feedback: {validation_result.get('feedback')}. "
                    f"Metrics: {validation_result.get('metrics')}"
                )
                current_buggy_qasm = repaired_qasm

        total_duration = round(time.time() - start_time, 3)

        result_summary = {
            "circuit_id": circuit_id,
            "success": success,
            "status": "PASS" if success else "FAIL",
            "iterations": len(history),
            "total_duration_s": total_duration,
            "final_repaired_qasm": final_repaired_qasm,
            "history": history
        }

        # Write log file
        log_file = os.path.join(self.log_dir, f"{circuit_id}.json")
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(result_summary, f, indent=2)

        result_summary["log_file"] = log_file
        return result_summary
