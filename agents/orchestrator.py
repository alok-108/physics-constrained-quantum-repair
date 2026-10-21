"""Feedback loop orchestrator (agents/orchestrator.py).

Coordinates Diagnostician -> Repairer -> Validator under a fixed attempt budget
K. Information boundary:

* the reference *source* reaches the ValidatorAgent only;
* the repair side receives a query-counted ``BehaviouralOracle`` built from the
  reference, never the reference itself, and never the injected fault label;
* every system runs through this same loop with the same K.

Independent switches define the ablation design:

``use_physics_context``  hardware knowledge in diagnosis and generation
                         (routing, coherence trimming).
``use_oracle``           whether the repair side gets the behavioural oracle at
                         all. Off = static repair only (syntax recovery, inverse
                         cancellation, routing); a lower bound that shows what
                         the oracle contributes.
``verbose_feedback``     after a rejection the repair side learns the rejecting
                         stage and -- for semantic rejections -- receives a
                         counterexample test (the pool state on which the
                         candidate disagrees most) that joins the oracle's test
                         suite. With it off, the repair side learns only that
                         the candidate was rejected.

The acceptance predicate itself is never ablated.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional, Set, Tuple

import qiskit.qasm3

from .diagnostician import DiagnosticianAgent, LEAKY_KEYS
from .oracle import BehaviouralOracle
from .repairer import RepairerAgent
from .validator import ValidatorAgent
from src.routing import circuit_key


def _digest(text: Optional[str]) -> Optional[str]:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class Orchestrator:
    """Bounded repair loop with an explicit unresolved outcome."""

    def __init__(self, max_iterations: int = 3, use_physics_context: bool = True,
                 verbose_feedback: bool = True, validator: Optional[ValidatorAgent] = None,
                 log_dir: Optional[str] = None, variant: str = "ours_full",
                 oracle_budget: int = 8000, use_oracle: bool = True,
                 oracle_suite_size: int = 4, router: str = "swapback"):
        self.max_iterations = max_iterations
        self.use_physics_context = use_physics_context
        self.verbose_feedback = verbose_feedback
        self.variant = variant
        self.oracle_budget = oracle_budget
        self.use_oracle = use_oracle
        self.oracle_suite_size = oracle_suite_size
        self.diagnostician = DiagnosticianAgent(use_physics_checks=use_physics_context)
        self.repairer = RepairerAgent(use_physics_context=use_physics_context, router=router)
        self.validator = validator or ValidatorAgent()
        self.log_dir = log_dir

    def run(self, circuit_id: str, buggy_qasm: str, ground_truth_qasm: str,
            circuit_spec: Optional[Dict[str, Any]] = None, write_log: bool = False) -> Dict[str, Any]:
        spec = {k: v for k, v in dict(circuit_spec or {}).items() if k not in LEAKY_KEYS}

        oracle = None
        if self.use_oracle:
            try:
                oracle = BehaviouralOracle(qiskit.qasm3.loads(ground_truth_qasm),
                                           query_budget=self.oracle_budget,
                                           suite_size=self.oracle_suite_size,
                                           seed_key=f"{circuit_id}:oracle")
            except Exception:
                oracle = None

        history = []
        feedback: Optional[str] = None
        current = buggy_qasm
        final_candidate: Optional[str] = None
        seen_digests: Set[str] = set()
        tabu: Set[Tuple] = set()
        success = False
        start = time.time()

        for it in range(1, self.max_iterations + 1):
            t0 = time.time()
            diagnosis = self.diagnostician.diagnose(buggy_qasm=current, circuit_spec=spec,
                                                    feedback=feedback)
            candidate = self.repairer.repair(buggy_qasm=current, bug_report=diagnosis,
                                             circuit_spec=spec, feedback=feedback,
                                             oracle=oracle, tabu=tabu)
            digest = _digest(candidate)
            repeated = digest in seen_digests
            seen_digests.add(digest)
            final_candidate = candidate
            cand_circuit = None
            try:
                cand_circuit = qiskit.qasm3.loads(candidate)
                tabu.add(circuit_key(cand_circuit))
            except Exception:
                pass

            verdict = self.validator.validate(
                repaired_qasm=candidate, ground_truth=ground_truth_qasm,
                redact=not self.verbose_feedback, gt_key=circuit_id)

            history.append({
                "iteration": it,
                "diagnosis_bug_type": diagnosis.get("bug_type"),
                "strategy": getattr(self.repairer, "last_strategy", "unknown"),
                "candidate_digest": digest,
                "candidate_repeated": repeated,
                "status": verdict["status"],
                "failed_stage": verdict.get("failed_stage"),
                "metrics": verdict.get("metrics", {}),
                "oracle_queries": oracle.queries_used if oracle is not None else 0,
                "duration_s": round(time.time() - t0, 3),
            })

            if verdict["status"] == "PASS":
                success = True
                break

            if self.verbose_feedback:
                feedback = (f"Attempt {it} rejected at stage '{verdict.get('failed_stage')}'. "
                            f"{verdict.get('feedback')}")
                if (verdict.get("failed_stage") == "semantic" and oracle is not None
                        and cand_circuit is not None):
                    oracle.counterexample(cand_circuit)
            else:
                feedback = None
            current = candidate

        gap = None
        if oracle is not None and final_candidate is not None:
            try:
                gap = oracle.holdout_gap(qiskit.qasm3.loads(final_candidate))
            except Exception:
                gap = None

        summary = {
            "circuit_id": circuit_id,
            "variant": self.variant,
            "success": success,
            "status": "PASS" if success else "UNRESOLVED",
            "iterations": len(history),
            "distinct_candidates": len(seen_digests),
            "oracle_queries": oracle.queries_used if oracle is not None else 0,
            "counterexamples": oracle.counterexamples_used if oracle is not None else 0,
            "holdout": gap,
            "total_duration_s": round(time.time() - start, 3),
            "final_candidate_qasm": final_candidate,
            "history": history,
        }
        if write_log and self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"{circuit_id}__{self.variant}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            summary["log_file"] = path
        return summary
