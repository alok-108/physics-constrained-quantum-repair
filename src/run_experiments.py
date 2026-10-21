"""Benchmark runner (src/run_experiments.py).

Every system runs under the same acceptance predicate, the same attempt budget
K and the same information boundary. No outcome is hard-coded.

Arms
----
ours_full                  physics context ON,  diagnostic feedback ON
ours_reject_only           physics context ON,  feedback = bare rejection
unconstrained_diag         physics context OFF, diagnostic feedback ON
unconstrained_reject_only  physics context OFF, feedback = bare rejection
static_no_oracle           physics context ON,  no behavioural oracle (static repairs only)
pyzx_zx_calculus           ZX-calculus simplification (transformation baseline)

The first four form a 2x2 factorial design (physics x feedback). Extra arms (run
separately, see docs/PROTOCOL.md):

ours_full_qiskit_router    ours_full, but routing is done by Qiskit (SABRE + final-layout undo)
qiskit_transpile           Qiskit transpiler only (routing + O1 cancellation), no repair logic
qiskit_transpile_l0/l2/l3  the same compiler baseline at optimisation levels 0, 2 and 3

Usage
-----
    python src/run_experiments.py --workers 8                  # full 1000 circuits
    python src/run_experiments.py --limit 200 --workers 8      # stratified subset
    python src/run_experiments.py --resume                     # continue after interruption
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import qiskit
import qiskit.qasm2
import qiskit.qasm3

try:
    import pyzx as zx
except ImportError:                                   # pragma: no cover
    zx = None

BUDGET_K = 3

ARM_CONFIG: Dict[str, Dict[str, Any]] = {
    "ours_full":                 dict(use_physics_context=True,  verbose_feedback=True,  use_oracle=True),
    "ours_reject_only":          dict(use_physics_context=True,  verbose_feedback=False, use_oracle=True),
    "unconstrained_diag":        dict(use_physics_context=False, verbose_feedback=True,  use_oracle=True),
    "unconstrained_reject_only": dict(use_physics_context=False, verbose_feedback=False, use_oracle=True),
    "static_no_oracle":          dict(use_physics_context=True,  verbose_feedback=True,  use_oracle=False),
    # robustness check: same agent, but physical faults are routed by Qiskit's SABRE router
    "ours_full_qiskit_router":   dict(use_physics_context=True,  verbose_feedback=True,  use_oracle=True,
                                      router="qiskit"),
}
CORE_SYSTEMS = ["ours_full", "ours_reject_only", "unconstrained_diag", "unconstrained_reject_only",
                "static_no_oracle", "pyzx_zx_calculus"]
EXTRA_SYSTEMS = ["ours_full_qiskit_router", "qiskit_transpile", "qiskit_transpile_l0",
                 "qiskit_transpile_l2", "qiskit_transpile_l3"]
SYSTEMS = CORE_SYSTEMS + EXTRA_SYSTEMS


# --------------------------------------------------------------------------- #
# Resource and physics accounting
# --------------------------------------------------------------------------- #
def circuit_resources(qasm: Optional[str]) -> Dict[str, Optional[int]]:
    if not qasm:
        return {"gates": None, "depth": None}
    try:
        qc = qiskit.qasm3.loads(qasm)
        return {"gates": len(qc.data), "depth": qc.depth()}
    except Exception:
        return {"gates": None, "depth": None}


def physics_valid(qasm: Optional[str], physics) -> Optional[int]:
    """1 iff the final candidate parses and passes connectivity, coherence and depth."""
    if not qasm:
        return None
    try:
        qc = qiskit.qasm3.loads(qasm)
    except Exception:
        return 0
    conn, _ = physics.validate_circuit_connectivity(qc)
    return int(conn and physics.is_coherence_valid(qc) and physics.is_depth_valid(qc, max_depth=250))


# --------------------------------------------------------------------------- #
# PyZX arm
# --------------------------------------------------------------------------- #
def pyzx_candidate(qasm: str) -> Optional[str]:
    try:
        qc = qiskit.qasm3.loads(qasm)
    except Exception:
        return None
    try:
        if zx is not None:
            g = zx.Circuit.from_qasm(qiskit.qasm2.dumps(qc)).to_graph()
            zx.full_reduce(g)
            opt = zx.extract_circuit(g)
            return qiskit.qasm3.dumps(qiskit.qasm2.loads(opt.to_qasm()))
    except Exception:
        pass
    return None


def run_pyzx(circuit_id: str, buggy: str, gt: str, validator, budget: int) -> Dict[str, Any]:
    history, seen, final = [], set(), None
    success = False
    for it in range(1, budget + 1):
        cand = pyzx_candidate(buggy if it == 1 else (final or buggy))
        if cand is None:
            history.append({"iteration": it, "status": "FAIL", "failed_stage": "generation",
                            "candidate_digest": None, "candidate_repeated": False, "strategy": "pyzx"})
            continue
        dig = hashlib.sha256(cand.encode()).hexdigest()[:12]
        repeated = dig in seen
        seen.add(dig)
        final = cand
        v = validator.validate(repaired_qasm=cand, ground_truth=gt, gt_key=circuit_id)
        history.append({"iteration": it, "status": v["status"], "failed_stage": v.get("failed_stage"),
                        "candidate_digest": dig, "candidate_repeated": repeated, "strategy": "pyzx",
                        "metrics": v.get("metrics", {})})
        if v["status"] == "PASS":
            success = True
            break
    return {"success": success, "iterations": len(history), "distinct_candidates": len(seen),
            "oracle_queries": 0, "counterexamples": 0, "holdout": None,
            "final_candidate_qasm": final, "history": history}


def run_qiskit_transpile(circuit_id: str, buggy: str, gt: str, validator, budget: int,
                         level: int = 1) -> Dict[str, Any]:
    """Compiler-only baseline: Qiskit routing with exact layout bookkeeping, no repair logic."""
    from src.routing import compile_with_qiskit
    try:
        cand = qiskit.qasm3.dumps(compile_with_qiskit(qiskit.qasm3.loads(buggy), optimization_level=level))
    except Exception:
        cand = None
    history, seen, success = [], set(), False
    for it in range(1, budget + 1):
        if cand is None:
            history.append({"iteration": it, "status": "FAIL", "failed_stage": "generation",
                            "candidate_digest": None, "candidate_repeated": False, "strategy": "qiskit"})
            continue
        dig = hashlib.sha256(cand.encode()).hexdigest()[:12]
        repeated = dig in seen
        seen.add(dig)
        v = validator.validate(repaired_qasm=cand, ground_truth=gt, gt_key=circuit_id)
        history.append({"iteration": it, "status": v["status"], "failed_stage": v.get("failed_stage"),
                        "candidate_digest": dig, "candidate_repeated": repeated, "strategy": "qiskit",
                        "metrics": v.get("metrics", {})})
        if v["status"] == "PASS":
            success = True
            break
    return {"success": success, "iterations": len(history), "distinct_candidates": len(seen),
            "oracle_queries": 0, "counterexamples": 0, "holdout": None,
            "final_candidate_qasm": cand, "history": history}


# --------------------------------------------------------------------------- #
def row_from_run(meta: Dict[str, Any], system: str, res: Dict[str, Any], buggy: str,
                 physics, budget: int, duration: float) -> Dict[str, Any]:
    hist = res["history"]
    pass_at = {f"pass_at_{k}": int(any(h.get("status") == "PASS" for h in hist[:k]))
               for k in range(1, budget + 1)}
    final = res.get("final_candidate_qasm")
    before, after = circuit_resources(buggy), circuit_resources(final)
    accepted = bool(res["success"])
    gap = res.get("holdout") or {}
    final_metrics = (hist[-1].get("metrics", {}) if hist else {}) or {}
    ref_gates = int(meta["ref_gates"])
    return {
        "circuit_id": meta["circuit_id"], "circuit_type": meta["circuit_type"],
        "num_qubits": int(meta["num_qubits"]), "bug_type": meta["bug_type"],
        "fault_mode": meta["fault_mode"], "mode_family": meta["mode_family"],
        "system_name": system, **pass_at,
        "accepted": int(accepted),
        "iterations_used": res["iterations"],
        "distinct_candidates": res["distinct_candidates"],
        "oracle_queries": res.get("oracle_queries", 0),
        "counterexamples": res.get("counterexamples", 0),
        "repeated_candidate_any": int(any(h.get("candidate_repeated") for h in hist)),
        "final_failed_stage": None if accepted else (hist[-1].get("failed_stage") if hist else "generation"),
        "first_failed_stage": (hist[0].get("failed_stage") if hist else "generation"),
        "strategy": hist[-1].get("strategy") if hist else None,
        "physics_valid_final": physics_valid(final, physics),
        "process_fidelity_final": final_metrics.get("process_fidelity"),
        "holdout_suite": gap.get("suite"), "holdout_states": gap.get("holdout"),
        "holdout_process": gap.get("process"), "overfit_gap": gap.get("gap"),
        "gates_before": before["gates"], "gates_after": after["gates"], "ref_gates": ref_gates,
        "depth_before": before["depth"], "depth_after": after["depth"],
        "overhead_vs_ref_pct": (None if after["gates"] is None else 100.0 * (after["gates"] - ref_gates) / ref_gates),
        "duration_s": round(duration, 3),
    }


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
_STATE: Dict[str, Any] = {}


def _init_worker(systems: List[str], budget: int, oracle_budget: int, suite_size: int) -> None:
    warnings.filterwarnings("ignore")
    from agents.orchestrator import Orchestrator
    from agents.validator import ValidatorAgent
    from src.physics_model import get_default_physics_model
    validator = ValidatorAgent()
    _STATE["validator"] = validator
    _STATE["physics"] = get_default_physics_model()
    _STATE["budget"] = budget
    _STATE["systems"] = systems
    _STATE["orch"] = {
        name: Orchestrator(max_iterations=budget, validator=validator, variant=name,
                           oracle_budget=oracle_budget, oracle_suite_size=suite_size, **ARM_CONFIG[name])
        for name in systems if name in ARM_CONFIG
    }


def _run_circuit(meta: Dict[str, Any], root: str) -> List[Dict[str, Any]]:
    buggy = open(os.path.join(root, meta["buggy_file"]), encoding="utf-8").read()
    gt = open(os.path.join(root, meta["ground_truth_file"]), encoding="utf-8").read()
    spec = {"circuit_type": meta["circuit_type"], "num_qubits": int(meta["num_qubits"])}
    rows = []
    for system in _STATE["systems"]:
        t0 = time.time()
        if system == "pyzx_zx_calculus":
            res = run_pyzx(meta["circuit_id"], buggy, gt, _STATE["validator"], _STATE["budget"])
        elif system.startswith("qiskit_transpile"):
            level = 1 if system == "qiskit_transpile" else int(system[-1])      # ..._l0 / _l2 / _l3
            res = run_qiskit_transpile(meta["circuit_id"], buggy, gt, _STATE["validator"],
                                       _STATE["budget"], level)
        else:
            res = _STATE["orch"][system].run(circuit_id=meta["circuit_id"], buggy_qasm=buggy,
                                             ground_truth_qasm=gt, circuit_spec=spec)
        rows.append(row_from_run(meta, system, res, buggy, _STATE["physics"], _STATE["budget"],
                                 time.time() - t0))
    return rows


# --------------------------------------------------------------------------- #
def stratified_subset(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Stratify on fault class x circuit family so every cell is represented."""
    cells = df.groupby(["bug_type", "circuit_type"], sort=True)
    per = max(1, limit // cells.ngroups)
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in cells]
    return pd.concat(parts).sort_values("circuit_id").reset_index(drop=True)


def run_experiments(metadata_path: str, output_csv: str, systems: List[str],
                    limit: Optional[int] = None, budget: int = BUDGET_K, seed: int = 20260915,
                    workers: int = 1, resume: bool = False, oracle_budget: int = 8000,
                    suite_size: int = 4) -> pd.DataFrame:
    root = os.path.abspath(os.path.join(os.path.dirname(metadata_path), ".."))
    meta = pd.read_csv(metadata_path)
    if limit:
        meta = stratified_subset(meta, limit, seed)

    partial = output_csv + ".partial.jsonl"
    done: Dict[str, List[Dict[str, Any]]] = {}
    if resume and os.path.exists(partial):
        with open(partial, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["circuit_id"]] = rec["rows"]
        print(f"Resuming: {len(done)} circuits already complete.")
    elif os.path.exists(partial):
        os.remove(partial)

    todo = [m for m in meta.to_dict("records") if m["circuit_id"] not in done]
    print(f"Evaluating {len(todo)} circuits x {len(systems)} systems, budget K={budget}, "
          f"oracle budget={oracle_budget}, suite size={suite_size}, workers={workers}.")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    t0 = time.time()
    with open(partial, "a", encoding="utf-8") as sink:
        def record(cid, rows):
            done[cid] = rows
            sink.write(json.dumps({"circuit_id": cid, "rows": rows}, default=str) + "\n")
            sink.flush()
            if len(done) % 25 == 0:
                print(f"  {len(done)}/{len(meta)} circuits ({time.time() - t0:.0f}s)", flush=True)

        if workers <= 1:
            _init_worker(systems, budget, oracle_budget, suite_size)
            for m in todo:
                record(m["circuit_id"], _run_circuit(m, root))
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                     initargs=(systems, budget, oracle_budget, suite_size)) as pool:
                futs = {pool.submit(_run_circuit, m, root): m["circuit_id"] for m in todo}
                for fut in as_completed(futs):
                    record(futs[fut], fut.result())

    rows = [r for cid in sorted(done) for r in done[cid]]
    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    print(f"\nWrote {len(out)} rows to {output_csv} in {time.time() - t0:.0f}s")
    summarise(out, budget)
    return out


def summarise(df: pd.DataFrame, budget: int = BUDGET_K) -> None:
    cols = [f"pass_at_{k}" for k in range(1, budget + 1)]
    agg = df.groupby("system_name")[cols + ["physics_valid_final", "iterations_used", "oracle_queries"]].mean()
    agg[cols + ["physics_valid_final"]] *= 100
    print("\n--- Acceptance (%) under the shared predicate ---")
    print(agg.round(2).to_string())
    print(f"\n--- Pass@{budget} (%) by fault class ---")
    print((df.pivot_table(index="bug_type", columns="system_name",
                          values=f"pass_at_{budget}", aggfunc="mean") * 100).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=os.path.join(PROJECT_ROOT, "data", "metadata.csv"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "experiments", "results.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=int, default=BUDGET_K)
    ap.add_argument("--systems", nargs="+", default=CORE_SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--oracle-budget", type=int, default=8000, help="max oracle queries per run")
    ap.add_argument("--suite-size", type=int, default=4, help="initial behavioural test-suite size")
    a = ap.parse_args()
    run_experiments(a.metadata, a.out, a.systems, a.limit, a.budget, a.seed, a.workers, a.resume,
                    a.oracle_budget, a.suite_size)
