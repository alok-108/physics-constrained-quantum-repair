"""Benchmark audit (src/audit.py).

Checks the two properties a repair benchmark must have before any success rate
means something, using the validator's own physical checks:

1. every ground-truth circuit is admissible (parses, connectivity, coherence,
   depth) and irreducible (no cancellable inverse pair);
2. every injected fault is observable: syntax faults do not parse, structural /
   semantic faults change the unitary (process fidelity < 0.99) while the
   circuit stays admissible, redundancy faults are removable by inverse
   cancellation, physics faults fail a physical check.

Works on any dataset directory with a ``data/metadata.csv`` (including the
legacy v1 benchmark under ``legacy_v1/``).

    python src/audit.py                      # audit data/
    python src/audit.py --root legacy_v1     # audit the v1 benchmark
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional

import pandas as pd
import qiskit.qasm3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.generate_dataset import is_observable
from src.physics_model import get_default_physics_model
from src.routing import is_irreducible
from src.sim import process_fidelity


def audit_dataset(root: str, sample: Optional[int] = None, seed: int = 0) -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(root, "data", "metadata.csv"))
    if sample:
        per = max(1, sample // meta.bug_type.nunique())
        meta = pd.concat([g.sample(min(len(g), per), random_state=seed)
                          for _, g in meta.groupby("bug_type")])
    physics = get_default_physics_model()
    rows = []
    for m in meta.to_dict("records"):
        gt = qiskit.qasm3.loads(open(os.path.join(root, m["ground_truth_file"]), encoding="utf-8").read())
        btxt = open(os.path.join(root, m["buggy_file"]), encoding="utf-8").read()
        try:
            buggy = qiskit.qasm3.loads(btxt)
        except Exception:
            buggy = btxt
        # Admissibility-independent view of the fault (meaningful for any dataset):
        unitary_changed = None
        reducible_buggy = None
        if not isinstance(buggy, str):
            reducible_buggy = not is_irreducible(buggy)
            unitary_changed = (buggy.num_qubits != gt.num_qubits) or process_fidelity(buggy, gt) < 0.99
        conn, _ = physics.validate_circuit_connectivity(gt)
        coh = physics.is_coherence_valid(gt)
        dep = physics.is_depth_valid(gt, 250)
        rows.append({
            "circuit_id": m["circuit_id"], "circuit_type": m["circuit_type"], "bug_type": m["bug_type"],
            "ref_connectivity_ok": bool(conn), "ref_coherence_ok": bool(coh), "ref_depth_ok": bool(dep),
            "ref_irreducible": bool(is_irreducible(gt)),
            "fault_observable": bool(is_observable(m["bug_type"], buggy, gt, physics)),
            "unitary_changed": unitary_changed, "buggy_reducible": reducible_buggy,
        })
    out = pd.DataFrame(rows)
    out["ref_admissible"] = out.ref_connectivity_ok & out.ref_coherence_ok & out.ref_depth_ok
    return out


def summarise(df: pd.DataFrame) -> Dict[str, float]:
    sem = df[df.bug_type.isin(["structural", "semantic_deviation"])]
    red = df[df.bug_type == "gate_redundancy"]
    return {
        "records": len(df),
        "structural_semantic_faults_leaving_unitary_unchanged": int((sem.unitary_changed == False).sum()),  # noqa: E712
        "redundancy_faults_with_no_cancellable_pair": int((red.buggy_reducible == False).sum()),  # noqa: E712
        "reference_not_admissible": int((~df.ref_admissible).sum()),
        "reference_connectivity_violations": int((~df.ref_connectivity_ok).sum()),
        "reference_reducible": int((~df.ref_irreducible).sum()),
        "faults_not_observable": int((~df.fault_observable).sum()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=PROJECT_ROOT, help="directory containing data/metadata.csv")
    ap.add_argument("--sample", type=int, default=None)
    a = ap.parse_args()
    res = audit_dataset(a.root, a.sample)
    print(summarise(res))
    print("\nReference circuits failing physical checks, by family:")
    print(res.groupby("circuit_type")[["ref_connectivity_ok", "ref_irreducible"]].agg(lambda s: int((~s).sum())))
    print("\nUnobservable faults, by class:")
    print(res.groupby("bug_type").fault_observable.agg(lambda s: int((~s).sum())))
