"""Run buggy, repaired and reference circuits on a quantum device (src/run_hardware.py).

For each chosen benchmark circuit the script takes the faulty circuit, the circuit repaired by the tool
(``product.repair`` against the reference) and the reference, compiles all three with the same fixed layout
(virtual qubit i -> physical qubit i, the layout the acceptance predicate assumes), measures them, and reports the
total variation distance (TVD) of each measured distribution to the ideal output distribution of the reference.

    python src/run_hardware.py --dry-run --ids circuit_0001 circuit_0301     # FakeSherbrooke noisy simulator, no account
    IBM_QUANTUM_TOKEN=... python src/run_hardware.py --backend ibm_sherbrooke --shots 1000 --count 4

Real-device mode needs an IBM Quantum account token (environment variable ``IBM_QUANTUM_TOKEN``, optionally
``IBM_QUANTUM_INSTANCE``); the token is never printed or stored. All circuits go to the device as ONE job, and the
printed plan shows how many circuits and shots that is, because the free plan has a small monthly QPU allowance.

What this does and does not show: it measures, on noisy hardware or a noise model of it, whether the repaired
circuit's output is as close to the ideal output as the reference's is. A small circuit's output distribution does not
certify unitary equivalence; the exact check is the acceptance predicate. The real-device path was written against the
qiskit-ibm-runtime SamplerV2 interface and is exercised in this repository only through the simulator mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import qiskit.qasm3
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector


def ideal_distribution(reference: QuantumCircuit) -> Dict[str, float]:
    probs = Statevector.from_instruction(reference).probabilities_dict()
    return {k: v for k, v in probs.items() if v > 1e-12}


def tvd(counts: Dict[str, int], ideal: Dict[str, float]) -> float:
    shots = sum(counts.values())
    keys = set(counts) | set(ideal)
    return 0.5 * sum(abs(counts.get(k, 0) / shots - ideal.get(k, 0.0)) for k in keys)


def build_variants(meta_row: Dict, root: str) -> Dict[str, str]:
    from src import product
    read = lambda f: open(os.path.join(root, f), encoding="utf-8").read()
    ref, bug = read(meta_row["ground_truth_file"]), read(meta_row["buggy_file"])
    rep = product.repair(bug, reference=ref)
    variants = {"reference": ref, "buggy": bug}
    if rep.repaired_qasm:
        variants["repaired"] = rep.repaired_qasm
    return variants


def run(ids: List[str], backend_name: str, dry_run: bool, shots: int, root: str = PROJECT_ROOT) -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(root, "data", "metadata.csv")).set_index("circuit_id")
    if dry_run:
        from qiskit_aer import AerSimulator
        from src.hardware import backend as fake_backend
        target = fake_backend()
        runner = AerSimulator.from_backend(target)
    else:
        token = os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            raise SystemExit("set IBM_QUANTUM_TOKEN (an IBM Quantum account token) or use --dry-run")
        from qiskit_ibm_runtime import QiskitRuntimeService
        kw = {"channel": "ibm_quantum_platform", "token": token}
        if os.environ.get("IBM_QUANTUM_INSTANCE"):
            kw["instance"] = os.environ["IBM_QUANTUM_INSTANCE"]
        target = runner = QiskitRuntimeService(**kw).backend(backend_name)

    jobs, labels, ideals = [], [], {}
    for cid in ids:
        row = meta.loc[cid].to_dict()
        variants = build_variants(row, root)
        ideals[cid] = ideal_distribution(qiskit.qasm3.loads(variants["reference"]))
        for name, qasm in variants.items():
            try:
                qc = qiskit.qasm3.loads(qasm)
            except Exception:                           # a syntax fault: the faulty circuit cannot be submitted at all
                print(f"  {cid}: the {name} circuit does not parse, so it cannot be run", flush=True)
                continue
            qc.measure_all()
            tc = transpile(qc, backend=target, initial_layout=list(range(qc.num_qubits)), optimization_level=1,
                           seed_transpiler=0)
            jobs.append(tc)
            labels.append((cid, name, qc.num_qubits))
    print(f"plan: {len(ids)} circuits, {len(jobs)} compiled variants, {shots} shots each, backend "
          f"{'FakeSherbrooke (noisy simulator)' if dry_run else backend_name}", flush=True)

    if dry_run:
        counts = [runner.run(c, shots=shots, seed_simulator=1).result().get_counts() for c in jobs]
    else:
        from qiskit_ibm_runtime import SamplerV2
        res = SamplerV2(mode=runner).run(jobs, shots=shots).result()
        counts = [r.data.meas.get_counts() for r in res]

    rows = [{"circuit_id": cid, "variant": name, "num_qubits": n, "shots": shots, "tvd_to_ideal": round(tvd(c, ideals[cid]), 4),
             "backend": "FakeSherbrooke-simulator" if dry_run else backend_name}
            for (cid, name, n), c in zip(labels, counts)]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=None, help="benchmark circuit ids (default: the first --count small circuits)")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--max-qubits", type=int, default=4)
    ap.add_argument("--backend", default="ibm_sherbrooke")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true", help="noisy FakeSherbrooke simulator instead of a device")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "metadata.csv"))
    ids = a.ids
    if ids is None:                                     # one small circuit per fault class, spread over families
        small = meta[meta.num_qubits <= a.max_qubits]
        ids = list(small.groupby(["bug_type", "circuit_type"]).head(1).circuit_id)[: a.count]
    df = run(ids, a.backend, a.dry_run, a.shots)
    out = a.out or os.path.join(PROJECT_ROOT, "experiments",
                                "results_hardware_run_sim.csv" if a.dry_run else "results_hardware_run.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(df.pivot_table(index="circuit_id", columns="variant", values="tvd_to_ideal").round(3).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
