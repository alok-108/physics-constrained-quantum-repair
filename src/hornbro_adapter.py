"""Run HornBro (FSE 2025, https://github.com/JanusQ/HornBro, MIT) on this repository's benchmark circuits.

HornBro needs Linux, Python 3.10 and its own pinned-by-resolution dependencies, which conflict with the Qiskit used by
the rest of this repository. The comparison is therefore split into two steps that exchange OpenQASM 3 files only:

    step 1, in the HornBro environment (this file):
        python src/hornbro_adapter.py --metadata data/metadata.csv --limit 60 --out hornbro_out

    step 2, in this repository's environment (src/score_hornbro.py):
        python src/score_hornbro.py --dir hornbro_out

For each benchmark circuit HornBro receives the *reference* circuit and the faulty circuit as Qiskit objects, which is
the specification it is designed around (the reference serves as its test oracle), then its own repair loop
(``evaluation/efficiencyHornBro.py::repair``) runs unchanged. The circuit it returns is written as OpenQASM 3 and judged by
this repository's acceptance predicate in step 2, never by HornBro's own ``distance < 0.1`` criterion (which is recorded
too, for reference).

Limits to keep in mind: HornBro repairs gate parameters and a small set of gate patches of *unitary* circuits; faults that
leave the program unparsable (syntax class) cannot be given to it and count as failures with the reason recorded; this file
has been written against HornBro's public source and has not been run inside this repository's test suite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def stratified(meta, limit, seed=20260915):
    """Same selection rule as src/run_experiments.py (fault class x circuit family)."""
    cells = meta.groupby(["bug_type", "circuit_type"], sort=True)
    per = max(1, limit // cells.ngroups)
    import pandas as pd
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in cells]
    return pd.concat(parts).sort_values("circuit_id").reset_index(drop=True)


def _hornbro_repair(correct, buggy, tag, n_qubits, workdir):
    """Call HornBro's repair() and return (qiskit circuit, its own success flag, seconds)."""
    import evaluation.efficiencyHornBro as H                       # noqa: E402  (HornBro checkout on sys.path)
    from hornbro.circuit import Circuit

    stash = {}
    original = Circuit.to_json

    def capture(self):                                             # repair() ends with repaired_circuit.to_json()
        stash["circuit"] = self
        return original(self)

    Circuit.to_json = capture
    cwd = os.getcwd()
    os.chdir(workdir)                                              # repair() writes results/hornbro/... relative to cwd
    t0 = time.time()
    try:
        H.repair(correct, buggy, tag, n_qubits, 1, 0)
    finally:
        Circuit.to_json = original
        os.chdir(cwd)
    seconds = time.time() - t0
    metrics_path = os.path.join(workdir, "results", "hornbro", tag, f"{n_qubits}_errors_1", "metrics0.json")
    own = None
    if os.path.exists(metrics_path):
        own = json.load(open(metrics_path)).get("state") == "success"
    return stash["circuit"].to_qiskit(), own, seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=os.path.join(PROJECT_ROOT, "data", "metadata.csv"))
    ap.add_argument("--hornbro", default=os.environ.get("HORNBRO_DIR", "HornBro"), help="path to the HornBro checkout")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="hornbro_out")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.hornbro))
    import pandas as pd
    import qiskit.qasm3

    try:                                  # an environment problem must stop the run, not become 60 recorded "failures"
        import evaluation.efficiencyHornBro  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"HornBro cannot be imported in this environment ({type(e).__name__}: {e}). Install the missing "
                         "package into the HornBro environment (e.g. `uv pip install qiskit-aer`) and run again; nothing was written.")

    meta = stratified(pd.read_csv(a.metadata), a.limit)
    root = os.path.abspath(os.path.join(os.path.dirname(a.metadata), ".."))
    os.makedirs(a.out, exist_ok=True)
    work = os.path.abspath(os.path.join(a.out, "_work"))
    os.makedirs(work, exist_ok=True)
    status = {}
    for m in meta.to_dict("records"):
        cid = m["circuit_id"]
        done = os.path.join(a.out, cid + ".json")
        if os.path.exists(done):                                   # resumable; a record that only crashed is run again
            if json.load(open(done)).get("outcome") in ("returned_circuit", "input_unparsable"):
                continue
        rec = {"circuit_id": cid, "outcome": None, "hornbro_says_success": None, "seconds": None, "error": None}
        try:
            ref_t = open(os.path.join(root, m["ground_truth_file"]), encoding="utf-8").read()
            bug_t = open(os.path.join(root, m["buggy_file"]), encoding="utf-8").read()
            correct = qiskit.qasm3.loads(ref_t)
            try:
                buggy = qiskit.qasm3.loads(bug_t)
            except Exception as e:
                rec["outcome"], rec["error"] = "input_unparsable", f"faulty program does not parse: {type(e).__name__}"
                json.dump(rec, open(done, "w"))
                continue
            repaired, own, secs = _hornbro_repair(correct, buggy, "bench_" + cid, correct.num_qubits, work)
            open(os.path.join(a.out, cid + ".qasm"), "w", encoding="utf-8").write(qiskit.qasm3.dumps(repaired))
            rec.update(outcome="returned_circuit", hornbro_says_success=own, seconds=round(secs, 2))
        except ImportError:
            raise                                                  # environment problem: stop, do not record a failure
        except Exception as e:                                     # HornBro crashed or does not support the circuit
            rec["outcome"], rec["error"] = "crashed", f"{type(e).__name__}: {str(e)[:200]}"
            rec["trace"] = traceback.format_exc()[-600:]
        json.dump(rec, open(done, "w"))
        status[cid] = rec["outcome"]
        print(cid, rec["outcome"], rec.get("hornbro_says_success"), flush=True)
    print("done:", len(os.listdir(a.out)), "files in", a.out)


if __name__ == "__main__":
    main()
