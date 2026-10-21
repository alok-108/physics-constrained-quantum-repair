"""qrepair: command line interface (src/cli.py).

    python -m src.cli lint      circuit.qasm
    python -m src.cli verify    circuit.qasm (--reference ref.qasm | --tests tests.json)
    python -m src.cli localize  circuit.qasm (--reference ref.qasm | --tests tests.json) [--top 5]
    python -m src.cli repair    circuit.qasm (--reference ref.qasm | --tests tests.json) [--out fixed.qasm]

Add ``--json`` for machine-readable output. Exit status: 0 success (clean lint / PASS /
localised), 1 findings or failure, 2 usage or input error.

Test file format (amplitudes, little-endian Qiskit bitstrings, global phase ignored):

    {"num_qubits": 3,
     "tests": [{"input": "000", "expect": {"000": 0.7071, "111": 0.7071}},
               {"input": "100", "expect": {"100": 0.7071, "011": -0.7071}}]}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _spec(args):
    ref = _read(args.reference) if args.reference else None
    return ref, args.tests


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="qrepair", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_spec(p, required=True):
        g = p.add_mutually_exclusive_group(required=required)
        g.add_argument("--reference", help="reference circuit (OpenQASM 3)")
        g.add_argument("--tests", help="JSON test file of (input, expected output) states")

    p = sub.add_parser("lint", help="static checks, no specification needed")
    p.add_argument("circuit")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("verify", help="run the acceptance predicate")
    p.add_argument("circuit")
    add_spec(p)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("localize", help="rank suspicious instructions")
    p.add_argument("circuit")
    add_spec(p)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--budget", type=int, default=8000)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("repair", help="search for a minimal repair")
    p.add_argument("circuit")
    add_spec(p)
    p.add_argument("--out", help="write the repaired circuit here")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--budget", type=int, default=8000)
    p.add_argument("--no-physics", action="store_true", help="skip device-aware repairs (routing, delay trimming)")
    p.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    try:
        from src import product
        qasm = _read(args.circuit)

        if args.cmd == "lint":
            findings = product.lint(qasm)
            if args.json:
                print(json.dumps([asdict(f) for f in findings], indent=2))
            else:
                for f in findings:
                    loc = f" (instruction {f.instruction})" if f.instruction is not None else (
                        f" (line {f.line})" if f.line else "")
                    print(f"{f.severity.upper():7s} {f.code}{loc}: {f.message}")
            return 1 if any(f.severity in ("error", "warning") for f in findings) else 0

        ref, tests = _spec(args)
        if args.cmd == "verify":
            v = product.verify(qasm, ref, tests)
            if args.json:
                print(json.dumps(v, indent=2, default=str))
            else:
                print(f"{v['status']}" + (f" (rejected at stage '{v['failed_stage']}')" if v["failed_stage"] else ""))
                print(v["feedback"])
            return 0 if v["status"] == "PASS" else 1

        if args.cmd == "localize":
            oracle = product.tests_oracle(tests, args.budget) if tests else product.reference_oracle(ref, args.budget)
            rows = product.localize(qasm, oracle, args.top)
            if args.json:
                print(json.dumps(rows, indent=2))
            elif not rows:
                print("no single edit improves agreement with the specification")
            else:
                for r in rows:
                    print(f"#{r['rank']}  {r['kind']} {r['instruction']}: {r['text']}   "
                          f"[{r['suggested_edit']}, +{r['agreement_gain']:.3f}]")
            return 0

        rep = product.repair(qasm, ref, tests, args.attempts, args.budget, physics_context=not args.no_physics)
        if args.out and rep.repaired_qasm:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(rep.repaired_qasm)
        if args.json:
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(f"status: {rep.status}   attempts: {rep.attempts}   oracle queries: {rep.oracle_queries}")
            print(f"acceptance means: {rep.certificate}")
            for t in rep.trace:
                print(f"  attempt {t['attempt']}: {t['strategy']} -> {t['status']}"
                      + (f" ({t['failed_stage']})" if t.get("failed_stage") else ""))
            if rep.diff:
                print("\n" + rep.diff)
            for w in rep.warnings:
                print("warning:", w)
        return 0 if rep.status in ("PASS", "ALREADY_CORRECT") else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
