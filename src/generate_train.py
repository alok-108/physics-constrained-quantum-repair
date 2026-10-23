"""Fresh-seed training set with a pair-level held-out guarantee (src/generate_train.py).

Anyone who fine-tunes a model on repair traces needs training circuits that were never evaluated. The benchmark
(data/, seed 42) and benchmark v3 (data_v3/) are the *test* sets and are never used for training. This script
generates new circuits with a different seed from the same generator and enforces:

  * pair-level disjointness: a (faulty, reference) pair whose normalised text equals a pair in the benchmark
    (v2 or v3) is dropped;
  * a train / validation split that keeps all copies of one faulty circuit in one split.

What it does NOT guarantee, and the audit prints: the circuit *families* are deterministic (a 4-qubit GHZ circuit is
the same circuit in every set), so the *reference* circuits of the training set overlap with the benchmark's. A model
trained on this set has therefore seen the correct circuits for these families; only the faulty variants and the pairs
are new. Report that when you report results of a model trained on it.

    python src/generate_train.py --total 3000 --seed 7000001 --out train_set
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.generate_dataset import generate_dataset

BENCHMARK_SEED = 42


def _norm(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())


def digest(text: str) -> str:
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()[:16]


def pair_digests(root: str, meta_path: str, buggy_col: str, ref_col: str):
    meta = pd.read_csv(os.path.join(root, meta_path))
    pairs, refs, bugs = set(), set(), set()
    for m in meta.to_dict("records"):
        b = open(os.path.join(root, m[buggy_col]), encoding="utf-8").read()
        r = open(os.path.join(root, m[ref_col]), encoding="utf-8").read()
        pairs.add((digest(b), digest(r)))
        refs.add(digest(r))
        bugs.add(digest(b))
    return pairs, refs, bugs


def benchmark_digests(root: str = PROJECT_ROOT):
    p2, r2, b2 = pair_digests(root, "data/metadata.csv", "buggy_file", "ground_truth_file")
    p3, r3, b3 = pair_digests(root, "data_v3/metadata.csv", "buggy_file", "reference_file")
    return p2 | p3, r2 | r3, b2 | b3


def build(out: str, total: int, seed: int, val_fraction: float = 0.1, root: str = PROJECT_ROOT) -> pd.DataFrame:
    if seed == BENCHMARK_SEED:
        raise SystemExit("seed 42 regenerates the benchmark; use a different seed for training data")
    bench_pairs, bench_refs, bench_bugs = benchmark_digests(root)
    meta = generate_dataset(out, total, seed)
    rows = []
    for m in meta.to_dict("records"):
        b = open(os.path.join(out, m["buggy_file"]), encoding="utf-8").read()
        r = open(os.path.join(out, m["ground_truth_file"]), encoding="utf-8").read()
        db, dr = digest(b), digest(r)
        rows.append({**m, "buggy_digest": db, "reference_digest": dr,
                     "pair_in_benchmark": int((db, dr) in bench_pairs),
                     "buggy_in_benchmark": int(db in bench_bugs), "reference_in_benchmark": int(dr in bench_refs)})
    df = pd.DataFrame(rows)
    df = df[df.pair_in_benchmark == 0].drop_duplicates(["buggy_digest", "reference_digest"]).reset_index(drop=True)
    # split by faulty-circuit digest so no faulty text appears in both splits
    bucket = df.buggy_digest.map(lambda d: int(d, 16) % 1000 / 1000.0)
    df["split"] = ["val" if x < val_fraction else "train" for x in bucket]
    df.to_csv(os.path.join(out, "data", "metadata_train.csv"), index=False)
    return df


def audit(df: pd.DataFrame, total: int) -> str:
    n = len(df)
    lines = [f"generated {total}, kept {n} after dropping benchmark pairs and duplicates",
             f"splits: {df.split.value_counts().to_dict()}",
             f"faulty text also in the benchmark: {int(df.buggy_in_benchmark.sum())} ({100 * df.buggy_in_benchmark.mean():.1f} %)",
             f"reference text also in the benchmark: {int(df.reference_in_benchmark.sum())} ({100 * df.reference_in_benchmark.mean():.1f} %)",
             "pair-level overlap with the benchmark: 0 (enforced)",
             "by class: " + str(df.bug_type.value_counts().to_dict()),
             "by family: " + str(df.circuit_type.value_counts().to_dict())]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "train_set"))
    ap.add_argument("--total", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7000001)
    a = ap.parse_args()
    df = build(a.out, a.total, a.seed)
    print(audit(df, a.total))


if __name__ == "__main__":
    main()
