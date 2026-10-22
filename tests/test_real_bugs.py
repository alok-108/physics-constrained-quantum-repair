"""The Bugs4Q real-bug pilot: the pairs are genuine faults and the repair fixes them."""

import os

import pandas as pd
import pytest

from src import dynamic as D
from src.eval_real import evaluate


@pytest.fixture(scope="module")
def meta(root):
    return pd.read_csv(os.path.join(root, "data_real", "metadata.csv"))


def test_three_pairs_with_provenance(meta):
    assert len(meta) == 3 and meta.bugs4q_entry.is_unique
    assert all(str(s).endswith("buggy.py") for s in meta.buggy_script)


def test_every_buggy_circuit_is_rejected_and_every_reference_accepts_itself(meta, root):
    for cid in meta.circuit_id:
        bug = open(os.path.join(root, "data_real", "pairs", f"{cid}_buggy.qasm"), encoding="utf-8").read()
        ref = open(os.path.join(root, "data_real", "pairs", f"{cid}_reference.qasm"), encoding="utf-8").read()
        assert D.validate(ref, D.parse(ref))["status"] == "PASS", cid
        assert D.validate(bug, D.parse(ref))["status"] == "FAIL", cid


def test_repair_fixes_the_real_bugs(root):
    df = evaluate(root)
    full = df[df.arm == "dyn_full"]
    assert len(full) == 3 and full.accepted.all(), full.to_string()
