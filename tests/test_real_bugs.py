"""The Bugs4Q real-bug pilot: the pairs are genuine faults; guided repair fixes three of five and misses two."""

import os

import pandas as pd
import pytest

from src import dynamic as D
from src.eval_real import evaluate


@pytest.fixture(scope="module")
def meta(root):
    return pd.read_csv(os.path.join(root, "data_real", "metadata.csv"))


def test_five_pairs_with_provenance(meta):
    assert len(meta) == 5 and meta.bugs4q_entry.is_unique
    assert all(str(s).endswith(".py") for s in meta.buggy_script)
    assert set(meta.device_check) == {0, 1} and list(meta[meta.device_check == 0].circuit_id) == ["real_04", "real_05"]


def test_every_buggy_circuit_is_rejected_and_every_reference_accepts_itself(meta, root):
    for m in meta.to_dict("records"):
        cid, dev = m["circuit_id"], bool(m["device_check"])
        bug = open(os.path.join(root, "data_real", "pairs", f"{cid}_buggy.qasm"), encoding="utf-8").read()
        ref = open(os.path.join(root, "data_real", "pairs", f"{cid}_reference.qasm"), encoding="utf-8").read()
        assert D.validate(ref, D.parse(ref), device=dev)["status"] == "PASS", cid
        assert D.validate(bug, D.parse(ref), device=dev)["status"] == "FAIL", cid


def test_reference_with_non_adjacent_gates_needs_logic_only_acceptance(meta, root):
    ref = open(os.path.join(root, "data_real", "pairs", "real_04_reference.qasm"), encoding="utf-8").read()
    assert D.validate(ref, D.parse(ref))["failed_stage"] == "connectivity"
    assert D.validate(ref, D.parse(ref), device=False)["status"] == "PASS"


def test_guided_repair_fixes_three_real_bugs_and_misses_two(root):
    df = evaluate(root)
    full = df[df.arm == "dyn_full"].set_index("circuit_id").accepted
    assert full[["real_01", "real_02", "real_03"]].all()
    assert not full[["real_04", "real_05"]].any()          # recorded limits, not hidden: see data_real/README.md
