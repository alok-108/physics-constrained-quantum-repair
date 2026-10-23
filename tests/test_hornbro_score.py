"""Scoring of HornBro outputs with the shared predicate (HornBro itself needs Linux and is not run here)."""

import json
import os

import pandas as pd

from src import hornbro_adapter, score_hornbro


def _meta(root):
    return pd.read_csv(os.path.join(root, "data", "metadata.csv"))


def test_selection_rule_matches_the_runner(root):
    from src.run_experiments import stratified_subset
    meta = _meta(root)
    assert list(hornbro_adapter.stratified(meta, 60).circuit_id) == list(stratified_subset(meta, 60, 20260915).circuit_id)


def test_scoring_accepts_a_correct_circuit_rejects_the_faulty_one_and_counts_failures(root, tmp_path, validator):
    meta = _meta(root).set_index("circuit_id")
    ok, bad = "circuit_0009", "circuit_0013"                       # one correct return, one faulty return
    ref = open(os.path.join(root, meta.loc[ok, "ground_truth_file"]), encoding="utf-8").read()
    bug = open(os.path.join(root, meta.loc["circuit_0009", "buggy_file"]), encoding="utf-8").read()
    (tmp_path / f"{ok}.qasm").write_text(ref, encoding="utf-8")
    (tmp_path / f"{bad}.qasm").write_text(open(os.path.join(root, meta.loc[bad, "buggy_file"]), encoding="utf-8").read(), encoding="utf-8")
    for cid, outcome in ((ok, "returned_circuit"), (bad, "returned_circuit"), ("circuit_0001", "input_unparsable")):
        (tmp_path / f"{cid}.json").write_text(json.dumps({"circuit_id": cid, "outcome": outcome, "hornbro_says_success": True,
                                                          "seconds": 1.0}))
    df = score_hornbro.score(str(tmp_path), os.path.join(root, "data", "metadata.csv"), validator).set_index("circuit_id")
    assert df.loc[ok, "pass_at_3"] == 1 and df.loc[bad, "pass_at_3"] == 0
    assert df.loc["circuit_0001", "pass_at_3"] == 0 and df.loc["circuit_0001", "failed_stage"] == "input_unparsable"
    assert bug  # the faulty text exists and differs from the reference
