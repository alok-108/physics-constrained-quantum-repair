"""LLM analysis: runs that stopped early are reported as partial and kept out of the tables."""

import pandas as pd

from src import analyze_llm as A


def _run(ids, passes, model):
    return pd.DataFrame({"circuit_id": ids, "pass_at_3": passes, "llm_model": model})


def test_partial_runs_are_detected_by_missing_common_circuits():
    common = {f"c{i}" for i in range(10)}
    full = _run(sorted(common), [1] * 10, "full")
    part = _run(sorted(common)[:4], [1] * 4, "part")
    partial = {m: d for m, d in {"full": full, "part": part}.items() if not common <= set(d.circuit_id)}
    assert list(partial) == ["part"]


def test_a_model_in_two_result_files_is_merged_not_an_error(tmp_path):
    import os
    os.makedirs(tmp_path / "llm_runs")
    pd.DataFrame({"circuit_id": ["a", "b"], "pass_at_3": [1, 0], "llm_model": "m"}).to_csv(tmp_path / "llm_runs" / "m_60.csv", index=False)
    pd.DataFrame({"circuit_id": ["b", "c"], "pass_at_3": [1, 1], "llm_model": "m"}).to_csv(tmp_path / "llm_runs" / "m_1000.csv", index=False)
    runs = A.load_runs(str(tmp_path))
    assert list(runs) == ["m"] and sorted(runs["m"].circuit_id) == ["a", "b", "c"]
    assert runs["m"].set_index("circuit_id").loc["b", "pass_at_3"] == 1      # first file (sorted by name) wins: m_1000.csv
