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
