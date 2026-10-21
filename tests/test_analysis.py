"""Statistics helpers."""

import numpy as np
import pytest
from scipy import stats

from src.analyze_results import holm, mcnemar_exact, newcombe_paired, wilson


def test_wilson_matches_known_value():
    lo, hi = wilson(52, 100)
    assert lo == pytest.approx(0.4231, abs=2e-3) and hi == pytest.approx(0.6149, abs=2e-3)


def test_mcnemar_exact_matches_binomial_test():
    a = np.array([1] * 21 + [0] * 79)
    b = np.zeros(100, dtype=int)
    r = mcnemar_exact(a, b)
    assert r["b10"] == 21 and r["b01"] == 0
    assert r["p_exact"] == pytest.approx(stats.binomtest(0, 21, 0.5).pvalue, rel=1e-9)
    assert r["rd"] == pytest.approx(0.21) and r["rd_lo"] > 0


def test_newcombe_interval_contains_the_estimate_and_is_symmetric_without_discordance():
    d, lo, hi = newcombe_paired(40, 0, 0, 60)
    assert d == 0 and lo < 0 < hi
    d, lo, hi = newcombe_paired(30, 10, 5, 55)
    assert lo < d < hi


def test_holm_is_monotone_and_bounded():
    adj = holm([0.001, 0.04, 0.03, 0.5])
    assert adj[0] == pytest.approx(0.004) and max(adj) <= 1.0
    order = np.argsort([0.001, 0.04, 0.03, 0.5])
    assert all(adj[order[i]] <= adj[order[i + 1]] + 1e-12 for i in range(3))
