"""Claims check: every headline number quoted in the prose must equal the number in the generated table.

If a table is regenerated and the text is not updated (or the other way round), these tests fail.
"""

import os
import re

import pandas as pd
import pytest


def read(root, *parts):
    return open(os.path.join(root, *parts), encoding="utf-8").read()


def norm(s):
    """Compare numbers regardless of '16.7 %', '16.7\\%' or '16.7%'."""
    return re.sub(r"\s*\\?%", "%", s).replace("$", "").replace("{,}", "")


@pytest.fixture(scope="module")
def tables(root):
    t = lambda name: pd.read_csv(os.path.join(root, "paper", "tables", name))
    return {n: t(n) for n in ("rates.csv", "llm_rates.csv", "tools_rates.csv", "guard.csv", "v3_rates.csv")}


@pytest.fixture(scope="module")
def docs(root):
    return {"README": norm(read(root, "README.md")), "V3": norm(read(root, "docs", "BENCHMARK_V3.md")),
            "paper": norm(read(root, "research_paper", "manuscript.tex")), "site": norm(read(root, "docs", "site", "index.html"))}


def has(text, value):
    return f"{value:.1f}%" in text


def test_main_table_rows_appear_in_the_readme(tables, docs):
    r = tables["rates.csv"].set_index("System")
    for name in ("Physics + diagnostic feedback", "No physics, diagnostic feedback", "Static repair (no oracle)", "PyZX",
                 "Qiskit transpiler only", "Physics + diagnostic, extended edits"):
        assert has(docs["README"], r.loc[name, "Pass@3 (%)"]), name
        assert r.loc[name, "Pass@3 95% CI"] in docs["README"], name


def test_llm_only_rates_match_everywhere(tables, docs):
    r = tables["llm_rates.csv"].set_index("system")
    for name in ("qwen2.5-coder:1.5b", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "guided search + oracle",
                 "static repairs, no oracle"):
        v = r.loc[name, "P@3 (%)"]
        for where in ("README", "V3"):
            assert has(docs[where], v), (name, where, v)
        assert r.loc[name, "P@3 95% CI"] in docs["README"] or name.startswith("static") or name.startswith("guided")
    for v in (r.loc["qwen2.5-coder:1.5b", "P@3 (%)"], r.loc["gemini-3.1-flash-lite", "P@3 (%)"],
              r.loc["gemini-3.5-flash-lite", "P@3 (%)"], r.loc["guided search + oracle", "P@3 (%)"]):
        assert has(docs["paper"], v), v


def test_tool_arm_rates_match(tables, docs):
    r = tables["tools_rates.csv"].set_index("arm")
    for name in ("tools: gemini-3.1-flash-lite", "tools: gemini-3.5-flash-lite", "tools: no model (greedy)"):
        v = r.loc[name, "P@3 (%)"]
        for where in ("README", "V3", "paper"):
            assert has(docs[where], v), (name, where, v)


def test_guard_numbers_match_and_are_labelled_post_hoc(tables, docs):
    g = tables["guard.csv"].iloc[0]
    for where in ("README", "V3", "paper"):
        assert has(docs[where], g["guard_P3"]) and has(docs[where], g["ext_P3"]), where
    assert "post-hoc" in docs["README"].lower() and "post-hoc" in docs["V3"].lower()
    assert "development-set" in docs["V3"] and "development-set" in docs["paper"]


def test_v3_dynamic_rates_match(tables, docs):
    r = tables["v3_rates.csv"].set_index("system")
    for name in ("guided + feedback", "random search"):
        for where in ("README", "paper"):
            assert has(docs[where], r.loc[name, "Pass@3 (%)"]), (name, where)


def test_site_quotes_the_current_llm_numbers(tables, docs):
    r = tables["llm_rates.csv"].set_index("system")
    t = tables["tools_rates.csv"].set_index("arm")
    assert has(docs["site"], r.loc["guided search + oracle", "P@3 (%)"])
    assert has(docs["site"], r.loc["qwen2.5-coder:1.5b", "P@3 (%)"])
    assert has(docs["site"], t.loc["tools: no model (greedy)", "P@3 (%)"])


def test_partial_runs_are_not_in_any_table(root):
    t = pd.read_csv(os.path.join(root, "paper", "tables", "llm_rates.csv"))
    assert not t.system.str.contains("3.8-flash|gemini-3.5-flash$", regex=True).any()
    assert (t.n == 60).all()


def test_no_secrets_in_tracked_text(root):
    pattern = re.compile(r"AQ\.Ab8RN|AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}")
    for rel in ("README.md", ".env.example", "docs/BENCHMARK_V3.md", "docs/site/index.html", "agents/llm.py",
                "src/run_experiments.py", "notebooks/colab_llm_runs.ipynb", "notebooks/colab_hornbro.ipynb"):
        assert not pattern.search(read(root, *rel.split("/"))), rel
