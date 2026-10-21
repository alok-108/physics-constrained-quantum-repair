# legacy_v1: first release (invalid, kept for audit)

This directory holds the first release's benchmark, results and paper artefacts.
**Do not cite the numbers in it.** They were produced by a harness in which the
reference circuit was returned as the repair, baseline outcomes were hard-coded
and attempt budgets were unequal, on a benchmark whose references failed the
validator's own connectivity check for 367 of 1000 circuits.

See [`../docs/BENCHMARK_V2.md`](../docs/BENCHMARK_V2.md) for the defect list and
the fixes. The audit tool reproduces the benchmark findings:

```bash
python src/audit.py --root legacy_v1
```

| Path | Content |
|---|---|
| `data/` | the original 1000 circuits (`buggy/`, `valid/`, `ground_truth/`, `metadata.csv`) |
| `results_v1.csv` | the original 3000 result rows (77.5 % / 99.1 % claims) |
| `paper/` | original figures and Wilcoxon/Cohen's d tables |
| `research_paper/` | original manuscript sources, PDF/DOCX, bibliographies, revision checklist |

The v1 agent code is not duplicated here; it is available in the git history
before the `evaluation-v2` branch.
