# Physics-Constrained Quantum Circuit Repair

A leakage-free evaluation protocol, a corrected 1000-circuit benchmark, and a
repair agent for faulty OpenQASM 3 circuits under superconducting-device
constraints (IBM FakeSherbrooke model).

> **Status.** The first release of this repository reported 77.5 % Pass@1 and
> 99.1 % Pass@3. Those numbers are **not valid** (reference leakage, hard-coded
> baseline outcomes, an inadmissible benchmark; see
> [docs/BENCHMARK_V2.md](docs/BENCHMARK_V2.md)). They are kept under
> [`legacy_v1/`](legacy_v1/) for audit only. Everything below is v2.

## Results (v2, 1000 circuits, attempt budget K = 3, deterministic agent)

| System | Pass@3 (95 % CI) | Physically valid | Mean oracle queries |
|---|---|---|---|
| P+D: physics context, diagnostic feedback | **97.0 %** [95.7, 97.9] | 100 % | 553 |
| P+R: physics context, bare rejection | 97.0 % [95.7, 97.9] | 100 % | 555 |
| U+D: no physics context, diagnostic feedback | 77.0 % [74.3, 79.5] | 80.1 % | 963 |
| U+R: no physics context, bare rejection | 77.0 % [74.3, 79.5] | 80.1 % | 965 |
| Static repair (no oracle) | 60.0 % [56.9, 63.0] | 100 % | 0 |
| PyZX (ZX-calculus simplification) | 5.0 % [3.8, 6.5] | 15.1 % | 0 |
| Qiskit transpiler only (SABRE + exact layout undo) | 29.7 % [26.9, 32.6] | 69.7 % | 0 |
| P+D with Qiskit's router instead of the agent's own | 97.0 % [95.7, 97.9] | 100 % | 553 |

What the numbers do and do not say:

* **Physics context** adds +20.0 points (200 discordant circuits, none reversed).
  It is confined to the 200 physical-fault circuits, where an arm without delay
  trimming and routing cannot succeed; on logical faults the arms are identical.
  This is a *capability* effect, not evidence that physical knowledge improves
  semantic repair.
* **The behavioural oracle** is what separates semantic repair from static
  repair (+37.0 points).
* **Diagnostic feedback** (counterexample-guided) has no effect with a strong
  oracle (suite of 4 states) but raises Pass@3 from 88.1 % to 97.0 %
  (+8.9 points, Holm p ≈ 3e-25) when the initial suite is one state.
* The benchmark consists of small edits whose inverse lies inside the agent's
  edit neighbourhood, so ~97 % is **not** evidence of general repair ability.
  Per-fault-mode results (e.g. 63 % on CX→CZ replacement, 0 % when a rotation
  was removed) show where it fails; see `paper/tables/mode_breakdown.csv`.
* **No language model was executed.** The LLM path exists but needs an API key;
  every number comes from the deterministic path.

Full tables: `paper/tables/`. Figures: `paper/figures/`. Paper source:
`research_paper/manuscript.tex`.

## What is in here

```
agents/
  diagnostician.py   fault localisation (parser, device checks, cancellation); labels stripped
  repairer.py        syntax / cancel / route, and oracle-guided edit search
                     (analytic angle solving, edit neighbourhood, tabu)
  oracle.py          query-counted behavioural oracle: suite, counterexample pool, held-out states
  validator.py       acceptance predicate (the only reader of the reference source)
  orchestrator.py    bounded loop, counterexample-guided feedback, tabu bookkeeping
src/
  generate_dataset.py  benchmark v2 generator (admissible references, observable faults)
  audit.py             admissibility / observability audit for any dataset
  physics_model.py     FakeSherbrooke connectivity, coherence, depth
  routing.py           swap-in/swap-back routing, inverse cancellation
  sim.py, edits.py     batched simulator, single-instruction edits
  run_experiments.py   parallel, resumable benchmark runner (6 arms)
  analyze_results.py   exact McNemar, Newcombe intervals, Holm; tables and figures
data/                 1000 circuits: buggy/, ground_truth/, metadata.csv
experiments/          results_main.csv, results_weak_oracle.csv, results_budget_*.csv
paper/                generated tables and figures
research_paper/       manuscript.tex, references.bib
docs/                 PROTOCOL.md, BENCHMARK_V2.md
legacy_v1/            the invalid first release (data, results, paper), for audit
tests/                pytest suite (information boundary, validator stages, search, statistics)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                                                # a few minutes
python src/audit.py                                      # 0 violations expected on data/
python src/audit.py --root legacy_v1                     # shows the v1 defects (367 inadmissible references)

python src/run_experiments.py --workers 8 --out experiments/results_main.csv
python src/analyze_results.py
```

All experiments are deterministic; the exact commands for E1 (main), E2 (weak
oracle) and E3 (budget sweep) are in [docs/PROTOCOL.md](docs/PROTOCOL.md).
Regenerating the dataset (`python src/generate_dataset.py`, ~1 min) reproduces
`data/` byte for byte.

To compile the paper: `cd research_paper && pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript`
(tables and figures are read from `../paper/`).

## Protocol in one paragraph

The repair side never sees the reference circuit or the injected fault label. It
queries a behavioural oracle (a few input states with known outputs) that
returns scalar agreement scores; every scored edit costs one query. Acceptance is
a deterministic conjunction of syntax, register width, device admissibility,
irreducibility, exact unitary agreement (process fidelity ≥ 0.99) and an exact
noisy-TVD criterion below the 4096-shot sampling floor. Search signal and
acceptance signal differ, so overfitting is measured (held-out states, exact
process fidelity). Details: [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Limitations

Small edits of short circuits; connectivity under the identity layout; device
model rather than live hardware; no externally executed QBugLM run, no
language-model results. See
[docs/BENCHMARK_V2.md](docs/BENCHMARK_V2.md#d-what-v2-does-not-fix) and the
paper's threats-to-validity section.

## Configuration

`.env.example` documents the optional `OPENAI_API_KEY` for the LLM repair path.
No reported result uses it.
