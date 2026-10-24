# Physics-Constrained Quantum Circuit Repair

[![tests](https://github.com/alok-108/physics-constrained-quantum-repair/actions/workflows/tests.yml/badge.svg)](https://github.com/alok-108/physics-constrained-quantum-repair/actions/workflows/tests.yml)

A leakage-free evaluation protocol, a corrected 1000-circuit benchmark, and a
repair agent for faulty OpenQASM 3 circuits under superconducting-device
constraints (IBM FakeSherbrooke model).

> **Status.** The first release of this repository reported 77.5 % Pass@1 and
> 99.1 % Pass@3. Those numbers are **not valid** (reference leakage, hard-coded
> baseline outcomes, an inadmissible benchmark; see
> [docs/BENCHMARK_V2.md](docs/BENCHMARK_V2.md)). They are kept under
> [`legacy_v1/`](legacy_v1/) for audit only. Everything below is v2.

**Interactive guide (plain English, 3D qubit, live repair demo, charts):** open [`docs/site/index.html`](docs/site/index.html) in a browser (self-contained; loads Three.js and fonts from CDNs).

## Use it: `qrepair`

```bash
python -m src.cli lint      examples/needs_lint.qasm
python -m src.cli verify    examples/ghz3_buggy.qasm --tests examples/ghz3_tests.json
python -m src.cli localize  examples/ghz3_buggy.qasm --tests examples/ghz3_tests.json
python -m src.cli repair    examples/ghz3_buggy.qasm --tests examples/ghz3_tests.json --out fixed.qasm
python -m src.cli repair    examples/ghz3_buggy.qasm --reference examples/ghz3_reference.qasm --json
```

| Command | What it does | Needs |
|---|---|---|
| `lint` | parse errors with a probable line, non-adjacent two-qubit gates, coherence budget, cancellable pairs, no-op rotations, idle qubits | nothing |
| `localize` | ranks instructions by the agreement gain of the best single edit there | reference circuit or test file |
| `verify` | runs the acceptance predicate and names the failing stage | reference circuit or test file |
| `repair` | minimal repair, unified diff, JSON report, device-aware routing and delay trimming | reference circuit or test file |

A specification is a reference circuit (exact unitary equivalence plus the noise criterion)
or a JSON file of `(input, expected output)` amplitude pairs. Test-file acceptance is
weaker and the report says so: passing the tests is not an equivalence certificate, so use
tests that include phase-sensitive inputs. Exit status: 0 success, 1 findings or failure,
2 input error. How this relates to MQT Debugger, Proq, HornBro, UnitAR and others, and
what is *not* claimed: [docs/COMPARISON.md](docs/COMPARISON.md).

Measured on the benchmark: the extended repair mode (used by the CLI) reaches 98.8 % Pass@3
against 97.0 % for the base agent, and the localiser ranks the faulty instruction first in
86.2 % of 400 structural/semantic faults (random ranking: 7.7 %).

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
| P+D with extended edit neighbourhood (two-qubit substitution, rotation insertion; added after E1) | 98.8 % [97.9, 99.3] | 100 % | 547 |
| P+D extended + oracle-guarded inverse cancellation (**post-hoc**: designed after reading the 12 failures of the row above on this same benchmark, so a development-set number) | 99.7 % [99.1, 99.9] | 100 % | 489 |
| Qiskit transpiler O0 / O2 / O3 | 9.7 % / 29.7 % / 29.7 % | 69.7 % | 0 |
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
* **The 1000-circuit table above uses no language model**; every number in it comes from the
  deterministic path. Language models were run separately, on a 60-circuit subset (below).

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
  llm.py             Ollama and Gemini clients (token accounting, rate-limit/quota handling)
  llm_tools.py       a model as the controller of verified repair tools, and the no-model greedy ablation
src/
  generate_dataset.py  benchmark v2 generator (admissible references, observable faults)
  audit.py             admissibility / observability audit for any dataset
  physics_model.py     FakeSherbrooke connectivity, coherence, depth
  routing.py           swap-in/swap-back routing, inverse cancellation
  sim.py, edits.py     batched simulator, single-instruction edits
  run_experiments.py   parallel, resumable benchmark runner (12 arms: 2x2 factorial, no-oracle, PyZX, Qiskit levels 0-3, Qiskit-router, extended edits)
  product.py, cli.py   the qrepair tool: lint, localize, verify, repair, hardware
  dynamic.py           measurement / reset / if / for programs: exact semantics, oracle, repair
  generate_v3.py, run_v3.py, analyze_v3.py   benchmark v3 (600 dynamic programs)
  analyze_llm.py, analyze_tools.py, analyze_guard.py   LLM-only arms, tool-controller arm, guarded cancellation
  eval_real.py, run_hardware.py   real-bug pilot (data_real/), hardware run (simulator mode tested)
  hornbro_adapter.py, score_hornbro.py   HornBro comparison (two environments; not yet run)
  generate_train.py    fresh-seed training set (train_set/, git-ignored)
  hardware.py, eval_hardware.py   calibration-based chip estimate
  eval_localization.py fault-localisation accuracy on the benchmark
  analyze_results.py   exact McNemar, Newcombe intervals, Holm; tables and figures
data/                 1000 circuits: buggy/, ground_truth/, metadata.csv
experiments/          results_main.csv, results_weak_oracle.csv, results_budget_*.csv
paper/                generated tables and figures
research_paper/       manuscript.tex, references.bib
docs/                 PROTOCOL.md, BENCHMARK_V2.md, BENCHMARK_V3.md, COMPARISON.md, ROADMAP.md, site/ (interactive guide)
data_v3/, data_real/  600 dynamic programs; 5 real Bugs4Q pairs
notebooks/            Colab notebooks: open-model LLM runs, HornBro comparison
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

## Beyond unitary circuits, a chip estimate, and an LLM arm (benchmark v3)

* **Dynamic programs.** `src/dynamic.py` gives an exact semantics for measurement, reset, `if`/`else`
  on one bit and constant-body `for` loops (cross-checked against Qiskit Aer). Benchmark v3
  (`python src/generate_v3.py`, 600 programs; wiring, control-flow, reset, loop, gate and
  multi-fault classes) is scored by exact output distributions. Guided repair: **92.3 %**
  Pass@3 [89.9, 94.2] vs 48.5 % for random search with the same budget; multi-fault programs
  67 %. 52.7 % of these faults are invisible to a unitary-equivalence check. The CLI handles such
  programs with `--reference` (`lint`, `verify`, `repair`).
* **Chip estimate.** `python -m src.cli hardware circuit.qasm` compiles to FakeSherbrooke's native
  gates and reports an estimated success probability from the calibration snapshot. On 200
  references the benchmark's identity layout has median ESP 0.66 against 0.89 with a compiler
  layout (QFT 0.34 vs 0.76): "physically valid" in the predicate is looser than "runs well".
  This is a model, not a hardware run.
* **LLM arms.** `agents/llm.py` talks to a local Ollama server or the Gemini API; an LLM arm uses the
  model as the only repair engine (no oracle, no search), with the same three attempts and stage-named
  feedback. On the same 60 circuits, Pass@3: `qwen2.5-coder:1.5b` **16.7 %** [9.3, 28.0],
  `gemini-3.1-flash-lite` **43.3 %** [31.6, 55.9], `gemini-3.5-flash-lite` **51.7 %** [39.3, 63.8], against
  98.3 % for guided search and 60.0 % for static repairs. Both Gemini models are "lite" models, so
  this is not the ceiling of language models. They repair semantic-deviation circuits only 8-17 % of the time
  (search: 100 %). Larger models stopped at the free-tier daily quota; partial runs are excluded from every
  table. See [docs/BENCHMARK_V3.md](docs/BENCHMARK_V3.md).
* **A model as the controller of verified tools** (`agents/llm_tools.py`). The model calls tools (localize,
  suggest oracle-scored edits, apply, submit) instead of writing the circuit. Same 60 circuits: the same tools with
  **no model** reach **98.3 %**; driven by `gemini-3.1-flash-lite` 95.0 % (writing the circuit: 43.3 %), by
  `gemini-3.5-flash-lite` 90.0 % (51.7 %). The model lifts itself a lot but never beats the no-model baseline, so
  correctness here comes from the tools and the verifier, not from the model's knowledge.
* **Training set.** `python src/generate_train.py` writes 3000 fresh-seed circuits, keeps 1900 after dropping
  every pair that also occurs in the benchmark (pair-level overlap 0, enforced); 21 % of its reference circuits
  coincide with benchmark references because the families are deterministic. For future fine-tuning only.
* **Real bugs (pilot, 5 pairs).** `data_real/` holds five circuit-logic bugs taken from Bugs4Q. Guided
  repair fixes 3 of 5 (random search 2 of 5); the two it misses are a QFT with a missing swap and four
  missing Hadamards. Language models fix 0 of 5 from the faulty circuit alone and 1 of 5 (`real_02`, an
  H-for-X swap) when also told the program's output on the all-zero input. Five pairs support no rate.
  Most Bugs4Q entries are library bugs, not circuit bugs ([data_real/README.md](data_real/README.md)).
* **Rival tools.** HornBro's code is public (<https://github.com/JanusQ/HornBro>, MIT). It needs Linux and
  Python 3.10, so the comparison runs in Colab: `notebooks/colab_hornbro.ipynb`, with an adapter
  (`src/hornbro_adapter.py`) and a scorer that applies this repository's predicate (`src/score_hornbro.py`).
  The adapter is written against HornBro's source but has not been run yet; no HornBro result is reported.
  QBugLM and UnitAR: no public code was found.
* **Not possible here:** running on a real quantum computer (needs an IBM Quantum account;
  `src/run_hardware.py` is ready and was exercised only on the noisy simulator).

Details, limits and commands: [docs/BENCHMARK_V3.md](docs/BENCHMARK_V3.md).

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

Small injected edits (v3 widens the fault types but they are still injected); connectivity under the
identity layout (the calibration estimate shows this is much looser than "runs well"); a device
model rather than live hardware; no externally executed QBugLM run, no
language-model results; the acceptance predicate covers only the purely unitary subset of
OpenQASM 3 and the benchmark does not represent real-fault prevalence (see `docs/ROADMAP.md`). See
[docs/BENCHMARK_V2.md](docs/BENCHMARK_V2.md#d-what-v2-does-not-fix) and the
paper's threats-to-validity section.

## Configuration

`.env.example` documents the optional `OPENAI_API_KEY` for the LLM repair path.
No reported result uses it.

## License and citation

Code, data and documentation are released under the [MIT License](LICENSE). `data_real/` contains circuits built from the
buggy and fixed scripts of [Bugs4Q](https://github.com/Z-928/Bugs4Q) (Apache-2.0); see `data_real/README.md`. If you use this
repository, cite it with [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button). The first release in
`legacy_v1/` reports numbers that are not valid and must not be cited.
