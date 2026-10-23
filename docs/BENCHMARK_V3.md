# Benchmark v3: dynamic programs, a calibration-based chip estimate, and an LLM arm

v2 answered "can the agent repair small unitary circuits under a strict protocol". Its published
limits were: faults are small gate edits; only the unitary part of OpenQASM 3 is checked; no LLM or
rival tool; the chip is the simplest model. This note records what was done about each, and what
could not be done.

## 1. Programs with measurement, reset, classical control and loops

`src/dynamic.py` adds an exact semantics for the subset

```
gates (standard library), measure q -> c, reset q,
if (c[i] == true|false) { ... } [else { ... }],   for int k in [0:n-1] { ... }   (constant body)
```

* The classical output distribution of a program is computed **exactly** by branching on
  measurement outcomes (no sampling). It was cross-checked against Qiskit Aer on teleportation,
  a reset-based parity check and a loop program: maximum TVD 0.0028 to 0.0037 at 200,000 shots,
  which is the shot noise of the Aer side.
* Acceptance (`dynamic.validate`): parses, same register sizes, every two-qubit gate adjacent
  (also inside `if`/`for` blocks), and the output distribution equals the reference's (TVD <= 1e-6)
  on 16 fixed held-out inputs.
* The search signal is a 4-input suite with a counterexample pool and the same query accounting as
  the unitary oracle; acceptance uses different inputs, so overfitting is measurable.
* The edit grammar adds what these programs need: re-pointing a measurement at another bit or
  qubit, **coordinated swaps of two measurements** (a swap cannot be undone one edit at a time),
  flipping or re-targeting a classical condition, deleting or inserting a reset, loop-count changes,
  inserting a classical correction before a qubit's last measurement, and 1-D angle search
  (36-point scan plus golden-section refinement, because a loop repeats the rotation).

### The benchmark (`python src/generate_v3.py`, 600 programs, deterministic)

Families: GHZ with terminal measurement (3-5 qubits), teleportation with two feed-forward
corrections, a parity check that reuses one ancilla with a reset, and a constant-body loop. Six fault
classes of 100 programs each: measurement **wiring**, classical **control flow**, **reset**, **loop**,
single **gate** faults, and **multi-fault** programs with 2 or 3 simultaneous faults (mean 2.5).
Every fault is verified observable (TVD > 1e-3 on at least one of 16 inputs).

| System (3 attempts, 3000-query budget) | Pass@3 | 95 % CI |
|---|---|---|
| guided search + counterexample feedback | 92.3 % | [89.9, 94.2] |
| guided search, bare rejection | 92.2 % | [89.7, 94.1] |
| random-mutation search, same grammar and budget | 48.5 % | [44.5, 52.5] |
| faulty program submitted unchanged | 0.0 % | [0.0, 0.6] |

By class (guided + feedback): wiring 100, control flow 100, reset 100, gate 96, loop 91,
**multi-fault 67**. Guidance over random search: +0.438 [0.396, 0.478] (268 vs 5 discordant
programs, Holm p < 1e-70). Counterexample feedback: no measurable effect (1 vs 0).

**Unitary-blind faults.** 52.7 % of the faults (all wiring, control-flow and reset faults and 16 %
of the multi-fault programs) leave the unitary of the gate sequence unchanged, so a unitary
equivalence check cannot see them. This is true by definition (measurement, reset and conditionals
are not part of a unitary), and the point is that the v2 predicate could not have scored them. The
guided search repairs 98.4 % of these 316 programs.

### What v3 does not do

* No noise stage and no minimality (inverse-pair) check for dynamic programs.
* Loops have constant bodies; no `while`, no multi-bit conditions, no expressions in conditions.
* `localize` and test-file specifications are not available for dynamic programs (a reference
  circuit is required).
* The faults are again edits whose inverse lies in the grammar. v3 widens the grammar and adds
  multi-fault programs (where the greedy search loses a third), but it is still a synthetic,
  injected-fault benchmark. Real faults (API misuse, SDK versions, transpilation) are not modelled.

## 2. A calibration-based chip estimate

`src/hardware.py` compiles a circuit to FakeSherbrooke's native gates and reports an estimated
success probability (ESP, from the snapshot's per-gate and readout errors), estimated duration and
duration / min(T1, T2), for an identity layout or a compiler-chosen one.
`python -m src.cli hardware circuit.qasm` prints it; `src/eval_hardware.py` runs 200 references.

Medians over 200 benchmark references (50 per family):

| family | ESP, identity layout | ESP, compiler layout | native 2q gates (id. / auto) |
|---|---|---|---|
| GHZ | 0.895 | 0.956 | 3.5 / 3 |
| QAOA | 0.223 | 0.794 | 36 / 26 |
| VQE | 0.880 | 0.938 | 6 / 6 |
| QFT | 0.342 | 0.760 | 110 / 31 |

The compiler layout gave the higher ESP on all 200 circuits. Consequences to keep in mind:

* "Physically valid" in the acceptance predicate (adjacency under the identity layout plus a coarse
  coherence check) is much looser than "would run well on the chip": under the identity layout the
  median QFT reference has ESP 0.34 and a duration/T ratio near 0.47.
* The agent's swap-in/swap-back routing is simple and wasteful on hardware. It preserves the unitary
  exactly (which the benchmark needs) but costs many native gates.
* This is a model of one calibration snapshot of a simulated device. No circuit was run on a real
  quantum computer: that needs an IBM Quantum account, calibration snapshots and queue time, none
  of which exist in this repository.

## 3. LLM arms (local Ollama and the Gemini API)

`agents/llm.py` provides two clients with the same interface: `OllamaChat` (a local Ollama server) and
`GeminiChat` (Gemini REST API; the key is read from `GEMINI_API_KEY`, sent in a header and scrubbed from errors;
free-tier rate limits are handled by spacing and retries, and an exhausted daily quota stops the run instead of
being recorded as wrong answers). An LLM arm uses the model as the **only** repair engine (no oracle queries, no
search, no heuristic fallback; a failed or unparsable reply leaves the circuit unchanged) with the same three
attempts and the same stage-named feedback. Tokens, calls and API errors are recorded per circuit; a circuit
with an API error is not recorded and is redone on `--resume`, so outages never count as model failures.

    QREPAIR_OLLAMA_MODEL=qwen2.5-coder:1.5b python src/run_experiments.py --systems llm_ollama --limit 60
    GEMINI_API_KEY=... QREPAIR_GEMINI_MODEL=<model id> python src/run_experiments.py --systems llm_gemini --limit 60         --out experiments/llm_runs/<model id>.csv --resume

In this arm the model is not forced to diversify: resubmitting the same circuit is its own choice and is
simply rejected again. `notebooks/colab_llm_runs.ipynb` runs open models on a free Colab GPU.

## 4. Rival tools

* **HornBro** (FSE 2025) has a public repository (<https://github.com/JanusQ/HornBro>, MIT) and a Zenodo
  artifact. It needs Linux and Python 3.10, and its dependencies conflict with this repository's Qiskit, so the
  comparison is done in two steps that exchange OpenQASM files: `src/hornbro_adapter.py` runs HornBro's own
  repair loop (given the reference circuit, as it is designed) on the same 60 circuits, and `src/score_hornbro.py`
  judges what it returns with this repository's predicate. The notebook is `notebooks/colab_hornbro.ipynb`.
  **No HornBro result is reported yet**: the adapter has not been run (the Windows development machine cannot
  host it). HornBro repairs gate parameters and patches of unitary circuits, so faults that leave the program
  unparsable count as failures with the reason recorded.
* **QBugLM, UnitAR:** no public code repository was found (search 2026-10-22). A Bugs4Q-based mutation study
  (arXiv 2601.12273) also links none.

Re-implementing a tool from its paper would not be a head-to-head, so none is reported.

## 5. LLM results (same 60 circuits, three attempts)

Stratified subset: three circuits per fault class and circuit family, compared on exactly the same
circuits (`python src/analyze_llm.py`). A run that stopped early (daily quota, server errors) covers only the
lowest circuit ids, which are the easiest classes, so partial runs are excluded from every table.

| System | Pass@1 | Pass@3 | 95 % CI (Pass@3) |
|---|---|---|---|
| `qwen2.5-coder:1.5b` (local CPU) | 11.7 % | 16.7 % | [9.3, 28.0] |
| `gemini-3.1-flash-lite` | 40.0 % | 43.3 % | [31.6, 55.9] |
| `gemini-3.5-flash-lite` | 51.7 % | 51.7 % | [39.3, 63.8] |
| static repairs, no oracle | 60.0 % | 60.0 % | [47.4, 71.4] |
| Qiskit compiler (O1) | 30.0 % | 30.0 % | [19.9, 42.5] |
| PyZX | 5.0 % | 5.0 % | [1.7, 13.7] |
| no hardware knowledge | 78.3 % | 78.3 % | [66.4, 86.9] |
| guided search + oracle | 98.3 % | 98.3 % | [91.1, 99.7] |

Pass@3 by fault class (%), 12 circuits per class:

| class | qwen 1.5B | gemini 3.1 lite | gemini 3.5 lite | static | guided |
|---|---|---|---|---|---|
| syntax | 33.3 | 66.7 | 91.7 | 100 | 100 |
| gate redundancy | 0.0 | 75.0 | 83.3 | 100 | 100 |
| hardware (physics) | 50.0 | 16.7 | 33.3 | 100 | 100 |
| structural | 0.0 | 41.7 | 41.7 | 0 | 91.7 |
| semantic deviation | 0.0 | 16.7 | 8.3 | 0 | 100 |

* Guided search vs `gemini-3.5-flash-lite`: +0.467 [0.334, 0.589], 28 circuits repaired by guided search only and
  none by the model only (Holm p ~ 5e-8); vs `gemini-3.1-flash-lite`: +0.550 [0.413, 0.667], 33 vs 0. Against
  `qwen2.5-coder:1.5b`: +0.817 [0.686, 0.891], 49 vs 0.
* Static repairs vs the Gemini models: +0.083 [-0.048, 0.210] (11 vs 6) for 3.5-lite and +0.167 [0.011, 0.310]
  (17 vs 7) for 3.1-lite; neither survives the Holm correction (p = 0.60, 0.19). Cheap deterministic repairs are
  not clearly worse than these models, and clearly better only on hardware faults.
* `gemini-3.5-flash-lite` vs `gemini-3.1-flash-lite`: +0.083 [-0.040, 0.202], not significant. Both Gemini models
  vs qwen: +0.267 and +0.350 (Holm p = 0.021, 1e-4).
* The Gemini models never failed to produce a parsable circuit (qwen: 26.7 % of circuits). Their failures end at
  the semantic stage (16 and 17 circuits) and the connectivity stage (16 and 11): they rarely fix angle errors and
  often miss the hardware adjacency rule.
* Cost per circuit: qwen 2.7 calls, 2,261 tokens, 61 s (CPU); gemini 3.1 lite 2.2 calls, 2,207 tokens, 78 s;
  gemini 3.5 lite 2.0 calls, 1,650 tokens, 12 s.
* In progress: `gemini-3.8-flash` (18/60 done) and `gemini-3.5-flash` (14/60) stopped at the free-tier daily
  quota; `gemma-4-31b-it` and `gemma-4-26b-a4b-it` hit repeated HTTP 500 errors from the service. None of these
  is reported. Open models on a Colab GPU and one 1000-circuit run are planned.

## 6. Real bugs (pilot, 5 pairs)

`data_real/` holds five circuit-logic bugs from Bugs4Q (see `data_real/README.md` for how they were selected and
why most Bugs4Q entries could not be used). Accepted within three attempts:

| arm | real_01 | real_02 | real_03 | real_04 | real_05 | total |
|---|---|---|---|---|---|---|
| guided search (oracle on the reference) | yes | yes | yes | no | no | 3 / 5 |
| random search, same grammar and budget | yes | yes | no | no | no | 2 / 5 |
| LLM, faulty circuit only (qwen, gemini 3.1 lite, gemini 3.5 lite) | no | no | no | no | no | 0 / 5 each |
| LLM, plus the expected output on the all-zero input (gemini, both) | no | yes | no | no | no | 1 / 5 each |
| LLM, same, qwen 1.5B | no | no | no | no | no | 0 / 5 |

The arms do not receive the same information: guided search queries the reference's behaviour, the LLMs see at
most one output distribution. Five pairs support no rate and no ranking; they show that the pipeline runs on
bugs that people actually wrote and that it fails on two of them (a QFT with a missing swap and reordered
gates; four missing Hadamards).

**What this does and does not show.** The harness hosts language models under the same predicate, attempts and
feedback, and none of the three models substitutes for oracle-guided search on this benchmark. A larger model
did better (qwen 1.5B 16.7 %, two Gemini "lite" models 43-52 %), but the gap to guided search stays large
(+0.47 to +0.55) and the Gemini models solve few semantic-deviation circuits. It does **not** characterise
language models in general: no frontier-size model has been run (the free API tier stopped the larger Gemini
runs), n = 60 gives wide intervals (the two Gemini models do not differ significantly), the circuits are
generated faults, and prompts were not tuned beyond the physical-constraint text already in the repository. A
first LLM run with a tabu check that rejected a repeated reply was discarded as unfair to the model and rerun
without it.
