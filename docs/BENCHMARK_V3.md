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

## 3. A local LLM arm

The LLM path is now runnable without a paid API: `agents/llm.py` talks to a local Ollama server.
The `llm_ollama` arm uses the model as the **only** repair engine (no oracle queries, no search, no
heuristic fallback; a failed or unparsable reply leaves the circuit unchanged) with the same three
attempts and the same stage-named feedback. Tokens and calls are recorded per circuit.
`QREPAIR_OLLAMA_MODEL=qwen2.5-coder:1.5b python src/run_experiments.py --systems llm_ollama --limit 20`.

A 1.5-billion-parameter model on a laptop CPU is a floor, not a frontier model; see section 5 for
the measured outcome and what it does and does not show. In this arm the model is not forced to
diversify: resubmitting the same circuit is its own choice and is simply rejected again.

## 4. Rival tools

QBugLM, HornBro and UnitAR could **not** be run. A search on 2026-10-22 found no public code
repository for any of them (QBugLM's paper, arXiv 2606.07314, links none; HornBro and UnitAR are
journal papers without a repository link in the sources found). Re-implementing them from their
papers would not be a head-to-head with the tools, so none is reported. If their artifacts become
available the runner can host them as additional arms under the same predicate and budgets.

## 5. LLM result (60 circuits, `qwen2.5-coder:1.5b`, three attempts)

Stratified subset: three circuits per fault class and circuit family, compared on exactly the same
circuits (`python src/analyze_llm.py`; run takes about one hour on a laptop CPU).

| System | Pass@1 | Pass@3 | 95 % CI (Pass@3) |
|---|---|---|---|
| local LLM only (1.5B) | 11.7 % | 16.7 % | [9.3, 28.0] |
| static repairs, no oracle | 60.0 % | 60.0 % | [47.4, 71.4] |
| Qiskit compiler (O1) | 30.0 % | 30.0 % | [19.9, 42.5] |
| PyZX | 5.0 % | 5.0 % | [1.7, 13.7] |
| no hardware knowledge | 78.3 % | 78.3 % | [66.4, 86.9] |
| guided search + oracle | 98.3 % | 98.3 % | [91.1, 99.7] |

* Guided search vs the LLM: +0.817 [0.686, 0.891], 49 circuits repaired by guided search only and
  none by the LLM only (Holm p ~ 7e-15). Static repairs vs the LLM: +0.433 [0.298, 0.544], 26 vs 0.
* By class the LLM repaired 4 of 12 syntax and 6 of 12 hardware-fault circuits and none of the
  gate-redundancy, semantic and structural circuits. 26.7 % of the circuits never got a parsable reply.
  The 50 unrepaired circuits ended at the semantic (26), minimality (10), syntax (8) or connectivity (6) stage.
* Cost per circuit: 2.7 calls, 2,261 tokens, 61 s; 1.0 hour in total.

**What this does and does not show.** It shows that the harness can host a language model under the same
predicate, attempts and feedback, and that a model of this size does not substitute for oracle-guided
search on this benchmark. It does **not** characterise language models in general: 1.5 billion
parameters on a CPU is far below the frontier models used in other work (whose authors report much higher
rates with one retry, on a different benchmark and protocol); n = 60 gives a wide interval; and prompts were
not tuned beyond the physical-constraint text already in the repository. A first LLM run with a tabu check
that rejected a repeated reply was discarded as unfair to the model and rerun without it.
