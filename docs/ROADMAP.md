# Roadmap (literature-informed)

Source: a Consensus AI literature report (search date 2026-10-21) on quantum bug
datasets, repair, oracles, equivalence checking, hardware-aware compilation and
LLM evaluation. This file records what the report implies for the project, what is
already done, and what is next. It is a planning document, not a result.

**Verification status of the report.** All DOIs we plan to cite resolve at
doi.org (registered); the titles of five arXiv entries
(2606.27124, 2609.13839, 2607.09007, 2606.07314, 2409.08844) match the report.
Publisher-DOI titles were not independently checked, and the report's evidence
scores (e.g. "Strong 9/10") are Consensus' own judgements, not ours. Treat every
statement below as "reported by the literature search, not re-derived".

## What the literature says about our position

* The defensible claim is a **leakage-controlled, benchmarked repair framework for
  small superconducting-style circuits**, not "state-of-the-art quantum repair".
  The paper already claims this and says so in its limitations.
* Real-bug benchmarks are scarce and fragile (Bugs4Q: small, Qiskit-specific,
  version-sensitive). A synthetic benchmark cannot estimate real-fault prevalence.
* Real faults are often cross-layer (APIs, transpilation, measurement, hybrid
  boundaries) and are not single-instruction edits.
* Dense unitary equivalence does not scale; no single scalable paradigm dominates
  (QCEC decision diagrams vs ZX; AutoQ for restricted specifications).
* Identity-layout adjacency is necessary, not sufficient, for physical validity.
* Published quantum repair baselines exist (HornBro, UnitAR, QBugLM) but differ in
  task, language and oracle, so head-to-head claims need a matched re-run.

## Done in this repository

| Item | Where |
|---|---|
| Version pinning of the environment | `requirements.lock` |
| Sampled (Haar-state) semantic check for > 10 qubits, with noise stage skipped and flagged | `src/sim.py`, `agents/validator.py`, `tests/test_scalable_semantics.py` |
| Compiler-only baseline (Qiskit SABRE + exact layout undo) | `qiskit_transpile` arm |
| Router-robustness check (own router vs Qiskit's) | `ours_full_qiskit_router` arm |
| Positioning against Bugs4Q, HornBro, UnitAR, AutoQ, QCEC-ZX, noise-adaptive mapping | `research_paper/manuscript.tex` |

## Next (CPU only, no paid services), ordered by impact x feasibility

| # | Item | Why (literature) | Effort | Success criterion |
|---|---|---|---|---|
| 1 | Harder fault distribution: rotation/controlled-phase insertion with analytic angle, 3-4 simultaneous faults, bit-order/measurement-mapping faults | Real faults are not local edits; our 97 % is edit-inversion | M | Report Pass@3 per mode on >= 3 new modes; at least one mode where the agent fails |
| 2 | Plug in MQT-QCEC as the exact checker at 10-24 qubits; compare with the sampled check | Hybrid equivalence stack is the report's top open question | M | Agreement table sampled vs QCEC; documented false-accept rate for small-angle faults |
| 3 | Scaling benchmark: 12-20 qubit GHZ/QAOA/VQE/QFT (generator already parameterised by qubit count) | Dense checks stop near 10-12 qubits | M | Pass@3 and wall time vs qubits; noise stage replaced by trajectory sampling or explicitly dropped |
| 4 | Stronger oracle study: projective / mixed-basis assertions vs random states | Computational-basis-only checks miss phase faults | M | Sweep of suite type and size on Pass@3 and overfitting gap |
| 5 | More compiler baselines: tket (`pytket`), Qiskit levels 0-3, with exact layout undo | Fair compiler baselines are standard | S-M | Table of baselines under the same predicate |
| 6 | Statistics upgrade: cluster-robust intervals by circuit family, power analysis for 5-point differences | Clustering and power are reporting norms | S | Intervals recomputed; sample-size note in the paper |
| 7 | Bugs4Q adapter in a pinned, per-version environment | Only public real-bug set; fragile across SDK versions | L | A subset of reproducible Bugs4Q bugs evaluated under the same predicate |
| 8 | Physical validity beyond adjacency: native basis, directed edges, scheduling | Placement and routing change success rates a lot | M | Validity definition v2 with an ablation of each added check |

## Needs resources outside this machine

| Item | Needs |
|---|---|
| LLM arms (generation vs feedback ablation, token/cost accounting, mutation-guided prompts) | An API key or a local model (an Ollama install exists on this machine); fixed prompts, seeds, model versions, held-out oracle reporting |
| QBugLM / HornBro / UnitAR head-to-head | The authors' releases run under this protocol with matched budgets |
| Live-hardware validation | An IBM Quantum account, calibration snapshots, a minimum design (circuits x shots x devices) decided before running |

## Claims not to make

* "State-of-the-art quantum program repair."
* Anything about real-world fault prevalence or about LLM repair.
* That 97 % shows general repair ability (single-edit faults whose inverse is in the neighbourhood).
* That sampled-fidelity acceptance certifies equivalence (it is one-sided evidence).

## Kill criteria

* If harder fault modes (item 1) bring the agent to the level of the static baseline, the contribution is the protocol and benchmark only; reframe the paper accordingly.
* If sampled and exact checks disagree on more than a few percent of small-angle faults (item 2), do not use sampled acceptance for any reported number.
