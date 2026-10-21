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
| `qrepair` CLI (lint, localize, verify, repair), test-file specifications, localisation benchmark | `src/product.py`, `src/cli.py`, `src/eval_localization.py` |
| Extended edit neighbourhood (two-qubit substitution, analytic rotation insertion) | `extended_edits` flag, `ours_full_ext` arm |
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

---

# Second literature pass (gap tables, search date 2026-10-21)

Same caveat as above: figures below are as reported by the Consensus search, not
re-derived, except where marked **checked here**.

**Checked here.** The McNemar power approximation (z = 0.05 sqrt(n) / sqrt(0.20) = 3.54 at
n = 1000, power ~0.94; n ~ 628 for 80 % power), the Hoeffding shot count
(n >= ln(2/delta) / (2 eps^2) = 6623 at eps = 0.02, delta = 0.01) and the Holm thresholds
(0.05/16 ... 0.05) are arithmetically correct. The OpenQASM 3 paper (Cross et al., ACM TQC vol. 3) exists and
the formal specification lives in a GitHub-backed document. Items the report itself marks "no source found" (provider quotas
and prices, venue page limits and acceptance rates, 2026 model prices) are left open.

## Decisions

| Topic | Decision | Basis |
|---|---|---|
| Fault mix for a v3 benchmark | >= 40 % non-single-gate faults and >= 30 % not unitary-observable (measurement wiring, qubit order, API/IR, control flow). The report's 25/20/20/20/15 split is a *defensible choice*, not field prevalence. | Real-bug studies (Luo 2022, Paltenghi 2021) show API/platform/cross-layer faults dominate; fault-prevalence percentages quoted by the report were not re-checked |
| Language coverage | State plainly that the predicate covers the purely unitary subset of OpenQASM 3; add a structural check and a path-bounded trace oracle for control flow, measure and reset | Cross et al. 2021; done in the paper's limitations |
| Pass@k and budgets | Report external repair attempts K, internal search edits and oracle queries separately; never compare a bare Pass@3 | Already reported per arm |
| Baselines | Published quantum repair numbers (UnitAR, HornBro, ChatGPT/Bugs4Q) are not comparable to this benchmark; only a matched re-run counts | Tiny or different tasks and oracles |
| Oracle protocol | Two tiers: repair-time adaptive counterexamples; publication-time held-out states with X/Z/mixed-basis checks and Holm correction. Current implementation: Haar states only | Report's oracle table; mixed-basis checks are not implemented yet |
| Equivalence at 12-30 qubits | QCEC first, shallow-circuit methods (PBEC) where applicable, restricted-witness search for non-equivalence, bounded statistical check last | Peham 2022, Chen 2023, Yu 2025; not implemented beyond the sampled check |
| Compiler baselines | Qiskit O0-O3 done (9.7 / 29.7 / 29.7 / 29.7 %); tket and PyZX+routing still to add; publish seed, layout, routing, basis and whether final layout is undone | Done for Qiskit; the report's claim about Benchpress using level 2 is unverified |
| Physical validity | Two layers: hard (native gates, directed coupling, width, layout bookkeeping) and soft (duration / T1,T2, edge and readout quality, calibration age) | Report's rubric; thresholds are defensible choices |
| Duration threshold | The report suggests duration <= 0.2 min(T1,T2). **Checked here:** that would reject 4.1 % of this benchmark's own references (median ratio 0.033, p95 0.175, max 0.428), so adopting it requires regenerating the benchmark. Current rule is 0.5. | Own audit |
| Hardware study | 60 circuits x 3 devices x 2 calibration snapshots x 4096 shots (737,280 shots), readout mitigation only; 120 circuits if a 15-point paired effect must be detected at 80 % power | Report's design; needs an IBM Quantum account; costs and quotas not established |
| LLM study | Four arms: generation only, + oracle, + oracle + <= 2 retries, held-out score; one local open model first (CPU feasible per the cited llama.cpp study); no paid API until the non-LLM baseline is frozen | Contamination and retry confounds |
| Statistics | Exact McNemar primary; stratified-by-family McNemar or mixed-effects logistic regression as sensitivity; power n >= ~650 for a 5-point effect at ~20 % discordance | Fay and Lumbard 2020, Wu 2018 |
| Venues | TOSEM for protocol + benchmark; ASE/ISSTA for oracle or evaluation methodology; QCE/QSW/Q-SE for an early artifact. Fit scores are the report's judgement; check each CFP | No supplied source for 2026 CFPs |

## What changes in the work plan

1. v3 benchmark with non-single-gate and non-unitary faults (extends item 1 above).
2. Mixed-basis oracle tier and a held-out split reported with Holm correction.
3. tket and PyZX+routing baselines; structural/trace checks for control flow.
4. Stratified-by-family sensitivity analysis and a pre-registered primary test.
5. Local-LLM four-arm ablation, then the 60-circuit, 3-device hardware study.
