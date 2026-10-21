# Benchmark v2: what changed from v1 and why

The v1 results (77.5 % Pass@1, 99.1 % Pass@3, 99.2 % physics validity; kept in
`legacy_v1/` for audit only) are **not valid evidence**. Two groups of problems
were found; both are fixed in v2.

## A. Harness defects (leakage and bookkeeping)

| Defect in v1 | Where | v2 |
|---|---|---|
| Ground-truth circuit returned as the "repair" (`ground_truth_hint`) | `agents/repairer.py`, `agents/orchestrator.py` | argument removed; reference reaches only the validator and the query-counted oracle |
| Baseline failure hard-coded for all physics-violation circuits | `src/run_experiments.py` | every outcome measured |
| Baseline `pass_at_3` set equal to `pass_at_1` (1 attempt vs. 3) | `src/run_experiments.py` | every arm gets the same K, Pass@k read from the attempt trace |
| Arm named "QBugLM" was the project's own repairer and was not actually unconstrained: the hardware dict was a default argument (`physics_constraints or {...}`), and the routing branch ran whenever the diagnostician, which always ran the physics checks, reported a physics violation | `src/run_experiments.py`, `agents/repairer.py` | no arm carries an external system's name; physics context is an explicit switch |
| Injected fault label passed to the diagnostician inside `circuit_spec` | `src/run_experiments.py` | labels stripped (`LEAKY_KEYS`) |
| Unseeded 2048-shot TVD test: an identical candidate could fail then pass | `agents/validator.py` | exact, deterministic noise comparison |
| Integer gate-count difference reported as a percentage | `src/run_experiments.py` | overhead vs. reference in %, accepted repairs only |
| Wilcoxon signed-rank + Cohen's d on paired 0/1 outcomes | `src/analyze_results.py` | exact McNemar, Newcombe interval, Holm |
| Depth limit 250 in the validator, 200 in the metric | validator vs. runner | one constant |

## B. Benchmark and acceptance defects (found by `src/audit.py`, still present after fixing A)

Measured on the v1 data with the validator's own checks:

* **367 of 1000 reference circuits fail the validator's connectivity check**
  (all 250 QFT, 117 of 250 QAOA with a ring edge; GHZ and VQE: 0). A candidate
  could be semantically perfect and still be rejected, and any arm lacking a
  routing step could not pass those circuits at all. In v1 this made the effect
  of "physics awareness" depend on circuit family rather than on the fault.
* **The semantic check was blind on QFT.** On |0…0⟩ a QFT outputs the uniform
  superposition; a circuit with only the Hadamard layer, and one with every
  controlled phase negated, both have statevector overlap 1.0000 with the
  reference (checked on real v1 references). Controlled-phase and swap faults in
  QFT were therefore unobservable.
* **20 of 400 structural/semantic faults left the unitary unchanged**
  (process fidelity ≥ 0.99), i.e. were not faults.
* **Gate-redundancy faults preserve the unitary**, and the v1 predicate had no
  stage that distinguishes the redundant circuit from the reference; on
  admissible references an untouched circuit satisfies it.
* The repair neighbourhood mirrored the injector's modes while claiming not to
  be conditioned on the fault class.
* The search signal and the acceptance signal were the same single-state
  statevector overlap, so overfitting could not be measured.

## C. v2 design

| Property | Mechanism |
|---|---|
| References are hardware-admissible | `route_and_cancel`: swap-in/swap-back routing (exact unitary), then inverse cancellation. Audit: 0 / 1000 violations. |
| Every fault is observable | The generator verifies observability and re-samples otherwise (`is_observable`). Audit: 0 / 1000 unobservable. |
| Redundancy is a measurable fault | New stage `V_min` (no cancellable inverse pair). |
| Semantics are exact | Process fidelity on the full unitary (global-phase invariant). |
| Physics faults are physical | "Unmapped" = a device-agnostic circuit that needs routing (reference = its routed form); "coherence delay" = 1200 µs idle (reference = without it). Both are semantically correct before repair. |
| Search signal ≠ acceptance signal | Oracle suite of 1–4 states; acceptance on the full unitary; held-out states; reported overfitting gap. |
| Feedback is testable without an LLM | Counterexample-guided repair: a semantic rejection adds the worst-case pool state to the suite (diagnostic arms only). |
| Results can be reported per fault mode | `fault_mode`, `mode_family` columns (`legacy` = v1 injector modes, `extended` = added in v2: scale, shift π/3, compound faults, CX→CZ). |
| Effects are separable | 2 × 2 factorial design (physics × feedback) plus a no-oracle lower bound and PyZX. |

## D. What v2 does not fix

* Faults are small edits of a short circuit. The neighbourhood contains the
  inverse of most single-instruction edits, so a high rate on single-edit faults
  is expected and is **not** evidence of general repair ability. The informative
  cells are the extended modes, compound faults, and the weak-oracle and
  low-budget regimes.
* `RepairerAgent`'s default router is the swap-in/swap-back construction used to
  build physics-fault references. This was tested: replacing it by Qiskit's
  SABRE router with exact final-layout undo (`ours_full_qiskit_router`) leaves
  acceptance unchanged (97.0 %, zero discordant circuits), so agreement is not
  an artefact of a shared router. The physics-context contrast is still a
  capability contrast.
* Connectivity is checked under the identity layout; no live-device or
  calibration-drift evidence.
* No LLM configuration has been executed (no API key was available); the
  repository contains the LLM path but every reported number is from the
  deterministic path.
* An externally executed QBugLM run is still missing. (A compiler-only baseline,
  `qiskit_transpile`, is included: 29.7 % Pass@3.)
