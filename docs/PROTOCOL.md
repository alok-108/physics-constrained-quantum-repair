# Evaluation protocol

This document specifies exactly what is measured and what each system is allowed
to know. It is the reference for the code in `agents/`, `src/` and for the paper
in `research_paper/`.

## 1. Information boundary

| Component            | Reference circuit source | Fault label / mode | Behavioural oracle | Attempt budget |
|----------------------|:------------------------:|:------------------:|:------------------:|:--------------:|
| `ValidatorAgent`     | yes                      | no                 | n/a                | n/a            |
| `Orchestrator`       | passes it on, never inspects it for repair | no | builds it | K (same for every arm) |
| `DiagnosticianAgent` | no                       | stripped           | no                 | n/a            |
| `RepairerAgent`      | no                       | stripped           | queries only       | n/a            |

The labels `bug_type`, `fault_mode`, `mode_family`, `bug_description`,
`ground_truth_file`, `buggy_file`, `ref_gates` are removed from the specification
by `agents.diagnostician.LEAKY_KEYS` before any repair-side component sees it.
`tests/test_agents.py` enforces the boundary (no reference argument in the
`repair` signature, no access to private oracle state in `agents/repairer.py`).

## 2. The behavioural oracle

`agents/oracle.py` wraps the reference in a query-counted scoring function. It
holds three disjoint state sets:

* **suite** – guides the search; |0…0⟩ plus (suite size − 1) Haar-random product
  states. Default size 4; the weak-oracle experiment uses 1.
* **pool** – candidate counterexamples (16 Haar-random states).
* **holdout** – 8 Haar-random states never used by search or feedback.

Cost accounting: every candidate edit scored, every `score` call and every
`counterexample` call consumes one query from a per-run budget (default 8000).
`evaluate_edits` scores a whole single-instruction neighbourhood by
meet-in-the-middle from cached prefix/suffix states; its values equal naive
scoring to 1e-10 (tested) and it charges one query per edit. Only scalars leave
the oracle.

`holdout_gap` compares suite agreement with exact process fidelity and with
held-out state agreement for the final candidate; it is an evaluation
instrument and is not charged.

## 3. Acceptance predicate

Identical for every system and never ablated:

```
A(C; R, B) = V_syn ∧ V_str ∧ V_phys ∧ V_min ∧ V_sem ∧ V_noise
```

| Stage | Definition |
|-------|------------|
| `V_syn` | parses as OpenQASM 3 |
| `V_str` | register width equals the specification's |
| `V_phys` | every two-qubit gate on adjacent qubits of FakeSherbrooke (identity layout), per-qubit execution time under half of min(T1,T2), depth ≤ 250 |
| `V_min` | no adjacent self-inverse pair remains (DAG inverse cancellation fixed point) |
| `V_sem` | process fidelity ∣tr(U_C†U_R)∣²/d² ≥ 0.99 (exact unitary, global-phase invariant) |
| `V_noise` | exact noisy-output TVD between candidate and reference ≤ the 99th percentile of the TVD between two independent 4096-shot samples of the reference |

`V_noise` uses density-matrix simulation under a depolarising model (1q 2.5e-4,
2q 7.5e-3) with symmetric readout error 1.5 % applied analytically, so the
verdict is deterministic and a candidate identical to the reference is never
falsely rejected.

**Layout assumption.** Connectivity is checked on virtual qubit indices with no
layout or transpilation pass. "Physically valid" therefore means admissible under
the identity layout.

## 4. Systems

2 × 2 factorial design plus reference points (no-oracle, PyZX, Qiskit compiler baselines, router-robustness arm):

| Arm | Physics context | Feedback | Oracle |
|-----|:---:|:---:|:---:|
| `ours_full` | on | diagnostic (stage + counterexample) | yes |
| `ours_reject_only` | on | bare rejection | yes |
| `unconstrained_diag` | off | diagnostic | yes |
| `unconstrained_reject_only` | off | bare rejection | yes |
| `static_no_oracle` | on | – | no (syntax, cancellation, routing only) |
| `pyzx_zx_calculus` | – | – | no (equivalence-preserving transformation baseline) |
| `qiskit_transpile` | – | – | no (Qiskit routing + O1 cancellation, exact final-layout undo; compiler baseline) |
| `ours_full_qiskit_router` | on | diagnostic | yes (as `ours_full`, but physical faults routed by Qiskit SABRE) |

"Physics context" gates (i) device checks in diagnosis and (ii) delay trimming
plus swap routing in generation. It does not change what is required for
acceptance.

No arm named after an external system is reported. Comparing against QBugLM
requires executing the authors' release under this protocol.

## 5. Statistics

* Pass@k = fraction accepted within k attempts, read from the attempt trace.
* Rates: Wilson 95 % intervals.
* Paired outcomes → exact McNemar test on discordant pairs; effect size = paired
  risk difference with a Newcombe (method 10) interval; Holm correction over the
  rows of each contrast table.
* Resource overhead = gate count relative to the reference, over accepted
  repairs only.
* Experiments: **E1** main (suite 4, budget 8000), **E2** weak oracle (suite 1),
  **E3** query-budget sweep (50 … 3200).

## 6. Reproduction

```bash
python src/generate_dataset.py                 # 1000 circuits, deterministic (seed 42)
python src/audit.py                            # admissibility + observability audit
python src/run_experiments.py --workers 8 --out experiments/results_main.csv
python src/run_experiments.py --workers 8 --suite-size 1 --out experiments/results_weak_oracle.csv \
       --systems ours_full ours_reject_only unconstrained_diag unconstrained_reject_only
for Q in 50 100 200 400 800 1600 3200; do
  python src/run_experiments.py --workers 8 --oracle-budget $Q --out experiments/results_budget_$Q.csv \
         --systems ours_full unconstrained_reject_only
done
python src/run_experiments.py --workers 8 --out experiments/results_extra.csv        --systems ours_full_qiskit_router qiskit_transpile      # compiler baseline, router robustness
python src/analyze_results.py                  # tables -> paper/tables, figures -> paper/figures
pytest -q
```

All runs are deterministic: oracle states, search order and validator verdicts
are functions of the circuit id and candidate only.
