# How qrepair relates to existing quantum debugging and repair tools

Scope: what each tool does according to its own documentation or paper (web search,
2026-10-22), and what `qrepair` adds. This is **not** a head-to-head evaluation: no
other tool was run on this benchmark, and no claim of superiority is made.

## Tools surveyed

| Tool | What it does (as reported) | Needs from the user | Repairs? |
|---|---|---|---|
| **MQT Debugger** (Rovara, Burgholzer, Wille) | Interactive stepping on OpenQASM input; user-placed assertions (exact or approximate state, entanglement, superposition) checked by classical simulation; after a failure, diagnosis by cone-of-influence analysis; assertion refinement by moving assertions earlier through commutation rules | Hand-written assertions | No |
| **Proq** (Li et al., 2020) | Projection-based runtime assertions checked with a few projective measurements, including on a real quantum computer; more expressive than earlier classical-logic assertions | Assertions as projections | No |
| **QChecker** | Static analysis that detects bugs in Qiskit programs | Nothing | No |
| **Cirquo** | Slices a Qiskit circuit into subcircuits to analyse them separately | Manual use | No |
| **Qiskit Trebugger** | Logs and displays the transpiler passes applied to a circuit | Nothing | No |
| **HornBro** (Tan et al., FSE 2025) | Automated repair, homotopy-like iteration between classical and quantum parts; reports a high success rate on its own bug types; can add many gates (up to 249 reported in one study) | A test suite / implication assertions | Yes |
| **UnitAR** (Li et al., TOSEM 2024) | Generate-and-validate repair over an algebraic model of superposition and entanglement; evaluated on a small mixed set (mutants and real programs) | Test cases | Yes |
| **LLM repair** (ChatGPT on Bugs4Q; mutation-guided variants) | Prompt-based repair; one study reports 29 of 38 Bugs4Q bugs repaired after enriched dialogue and far lower success on complex algorithm-implementation bugs | A benchmark with tests | Yes |

Sources: MQT Debugger documentation and QSW 2025 paper (arXiv 2412.12269); Proq
(arXiv 1911.12855); "On the need for effective tools for debugging quantum programs"
(arXiv 2402.09547); HornBro (FSE 2025); UnitAR (TOSEM, doi 10.1145/3664604);
Guo et al. (Q-SE 2024); arXiv 2601.12273. Repository links for the repair tools were
not found in the search, so their artefacts were not run.

## What qrepair does

| Capability | Command | Needs |
|---|---|---|
| Static checks: parse errors with a probable line, non-adjacent two-qubit gates, coherence budget, cancellable pairs, no-op rotations, idle qubits | `lint` | nothing |
| Rank instructions by the agreement gain of the best single edit there | `localize` | reference circuit or test file |
| Acceptance check with the failing stage named | `verify` | reference circuit or test file |
| Minimal repair with a unified diff, JSON report, device-aware routing and delay trimming | `repair` | reference circuit or test file |

Specifications are either a reference circuit (strongest: exact unitary equivalence plus
the noise criterion) or a test file of amplitude pairs (weaker; the report states that
passing is not an equivalence certificate). Tests are amplitudes rather than output
probabilities because population-only checks miss phase faults.

## Where qrepair differs, and what is measured

* **Diagnosis and repair in one tool.** The debuggers above localise or detect and leave
  the fix to the user; the repair tools do not provide stepping or lint.
* **Localisation without user assertions.** The localiser needs only a specification and
  scores every single-instruction edit with a meet-in-the-middle oracle. Measured on the
  benchmark's 400 structural and semantic faults (position of the faulty instruction taken
  from the faulty-vs-reference difference): hit@1 / hit@3 / hit@5 =
  86.5 / 91.0 / 92.8 % with the base edit space and 86.2 / 93.8 / 95.2 % with the extended
  one, against 7.7 / 22.9 / 36.3 % for a random ranking. Weak spots: removed
  instructions (66 % hit@1 with the extended space).
* **Overhead and minimality are explicit.** Acceptance includes an irreducibility check, and
  repairs are reported with gate counts against the reference (accepted repairs average
  0.1 % above the reference). One tool above reports adding up to 249 gates.
* **Leakage-controlled evaluation.** The repair side never sees the reference; search cost
  is counted in oracle queries and reported, and overfitting to the test suite is measured
  against held-out states and exact process fidelity (reference mode).
* **Device awareness.** Admissibility under the identity layout, a coherence budget and
  routing are part of acceptance and repair.

## What qrepair does not do (yet)

* No interactive stepping or assertions inside a running program (MQT Debugger).
* No execution on real hardware (Proq).
* Unitary subset of OpenQASM 3 only: control flow, timing, measurement and reset are not
  checked semantically.
* Faults are small edits; real faults often involve APIs, transpilation and measurement
  wiring, which this tool does not model.
* No comparison with HornBro, UnitAR, QBugLM or LLM agents under a matched protocol.
