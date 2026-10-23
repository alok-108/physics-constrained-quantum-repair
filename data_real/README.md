# Real-bug pilot (5 pairs)

Source: **Bugs4Q**, a benchmark of real bugs in Qiskit programs (Apache-2.0),
<https://github.com/Z-928/Bugs4Q>. These files are *not* copies of that repository: each pair is the circuit
that the buggy script and the fixed script of one entry build, dumped as OpenQASM 3 by
`qiskit.qasm3.dumps` after running the (old-API) script under a small compatibility shim.

How the three pairs were chosen (all steps are mechanical except the last):

1. All Python scripts of the Qiskit entries were grouped into buggy / fixed pairs (135 groups; the repository
   mixes buggy, fixed and "mod" files across folders).
2. Each script was run in its own process (timeout, no network, IBM and simulator calls stubbed). 18 pairs
   yielded circuits for both versions.
3. 12 of those 18 differ. Reading each by hand: 3 are circuit-level logic fixes on the same qubits
   (`real_01`..`real_03`). The others are library bugs (the buggy and fixed scripts build the same circuit and
   differ only inside Qiskit, for example the transpiler), simulator-version effects (a dropped measurement),
   different qubit counts, a spelling of the same gate, or two different programs.

A second pass over the **Bugs4Q-Framework** release (42 Qiskit bugs, <https://github.com/Z-928/Bugs4Q-Framework>) added
two more circuit-logic pairs (`real_04`, `real_05`; entries 29 and 39); entries 26 and 31 there are the same bugs as
`real_02` and `real_01`. Entry 30 (an 8-qubit Grover-type program with resets) is a logic bug too but its exact
output distribution is too expensive for the dynamic-program validator, so it is left out.

So the honest summary is: **most Bugs4Q entries are not circuit-logic bugs**, and only 5 could be turned into
(buggy, reference) pairs. The selection is biased towards bugs that can be expressed this way and ran under the
shim; 3 pairs cannot support a rate, only an existence statement. Reproduce with `python src/eval_real.py`.

| id | Bugs4Q entry | bug |
|---|---|---|
| real_01 | StackExchange/9 | measurement results written to the classical bits in the wrong order |
| real_02 | stackoverflow-6-10/bug_2 | H gate where an X gate was intended |
| real_03 | StackExchange/10 | teleportation-style fix-up written as non-adjacent CX/CZ gates after measurement instead of a Z correction |
| real_04 | Framework 29 | QFT-based program: the qubit-order swap is missing and the gates are ordered differently |
| real_05 | Framework 39 | the four initial Hadamard gates are missing |

`real_04` and `real_05` use **logic-only acceptance** (`device_check = 0` in `metadata.csv`): their own reference
programs contain non-adjacent two-qubit gates, so the device-adjacency stage would reject the reference itself.

`real_03` is rejected at the *connectivity* stage first (non-adjacent two-qubit gates under the identity layout
of this benchmark), which is a property of our device model and not of the original bug.
