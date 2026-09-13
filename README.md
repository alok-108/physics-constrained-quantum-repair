# Physics-Constrained Quantum Circuit Repair

A benchmark and multi-agent framework for joint fault localization, repair generation, and hardware-grounded validation of OpenQASM circuits under realistic superconducting quantum processor constraints.

---

## Research Context

- **Gap**: Current quantum program repair techniques either evaluate correctness only via ideal statevector simulation (e.g., QBugLM), consider purely logical equivalences (e.g., HornBro), are restricted to small scales (e.g., QRep's 13-qubit limit), or focus on ZX-calculus diagram optimization rather than semantic and physical repair (e.g., PyZX). No end-to-end multi-agent system exists that simultaneously performs fault localization, repair synthesis, and physical constraint verification.
- **Research Question**: *Can a multi-agent system localize and repair faulty OpenQASM circuits using hardware-level physical constraints and live calibration/noise feedback?*
- **Baselines to Beat**:
  - **QBugLM**: Simulation-only LLM repair lacking physical layout and coherence grounding.
  - **HornBro**: Logical equivalence repair omitting hardware noise and native gate sets.
  - **QRep**: Scalability limited to $\le 13$ qubits.
  - **PyZX**: ZX-calculus circuit simplification without defect-targeted bug repair.

---

## Directory Structure

```text
physics-constrained-quantum-repair/
├── .env.example                # Environment variable template (OPENAI_API_KEY)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/
│   ├── valid/                  # 1,000 ground-truth valid OpenQASM 3.0 circuits
│   ├── buggy/                  # 1,000 buggy OpenQASM 3.0 circuits (1 bug injected per circuit)
│   ├── ground_truth/           # 1,000 paired ground-truth repaired circuits
│   └── metadata.csv            # Manifest with circuit metadata and bug classifications
├── src/
│   ├── __init__.py
│   ├── physics_model.py        # IBM FakeSherbrooke (127-qubit) physical constraints model
│   └── generate_dataset.py     # 1,000 circuit generation and bug injection engine
├── agents/                     # Multi-agent repair framework (localization, repair, validation)
├── experiments/
│   ├── logs/                   # Execution logs
│   └── results/                # Quantitative benchmark evaluation results
├── research_paper/             # Publication artifacts (manuscript, Word doc, PDF)
│   ├── research_paper.pdf      # Complete publication-ready PDF paper
│   ├── research_paper.docx     # Formatted IEEE-style Word manuscript
│   ├── manuscript.tex          # LaTeX source manuscript
│   ├── references.bib          # BibTeX bibliography
│   └── build_paper.py          # Document build automation
├── paper/
│   ├── figures/                # Publication plots and circuit diagrams
│   └── tables/                 # LaTeX benchmark tables
└── tests/
    ├── test_dataset.py         # Pytest dataset verification suite
    └── test_agents.py          # Pytest multi-agent verification suite
```

---

## Hardware Physics Model (`src/physics_model.py`)

Grounds circuit repair against IBM's 127-qubit **FakeSherbrooke** Eagle-generation superconducting processor:
- **Coupling Graph**: Verifies whether two-qubit operations respect physical connectivity without routing overhead.
- **Calibrated Gate Error Rates**: Queries device infidelities for single-qubit ($X, \sqrt{X}, R_z$) and entangling ($CX, ECR$) gates.
- **Gate Durations**: Evaluates duration in nanoseconds for pulse scheduling.
- **Coherence Verification**: Asserts that cumulative operational and idle time on any qubit does not exceed physical $T_1$ and $T_2$ coherence limits ($T_1 \sim 250\,\mu\text{s}$, $T_2 \sim 150\,\mu\text{s}$).

---

## Dataset Overview

The dataset contains **1,000 circuits** evenly split across 4 algorithmic families:
1. **GHZ State Preparation** (3–7 qubits)
2. **QAOA MaxCut Ansatz** (4–8 qubits)
3. **VQE Molecular Hydrogen ($H_2$) Ansatz** (4 qubits)
4. **Quantum Fourier Transform (QFT)** (4–6 qubits)

Each circuit is serialized in **OpenQASM 3.0** with exactly one injected bug from 5 taxonomy classes (200 circuits each):
1. **Syntax Error**: Malformed QASM syntax, omitted semicolons, corrupt keywords, unclosed brackets.
2. **Structural Error**: Missing entangling/superposition gates, inverted control/target qubits.
3. **Gate Redundancy**: Self-canceling duplicate gates ($X \cdot X$, $H \cdot H$, identical $CX \cdot CX$).
4. **Semantic Deviation**: Inverted rotation angles ($-\theta$), shifted phase angles, basis changes ($R_x \leftrightarrow R_z$).
5. **Physics Violation**: Non-nearest-neighbor coupling violations on FakeSherbrooke, or delays exceeding $T_2$ coherence.

### `data/metadata.csv` Schema
| Column | Type | Description |
|---|---|---|
| `circuit_id` | str | Unique circuit identifier (e.g. `circuit_0001`) |
| `circuit_type` | str | Algorithm class (`ghz`, `qaoa_maxcut`, `vqe_h2`, `qft`) |
| `num_qubits` | int | Number of active qubits in circuit |
| `bug_type` | str | Bug class (`syntax`, `structural`, `gate_redundancy`, `semantic_deviation`, `physics_violation`) |
| `valid_file` | str | Path to original valid OpenQASM 3.0 file |
| `buggy_file` | str | Path to buggy OpenQASM 3.0 file |
| `ground_truth_file` | str | Path to ground-truth repaired OpenQASM 3.0 file |
| `bug_description` | str | Detailed explanation of the injected defect |

---

## Installation & Setup

1. **Clone/Navigate to project root**:
   ```bash
   cd /path/to/physics-constrained-quantum-repair
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and enter your OPENAI_API_KEY
   ```

---

## Dataset Generation

To generate or rebuild the 1,000 circuits and manifest:
```bash
python src/generate_dataset.py
```

---

## Running Tests

Run the complete automated test suite verifying dataset integrity, execution, bug triggering, and physical models:
```bash
pytest -v tests/test_dataset.py
```
