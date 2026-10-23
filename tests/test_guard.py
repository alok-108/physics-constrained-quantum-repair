"""Oracle-guarded inverse cancellation (ours_full_ext_guard): fixes a misplaced rotation, keeps real redundancy fixes."""

import os

import pandas as pd
import pytest

from agents.orchestrator import Orchestrator
from src.run_experiments import ARM_CONFIG


def run(root, validator, cid, arm):
    m = pd.read_csv(os.path.join(root, "data", "metadata.csv")).set_index("circuit_id").loc[cid]
    orch = Orchestrator(max_iterations=3, validator=validator, variant=arm, oracle_budget=8000, oracle_suite_size=4,
                        **ARM_CONFIG[arm])
    return orch.run(cid, open(os.path.join(root, m.buggy_file), encoding="utf-8").read(),
                    open(os.path.join(root, m.ground_truth_file), encoding="utf-8").read(),
                    {"circuit_type": m.circuit_type, "num_qubits": int(m.num_qubits)})


def test_misplaced_rotation_is_repaired_only_with_the_guard(root, validator):
    # circuit_0282: the rotation between two CX moved to another wire; the two CX now look cancellable
    assert not run(root, validator, "circuit_0282", "ours_full_ext")["success"]
    res = run(root, validator, "circuit_0282", "ours_full_ext_guard")
    assert res["success"] and res["iterations"] == 1 and res["history"][0]["strategy"] == "search"


@pytest.mark.parametrize("cid", ["circuit_0053", "circuit_0303"])
def test_genuine_redundancy_is_still_cancelled(root, validator, cid):
    meta = pd.read_csv(os.path.join(root, "data", "metadata.csv")).set_index("circuit_id")
    assert meta.loc[cid, "bug_type"] == "gate_redundancy"
    res = run(root, validator, cid, "ours_full_ext_guard")
    assert res["success"] and res["history"][0]["strategy"] == "cancel"
