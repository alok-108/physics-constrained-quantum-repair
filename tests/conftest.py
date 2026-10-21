"""Shared pytest fixtures."""

import os
import sys
import warnings

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def metadata(root):
    import pandas as pd
    path = os.path.join(root, "data", "metadata.csv")
    assert os.path.exists(path), "data/metadata.csv missing: run `python src/generate_dataset.py`"
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def validator():
    from agents.validator import ValidatorAgent
    return ValidatorAgent()


def read_pair(root, row):
    buggy = open(os.path.join(root, row["buggy_file"]), encoding="utf-8").read()
    gt = open(os.path.join(root, row["ground_truth_file"]), encoding="utf-8").read()
    return buggy, gt
