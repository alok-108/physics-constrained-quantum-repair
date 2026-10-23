"""Training-set generator: disjoint from the benchmark at pair level, splits never share a faulty circuit."""

import pytest

from src import generate_train as T


def test_refuses_the_benchmark_seed(tmp_path):
    with pytest.raises(SystemExit, match="seed 42"):
        T.build(str(tmp_path), 8, 42)


def test_pairs_are_disjoint_from_the_benchmark_and_splits_do_not_share_faulty_text(tmp_path, root):
    df = T.build(str(tmp_path), 16, 7000001, val_fraction=0.5)
    pairs, _, _ = T.benchmark_digests(root)
    assert len(df) > 0 and not any((b, r) in pairs for b, r in zip(df.buggy_digest, df.reference_digest))
    assert (df.pair_in_benchmark == 0).all() and df.split.isin(["train", "val"]).all()
    assert not (set(df[df.split == "train"].buggy_digest) & set(df[df.split == "val"].buggy_digest))
    assert df.buggy_digest.is_unique or df.duplicated(["buggy_digest", "reference_digest"]).sum() == 0
    assert "pair-level overlap with the benchmark: 0" in T.audit(df, 16)


def test_digest_ignores_whitespace():
    assert T.digest("a\n  b \n") == T.digest("a\nb")
