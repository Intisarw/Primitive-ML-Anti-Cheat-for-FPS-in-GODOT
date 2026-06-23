"""
ml/v3/tests/test_v3.py — property tests for the v3 adversarial-robustness suite.

Run from the repo root or anywhere:  pytest ml/v3/tests -v

These lock in the four claims the branch makes:
  * the simulator is deterministic and emits the leakage-free schema,
  * the naive baseline detector collapses under evasion (the difficulty gap is real),
  * the robust detector holds across the whole evasion sweep at low FPR,
  * the leakage guards pass, and the streaming detector stays within the frame budget.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sim                                            # noqa: E402
from detectors import BaselineDetector, RobustDetector  # noqa: E402
from evasion_sweep import run_sweep                   # noqa: E402
from leakage_guards import run_all_guards             # noqa: E402
from streaming import benchmark, FRAME_BUDGET_US      # noqa: E402


# ----------------------------------------------------------------------------- #
# fixtures                                                                       #
# ----------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def corpus():
    return sim.generate_dataset(n_clean=40, n_cheat_per_level=14, seed=2024)


# ----------------------------------------------------------------------------- #
# simulator                                                                      #
# ----------------------------------------------------------------------------- #
def test_determinism_same_seed():
    a_s, a_m = sim.generate_dataset(8, 3, seed=123)
    b_s, b_m = sim.generate_dataset(8, 3, seed=123)
    pd.testing.assert_frame_equal(a_m, b_m)
    for sid in a_s:
        pd.testing.assert_frame_equal(a_s[sid], b_s[sid])


def test_different_seeds_differ():
    a_s, _ = sim.generate_dataset(8, 3, seed=1)
    b_s, _ = sim.generate_dataset(8, 3, seed=2)
    sid = next(iter(a_s))
    assert not a_s[sid].equals(b_s[sid])


def test_schema_has_no_label_column(corpus):
    sessions, _ = corpus
    df = next(iter(sessions.values()))
    assert list(df.columns) == ["session_id"] + sim.TELEMETRY_COLUMNS
    assert "label" not in df.columns          # labels live only in the manifest


def test_manifest_labels_are_session_level(corpus):
    _, manifest = corpus
    assert set(manifest["label"]) == {"clean", "cheat"}
    assert manifest.groupby("session_id").size().max() == 1


# ----------------------------------------------------------------------------- #
# the difficulty gap                                                             #
# ----------------------------------------------------------------------------- #
def test_baseline_catches_naive_cheat():
    """The naive detector must work at low evasion -- otherwise the gap is meaningless."""
    res = run_sweep(BaselineDetector(), n_clean=40, n_cheat_per_level=20,
                    evasion_levels=[1], seed=7)
    assert res.recall_by_level[1] >= 0.90


def test_baseline_collapses_under_evasion():
    """The naive detector must fail the full sweep (recall collapses at high evasion)."""
    res = run_sweep(BaselineDetector(), n_clean=40, n_cheat_per_level=20, seed=7)
    assert not res.passes(recall_floor=0.90, fpr_ceiling=0.05)
    assert res.recall_by_level[sim.MAX_EVASION] < 0.5


def test_robust_passes_full_sweep():
    """The robust detector must hold recall across every level at low FPR."""
    res = run_sweep(RobustDetector(), n_clean=40, n_cheat_per_level=20, seed=7)
    assert res.passes(recall_floor=0.90, fpr_ceiling=0.05), res.report()
    assert res.fpr <= 0.05
    assert res.min_recall >= 0.90


def test_robust_strictly_beats_baseline_at_max_evasion():
    rb = run_sweep(BaselineDetector(), n_clean=30, n_cheat_per_level=20, seed=11)
    rr = run_sweep(RobustDetector(), n_clean=30, n_cheat_per_level=20, seed=11)
    lvl = sim.MAX_EVASION
    assert rr.recall_by_level[lvl] > rb.recall_by_level[lvl] + 0.4


# ----------------------------------------------------------------------------- #
# leakage guards                                                                 #
# ----------------------------------------------------------------------------- #
def test_all_leakage_guards_pass(corpus):
    sessions, manifest = corpus
    results = run_all_guards(sessions, manifest)
    failed = [g.name for g in results if not g.passed]
    assert not failed, f"leakage guards failed: {failed}\n" + \
        "\n".join(g.detail for g in results)


# ----------------------------------------------------------------------------- #
# streaming / latency                                                            #
# ----------------------------------------------------------------------------- #
def test_streaming_within_frame_budget(corpus):
    sessions, manifest = corpus
    bm = benchmark(sessions, manifest)
    assert bm["max_frame_us"] < FRAME_BUDGET_US
    assert bm["within_budget"]


def test_streaming_detects_without_false_alarms(corpus):
    sessions, manifest = corpus
    bm = benchmark(sessions, manifest)
    assert bm["clean_false_alarm_rate"] <= 0.05
    assert bm["cheat_detection_rate"] >= 0.90
    assert bm["median_time_to_detection_s"] is not None
