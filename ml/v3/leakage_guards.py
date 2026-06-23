"""
ml/v3/leakage_guards.py — automated guards against the three leakage modes documented
in v2's REPORT.md. These are *tests on the dataset*, not on a detector: they assert the
labels cannot be recovered through illegitimate shortcuts.

v2 leakage modes and the guard that now catches each:

  1. Rule leakage (labels were a deterministic function of measured features) ->
     guard_no_rule_leakage: no single RAW feature separates the classes near-perfectly.
  2. Session-encoding via inconsistent fields -> removed by construction (one consistent
     code path) and covered by guard_positional_at_chance (no field encodes session id).
  3. Positional session-encoding (position == which session == which class) ->
     guard_positional_at_chance: a model on positional/raw features only must sit at
     chance under group-aware CV.

Plus guard_behavioral_separates: confirms the *legitimate* behavioural signal is present
(otherwise a "leak-free" dataset could just be unlearnable).

Self-contained: pure numpy logistic regression + rank AUC, group-aware CV. No sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features import session_feature_row

BEHAVIORAL = ["fire_av_hi_frac", "fire_av_p90", "fire_av_median"]
RAW_OR_POSITIONAL = ["player_x_mean", "player_z_mean", "aim_yaw_mean",
                     "snap_delta_mean", "n_rows"]


# --------------------------------------------------------------------------- #
# tiny, deterministic ML primitives                                           #
# --------------------------------------------------------------------------- #
def rank_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic. Symmetric -> report max(auc, 1-auc)
    so a perfectly anti-correlated feature still registers as separating."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = y == 1, y == 0
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return 0.5
    auc = (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return float(max(auc, 1.0 - auc))


def _logreg_fit(X, y, iters=600, lr=0.3, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return w, b


def grouped_cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   folds: int = 5, seed: int = 0) -> float:
    """Group-aware K-fold AUC. Each group (session) lands entirely in one fold, so the
    model can never train and test on the same session."""
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % folds for i, g in enumerate(uniq)}
    assign = np.array([fold_of[g] for g in groups])
    scores = np.full(len(y), np.nan)
    for f in range(folds):
        tr, te = assign != f, assign == f
        if y[tr].sum() in (0, tr.sum()) or te.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w, b = _logreg_fit((X[tr] - mu) / sd, y[tr], seed=seed + f)
        scores[te] = ((X[te] - mu) / sd) @ w + b
    ok = ~np.isnan(scores)
    return rank_auc(scores[ok], y[ok])


# --------------------------------------------------------------------------- #
# building the session-level feature table                                    #
# --------------------------------------------------------------------------- #
def build_feature_table(sessions: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    mi = manifest.set_index("session_id")
    rows = []
    for sid, df in sessions.items():
        feat = session_feature_row(df)
        feat["session_id"] = sid
        feat["y"] = int(mi.loc[sid, "label"] == "cheat")
        rows.append(feat)
    return pd.DataFrame(rows)


@dataclass
class GuardResult:
    name: str
    passed: bool
    detail: str


def _cols(tab, cols):
    return tab[cols].to_numpy(dtype=float)


def guard_positional_at_chance(tab: pd.DataFrame, ceiling: float = 0.70) -> GuardResult:
    auc = grouped_cv_auc(_cols(tab, RAW_OR_POSITIONAL), tab["y"].to_numpy(),
                         tab["session_id"].to_numpy())
    return GuardResult("positional_at_chance", auc <= ceiling,
                       f"positional/raw group-CV AUC={auc:.3f} (ceiling {ceiling:.2f}; "
                       f"0.5==chance)")


def guard_no_rule_leakage(tab: pd.DataFrame, ceiling: float = 0.97) -> GuardResult:
    worst = 0.0
    worst_col = None
    for c in RAW_OR_POSITIONAL:
        a = rank_auc(tab[c].to_numpy(dtype=float), tab["y"].to_numpy())
        if a > worst:
            worst, worst_col = a, c
    return GuardResult("no_rule_leakage", worst <= ceiling,
                       f"best single RAW-feature AUC={worst:.3f} via '{worst_col}' "
                       f"(ceiling {ceiling:.2f}; ~1.0 would mean a v2-style rule leak)")


def guard_behavioral_separates(tab: pd.DataFrame, floor: float = 0.90) -> GuardResult:
    auc = grouped_cv_auc(_cols(tab, BEHAVIORAL), tab["y"].to_numpy(),
                         tab["session_id"].to_numpy())
    return GuardResult("behavioral_separates", auc >= floor,
                       f"behavioural group-CV AUC={auc:.3f} (floor {floor:.2f})")


def run_all_guards(sessions: dict, manifest: pd.DataFrame) -> list[GuardResult]:
    tab = build_feature_table(sessions, manifest)
    return [guard_positional_at_chance(tab),
            guard_no_rule_leakage(tab),
            guard_behavioral_separates(tab)]


if __name__ == "__main__":
    import sim
    sessions, manifest = sim.generate_dataset(60, 24, seed=2024)
    print(f"sessions: {len(sessions)} | labels: {manifest.label.value_counts().to_dict()}\n")
    for g in run_all_guards(sessions, manifest):
        print(f"[{'PASS' if g.passed else 'FAIL'}] {g.name}: {g.detail}")
