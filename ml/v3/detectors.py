"""
ml/v3/detectors.py — two triggerbot detectors with a shared interface.

BaselineDetector  : the "obvious" reaction-latency rule. Catches a naive triggerbot but
                    collapses as the cheat humanizes its fire latency. Represents what a
                    first-pass solution (or a generalist agent) typically produces.

RobustDetector    : flags sessions by the fraction of shots fired at high crosshair
                    angular velocity (incidental sweep-grazes). Survives latency
                    humanization, so it holds across the whole evasion sweep. This is the
                    reference / oracle detector.

Both expose:
    .score(df)   -> float   (higher == more cheat-like)
    .predict(df) -> "cheat" | "clean"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import fire_angular_velocity, aimed_shot_dwell


class Detector:
    name = "detector"
    threshold = 0.5

    def score(self, df: pd.DataFrame) -> float:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> str:
        return "cheat" if self.score(df) >= self.threshold else "clean"


class BaselineDetector(Detector):
    """Fraction of *aimed* shots fired with a sub-human on-target dwell ("reaction time").
    Naive: defeated once the cheat samples human-like delays before firing."""

    name = "baseline_reaction_latency"

    def __init__(self, dwell_floor_s: float = 0.10, threshold: float = 0.20):
        self.dwell_floor_s = dwell_floor_s
        self.threshold = threshold

    def score(self, df: pd.DataFrame) -> float:
        dwell = aimed_shot_dwell(df)
        if dwell.size == 0:
            return 0.0
        return float(np.mean(dwell < self.dwell_floor_s))


class RobustDetector(Detector):
    """Fraction of shots fired while the crosshair is still moving fast (not settled).
    A human settles before firing; a triggerbot's incidental sweep-grazes do not. This
    coupling survives latency humanization."""

    name = "robust_fire_kinematics"

    def __init__(self, av_floor_deg_s: float = 200.0, threshold: float = 0.08):
        self.av_floor_deg_s = av_floor_deg_s
        self.threshold = threshold

    def score(self, df: pd.DataFrame) -> float:
        fav = fire_angular_velocity(df)
        if fav.size == 0:
            return 0.0
        return float(np.mean(fav > self.av_floor_deg_s))


if __name__ == "__main__":
    import sim
    sessions, manifest = sim.generate_dataset(8, 3, seed=3)
    mi = manifest.set_index("session_id")
    b, r = BaselineDetector(), RobustDetector()
    print("%-16s %-6s %8s %8s %6s %6s" % ("session", "label", "base_s", "rob_s", "base", "rob"))
    for sid, df in list(sessions.items())[:12]:
        print("%-16s %-6s %8.3f %8.3f %6s %6s" % (
            sid, mi.loc[sid, "label"], b.score(df), r.score(df), b.predict(df), r.predict(df)))
