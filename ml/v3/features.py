"""
ml/v3/features.py — telemetry feature extraction shared by the detectors and guards.

All features are derived from the logged schema only (aim_yaw/pitch, fov_to_target,
is_firing, timestamp). The two that matter for triggerbot detection:

  fire_latencies        : seconds from the crosshair reaching the target to the shot.
                          The NAIVE signal -- humanized away at high evasion.
  fire_angular_velocity : crosshair angular speed (deg/s) at each shot, from frame-to-
                          frame aim_yaw/pitch deltas. The ROBUST signal -- a human fires
                          only after settling (low speed); a triggerbot's incidental
                          sweep-grazes fire at high speed regardless of latency disguise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DT = 1.0 / 60.0
ONTARGET_DEG = 3.0     # fov below this == crosshair effectively on the target


def crosshair_angular_velocity(df: pd.DataFrame) -> np.ndarray:
    """Per-frame crosshair angular velocity (deg/s) from aim_yaw/aim_pitch deltas."""
    yaw = df["aim_yaw"].to_numpy(dtype=float)
    pitch = df["aim_pitch"].to_numpy(dtype=float)
    dyaw = (np.diff(yaw, prepend=yaw[:1]) + 180.0) % 360.0 - 180.0
    dpitch = np.diff(pitch, prepend=pitch[:1])
    ts = df["timestamp"].to_numpy(dtype=float)
    dt = np.diff(ts, prepend=ts[:1] - DT)
    dt = np.where(dt <= 0, DT, dt)
    return np.hypot(dyaw, dpitch) / dt


def fire_indices(df: pd.DataFrame) -> np.ndarray:
    return np.flatnonzero(df["is_firing"].to_numpy() == 1)


def fire_angular_velocity(df: pd.DataFrame) -> np.ndarray:
    av = crosshair_angular_velocity(df)
    fi = fire_indices(df)
    return av[fi] if fi.size else np.array([])


def fire_fov(df: pd.DataFrame) -> np.ndarray:
    """fov_to_target at each shot. Distinguishes aimed shots (on target) from grazes."""
    fi = fire_indices(df)
    return df["fov_to_target"].to_numpy(dtype=float)[fi] if fi.size else np.array([])


def aimed_shot_dwell(df: pd.DataFrame) -> np.ndarray:
    """For each *aimed* shot (crosshair on target when fired), the contiguous time the
    crosshair had been resting on the target before firing.

    This is the obvious "reaction time" feature: how long the player dwelled on the
    target before pulling the trigger. A naive triggerbot fires ~instantly (dwell ~0);
    a human reacts (~0.2 s). The cheat's evasion knob humanizes exactly this dwell, which
    is why a dwell threshold collapses at high evasion. Computed over the contiguous
    on-target run only, so it never bleeds across engagements."""
    fov = df["fov_to_target"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy(dtype=float)
    out = []
    for fi in fire_indices(df):
        if fov[fi] >= ONTARGET_DEG:
            continue                       # graze / off-target shot, not an "aimed" shot
        j = fi
        while j > 0 and fov[j - 1] < ONTARGET_DEG:
            j -= 1
        out.append(ts[fi] - ts[j])
    return np.array(out)


def session_feature_row(df: pd.DataFrame) -> dict:
    """A compact session-level feature vector, used by the leakage-guard classifier and
    for diagnostics. Mixes behavioural features with positional/raw ones on purpose so
    the guards can detect whether a model leans on non-behavioural shortcuts."""
    fav = fire_angular_velocity(df)
    dwell = aimed_shot_dwell(df)
    n_fire = int(df["is_firing"].sum())
    return {
        # behavioural (legitimate) signal
        "fire_av_hi_frac": float(np.mean(fav > 200.0)) if fav.size else 0.0,
        "fire_av_p90": float(np.percentile(fav, 90)) if fav.size else 0.0,
        "fire_av_median": float(np.median(fav)) if fav.size else 0.0,
        "aimed_dwell_median": float(np.median(dwell)) if dwell.size else 0.0,
        "aimed_dwell_lo_frac": float(np.mean(dwell < 0.10)) if dwell.size else 0.0,
        "fire_rate": n_fire / max(1, len(df)),
        # raw / positional features (potential leakage shortcuts)
        "player_x_mean": float(df["player_x"].mean()),
        "player_z_mean": float(df["player_z"].mean()),
        "aim_yaw_mean": float(df["aim_yaw"].mean()),
        "snap_delta_mean": float(df["snap_delta"].mean()),
        "n_rows": len(df),
    }
