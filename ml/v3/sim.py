"""
ml/v3/sim.py — deterministic FPS telemetry simulator with a humanized triggerbot.

Why this exists
---------------
The v2 aimbot was "too clean" (mouse_dx == mouse_dy == 0), so a single rolling-std
feature trivially separated it. Real cheats evade exactly that. This module generates
telemetry for an *adversarial, humanized triggerbot* whose evasion strength is a knob.

Key modelling decision
----------------------
A triggerbot automates *firing*, not *aiming*. So the human crosshair trajectory
(slews, settles, jitter, fast sweeps) is drawn from the SAME distributions for clean and
cheat sessions. The ONLY difference is *which frames a shot fires on*:

  * Clean / skilled human  -> fires only after deliberately slewing onto a target and
                              letting the crosshair settle (low angular velocity).
  * Humanized triggerbot   -> fires whenever the crosshair crosses an enemy hitbox,
                              after a humanized reaction delay. Crucially it also fires
                              during fast *incidental sweeps* -- when the crosshair pans
                              across an enemy at speed -- shots a human would never take.

Resulting signals
------------------
  reaction-latency (NAIVE) : time from hitbox crossing to shot. Trivial to catch at low
                             evasion; the evasion knob humanizes this latency, so a
                             latency threshold collapses as evasion rises.

  fire-time crosshair angular velocity (ROBUST) : on a deliberate engagement the
                             crosshair decelerates onto the target, so both humans and
                             the bot fire at low angular velocity there. But on an
                             incidental *sweep* the crosshair never decelerates, so the
                             bot's grazing shots fire at high angular velocity -- a thing
                             humans never do. That high-velocity fire population survives
                             latency humanization, because a sweep is fast the whole way
                             through regardless of when within it the shot lands.

Because the aim telemetry is identically distributed between classes, a detector cannot
"cheat" by fingerprinting the generator -- it must learn the fire/aim *coupling*, which
is the actual cheat behaviour.

Everything is seeded: generate_dataset(seed=...) is bit-for-bit reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants grounded in the real v2 telemetry (README + data inspection):
#   dt ~0.017s (~60 Hz); fov_to_target in [0, 60] deg; flick spikes ~160 deg/frame.
# ---------------------------------------------------------------------------
DT = 1.0 / 60.0
FOV_CONE_DEG = 60.0          # only frames with an enemy in this cone are logged
HITBOX_DEG = 2.0             # angular radius counted as "crosshair on enemy"
SETTLE_AV_DEG_S = 120.0      # crosshair angular velocity below this == "settled"

TELEMETRY_COLUMNS = [
    "timestamp",
    "player_x", "player_y", "player_z",
    "enemy_x", "enemy_y", "enemy_z",
    "aim_yaw", "aim_pitch",
    "mouse_dx", "mouse_dy",
    "fov_to_target", "snap_delta",
    "time_to_kill", "is_firing", "enemy_killed",
]


@dataclass
class SkillProfile:
    """Human motor parameters. Skilled players react faster and slew faster, but they
    still settle before firing -- which is what keeps them off the cheat signal."""
    react_mu: float          # ex-Gaussian mean reaction time (s)
    react_sigma: float
    react_tau: float
    peak_speed_deg_s: float  # peak slew speed on a deliberate engagement
    jitter_deg: float        # per-frame crosshair jitter while settled (hand tremor)

AVERAGE = SkillProfile(0.22, 0.04, 0.06, 520.0, 0.14)
SKILLED = SkillProfile(0.16, 0.03, 0.035, 900.0, 0.10)


# Evasion levels 1..5 (0 is reserved for clean). `delay_*` humanizes crossing->fire
# latency to defeat the NAIVE feature. `incidental_keep` is the fraction of sweep grazes
# the bot still fires on; it shrinks at the top end so the ROBUST signal weakens but
# never vanishes (a triggerbot must keep firing to be useful).
EVASION = {
    0: dict(delay_mu=0.004, delay_sigma=0.002, delay_tau=0.002, incidental_keep=0.95),
    1: dict(delay_mu=0.05,  delay_sigma=0.02,  delay_tau=0.02,  incidental_keep=0.92),
    2: dict(delay_mu=0.10,  delay_sigma=0.03,  delay_tau=0.03,  incidental_keep=0.88),
    3: dict(delay_mu=0.15,  delay_sigma=0.035, delay_tau=0.045, incidental_keep=0.82),
    4: dict(delay_mu=0.19,  delay_sigma=0.04,  delay_tau=0.055, incidental_keep=0.74),
    5: dict(delay_mu=0.22,  delay_sigma=0.045, delay_tau=0.065, incidental_keep=0.66),
}
MAX_EVASION = max(EVASION)


def _exgauss(rng, mu, sigma, tau, floor=0.1):
    """Ex-Gaussian RT sample: Gaussian + exponential, right-skewed, hard floor."""
    return float(max(floor, rng.normal(mu, sigma) + rng.exponential(tau)))


def _min_jerk(n):
    """Min-jerk position profile in [0,1] over n samples (0 -> 1)."""
    if n <= 1:
        return np.array([1.0])
    t = np.linspace(0, 1, n)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


class _SessionBuilder:
    """Tracks the true crosshair direction (yaw/pitch) and emits telemetry frames.

    Each `_step` moves the crosshair to a new orientation and logs one frame, deriving
    mouse delta from the crosshair motion and fov/snap from the crosshair-vs-enemy angle.
    Fires are written in a second pass by index so we can compute look-back conditions.
    """

    def __init__(self, rng: np.random.Generator, skill: SkillProfile):
        self.rng, self.skill = rng, skill
        self.t = float(rng.uniform(0.5, 3.0))
        self.cyaw = float(rng.uniform(-180, 180))   # crosshair yaw (deg)
        self.cpitch = float(rng.uniform(2, 12))     # crosshair pitch (deg)
        self.prev_fov = None
        self.px, self.pz, self.py = rng.uniform(-8, 8), rng.uniform(-8, 8), 0.81
        self.rows: list[dict] = []
        self.fire_marks: list[tuple[int, int]] = []  # (row_index, killed)

    def _step(self, new_yaw, new_pitch, eyaw, epitch, dist):
        dyaw = (new_yaw - self.cyaw + 180) % 360 - 180
        dpitch = new_pitch - self.cpitch
        self.cyaw = (new_yaw + 180) % 360 - 180
        self.cpitch = new_pitch
        off = (((self.cyaw - eyaw) + 180) % 360 - 180)
        fov = min(FOV_CONE_DEG, float(np.hypot(off, self.cpitch - epitch)))
        snap = 0.0 if self.prev_fov is None else abs(fov - self.prev_fov)
        self.prev_fov = fov
        ez = self.pz - dist * np.cos(np.deg2rad(eyaw))
        ex = self.px + dist * np.sin(np.deg2rad(eyaw))
        self.rows.append({
            "timestamp": round(self.t, 4),
            "player_x": round(self.px, 4), "player_y": self.py, "player_z": round(self.pz, 4),
            "enemy_x": round(ex, 4), "enemy_y": round(epitch / 6 + 2.0, 4), "enemy_z": round(ez, 4),
            "aim_yaw": round(self.cyaw, 5), "aim_pitch": round(self.cpitch, 5),
            "mouse_dx": round(dyaw, 6), "mouse_dy": round(dpitch, 6),
            "fov_to_target": round(fov, 5), "snap_delta": round(snap, 5),
            "time_to_kill": -1.0, "is_firing": 0, "enemy_killed": 0,
        })
        self.t += DT
        return len(self.rows) - 1   # index of the frame just emitted

    def _crossing_index(self, start, end):
        """First frame in [start,end) where the crosshair is on the hitbox, else None."""
        for i in range(start, end):
            if self.rows[i]["fov_to_target"] < HITBOX_DEG:
                return i
        return None

    # -- deliberate engagement: slew onto a target then settle ----------------
    def engagement(self, is_cheat, evp):
        rng, sk = self.rng, self.skill
        theta0 = rng.uniform(12, 45)
        dist = rng.uniform(4, 28)
        eyaw = (self.cyaw + rng.choice([-1, 1]) * theta0 + 180) % 360 - 180
        epitch = self.cpitch + rng.uniform(-3, 3)
        start_yaw = self.cyaw

        for _ in range(int(rng.integers(2, 8))):     # idle before reacting
            self._step(self.cyaw + rng.normal(0, sk.jitter_deg),
                       self.cpitch + rng.normal(0, sk.jitter_deg), eyaw, epitch, dist)

        slew_n = max(4, int((theta0 / sk.peak_speed_deg_s) / DT * rng.uniform(2.2, 3.4)))
        prof = _min_jerk(slew_n)
        seg0 = len(self.rows)
        for s in prof:                                # ballistic slew (decelerates onto target)
            self._step(start_yaw + (eyaw - start_yaw) * s,
                       self.cpitch + rng.normal(0, sk.jitter_deg * 0.3), eyaw, epitch, dist)
        seg1 = len(self.rows)
        settle_n = int(rng.integers(7, 16))
        for _ in range(settle_n):                     # settle: micro-corrections, low velocity
            self._step(eyaw + rng.normal(0, sk.jitter_deg),
                       epitch + rng.normal(0, sk.jitter_deg), eyaw, epitch, dist)
        seg2 = len(self.rows)

        if not is_cheat:
            fr = seg1 + int(_exgauss(rng, sk.react_mu, sk.react_sigma, sk.react_tau) / DT)
            fr = min(max(fr, seg1), seg2 - 1)
            self.fire_marks.append((fr, int(rng.random() < 0.75)))
        else:
            cx = self._crossing_index(seg0, seg2)
            if cx is not None:
                d = int(_exgauss(rng, evp["delay_mu"], evp["delay_sigma"], evp["delay_tau"], DT) / DT)
                self.fire_marks.append((min(cx + d, seg2 - 1), int(rng.random() < 0.7)))

    # -- incidental sweep: fast constant-velocity pan across an enemy ---------
    def sweep(self, is_cheat, evp):
        rng = self.rng
        dist = rng.uniform(3, 18)                     # incidental grazes happen near enemies
        # enemy angular size: close enemies subtend a wide angle (easy to graze at speed)
        graze_deg = float(np.clip(np.rad2deg(2 * np.arctan(0.9 / dist)), 2.5, 18.0))
        span = rng.uniform(28, 60)
        speed = rng.uniform(480, 1100)                # deg/s, no deceleration (not targeting)
        n = max(6, int((span / speed) / DT))
        eyaw = (self.cyaw + rng.choice([-1, 1]) * span / 2 + 180) % 360 - 180
        epitch = self.cpitch + rng.uniform(-1.5, 1.5)
        direction = 1 if ((eyaw - self.cyaw + 180) % 360 - 180) > 0 else -1
        per = direction * span / n
        seg0 = len(self.rows)
        for _ in range(n):
            self._step(self.cyaw + per, self.cpitch + rng.normal(0, 0.2), eyaw, epitch, dist)
        seg1 = len(self.rows)

        # closest-approach frame; the bot fires if the crosshair grazed the enemy's
        # angular extent -- a shot a human would never take mid-pan.
        fovs = [self.rows[i]["fov_to_target"] for i in range(seg0, seg1)]
        gi = seg0 + int(np.argmin(fovs))
        if is_cheat and min(fovs) < graze_deg and rng.random() < evp["incidental_keep"]:
            d = int(_exgauss(rng, evp["delay_mu"], evp["delay_sigma"], evp["delay_tau"], DT) / DT)
            self.fire_marks.append((min(gi + d, seg1 - 1), 0))

    def to_frame(self, session_id):
        for idx, killed in self.fire_marks:
            self.rows[idx]["is_firing"] = 1
            self.rows[idx]["enemy_killed"] = killed
        df = pd.DataFrame(self.rows, columns=TELEMETRY_COLUMNS)
        df.insert(0, "session_id", session_id)
        return df


def generate_session(session_id, is_cheat, evasion_level, skill, seed,
                     n_engagements=22, n_sweeps=18):
    """Generate one session of telemetry (label returned separately by
    generate_dataset; rows carry NO label column -> no rule leakage)."""
    rng = np.random.default_rng(seed)
    evp = EVASION[evasion_level]
    b = _SessionBuilder(rng, skill)
    plan = ["E"] * n_engagements + ["S"] * n_sweeps
    rng.shuffle(plan)
    for kind in plan:
        (b.engagement if kind == "E" else b.sweep)(is_cheat, evp)
    return b.to_frame(session_id)


def generate_dataset(n_clean, n_cheat_per_level,
                     evasion_levels: Iterable[int] = range(1, MAX_EVASION + 1),
                     seed=42, skilled_fraction=0.35):
    """Build a corpus of sessions.

    Returns
    -------
    sessions : dict[session_id -> telemetry DataFrame]   (NO label column)
    manifest : DataFrame[session_id, label, skill, evasion_level, seed]
               `label` is the SESSION-LEVEL ground truth: "clean" or "cheat".
    """
    evasion_levels = list(evasion_levels)
    master = np.random.default_rng(seed)
    sessions, manifest = {}, []

    def pick(rng):
        return SKILLED if rng.random() < skilled_fraction else AVERAGE

    for i in range(n_clean):
        sid = f"clean_{i:03d}"
        s = int(master.integers(0, 2**31 - 1))
        prof = pick(np.random.default_rng(s))
        sessions[sid] = generate_session(sid, False, 0, prof, s)
        manifest.append(dict(session_id=sid, label="clean",
                             skill="skilled" if prof is SKILLED else "average",
                             evasion_level=0, seed=s))

    for lvl in evasion_levels:
        for i in range(n_cheat_per_level):
            sid = f"cheat_L{lvl}_{i:03d}"
            s = int(master.integers(0, 2**31 - 1))
            prof = pick(np.random.default_rng(s))
            sessions[sid] = generate_session(sid, True, lvl, prof, s)
            manifest.append(dict(session_id=sid, label="cheat",
                                 skill="skilled" if prof is SKILLED else "average",
                                 evasion_level=lvl, seed=s))

    return sessions, pd.DataFrame(manifest)


if __name__ == "__main__":
    sessions, manifest = generate_dataset(6, 2, seed=1)
    print("sessions:", len(sessions), "| labels:", manifest.label.value_counts().to_dict())
    sid = next(iter(sessions))
    df = sessions[sid]
    print(f"\nexample {sid}: {df.shape}, fires={int(df.is_firing.sum())}")
    print(df.head(4).to_string())
