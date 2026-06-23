"""
ml/v3/streaming.py — online, latency-bounded triggerbot detector.

The batch RobustDetector proves the signal exists; production anti-cheat must run it
LIVE, tick by tick, within a per-frame compute budget (here: well under one 60 Hz frame
== 16.67 ms) so it can flag a cheating session mid-match instead of post-hoc.

StreamingDetector keeps O(1) running state and updates once per frame. It latches a
"cheat" flag the first time enough high-angular-velocity shots have accumulated, then
reports:

  * flagged              : did it fire a detection on this session?
  * detection_time_s     : seconds from session start to the flag (time-to-detection)
  * max_frame_us         : worst single-frame processing time (the latency budget metric)

This realizes the ONNX-sidecar / in-engine CheatDetector future-work item from REPORT.md
as a pure-Python reference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

DT = 1.0 / 60.0
FRAME_BUDGET_US = 16_667.0   # one 60 Hz frame in microseconds


@dataclass
class StreamResult:
    flagged: bool
    detection_time_s: float | None     # None if never flagged
    frames: int
    max_frame_us: float
    mean_frame_us: float
    n_fires: int
    n_hi_av_fires: int


class StreamingDetector:
    """Online version of RobustDetector. Flags when at least `min_hi` high-velocity
    shots have occurred AND they make up more than `frac` of shots so far."""

    name = "streaming_robust"

    def __init__(self, av_floor_deg_s: float = 200.0, frac: float = 0.08,
                 min_hi: int = 3):
        self.av_floor = av_floor_deg_s
        self.frac = frac
        self.min_hi = min_hi
        self.reset()

    def reset(self):
        self._prev_yaw = None
        self._prev_pitch = None
        self._prev_t = None
        self.n_fires = 0
        self.n_hi = 0
        self.flagged = False

    def update(self, yaw: float, pitch: float, t: float, is_firing: int) -> bool:
        """Process one frame; return True the instant the session flips to flagged."""
        if self._prev_yaw is None:
            self._prev_yaw, self._prev_pitch, self._prev_t = yaw, pitch, t
            return False
        dyaw = (yaw - self._prev_yaw + 180.0) % 360.0 - 180.0
        dpitch = pitch - self._prev_pitch
        dt = t - self._prev_t
        dt = dt if dt > 0 else DT
        av = (dyaw * dyaw + dpitch * dpitch) ** 0.5 / dt
        self._prev_yaw, self._prev_pitch, self._prev_t = yaw, pitch, t

        if is_firing:
            self.n_fires += 1
            if av > self.av_floor:
                self.n_hi += 1
        newly = False
        if not self.flagged and self.n_hi >= self.min_hi and \
                self.n_hi > self.frac * self.n_fires:
            self.flagged = True
            newly = True
        return newly


def run_stream(detector: StreamingDetector, df: pd.DataFrame) -> StreamResult:
    detector.reset()
    yaw = df["aim_yaw"].to_numpy(dtype=float)
    pitch = df["aim_pitch"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy(dtype=float)
    fire = df["is_firing"].to_numpy(dtype=int)
    t0 = ts[0] if len(ts) else 0.0
    det_time = None
    per_frame_us = np.empty(len(df))
    for i in range(len(df)):
        c0 = time.perf_counter()
        newly = detector.update(yaw[i], pitch[i], ts[i], fire[i])
        per_frame_us[i] = (time.perf_counter() - c0) * 1e6
        if newly and det_time is None:
            det_time = ts[i] - t0
    return StreamResult(
        flagged=detector.flagged,
        detection_time_s=det_time,
        frames=len(df),
        max_frame_us=float(per_frame_us.max()) if len(df) else 0.0,
        mean_frame_us=float(per_frame_us.mean()) if len(df) else 0.0,
        n_fires=detector.n_fires,
        n_hi_av_fires=detector.n_hi,
    )


def benchmark(sessions: dict, manifest: pd.DataFrame) -> dict:
    """Aggregate streaming behaviour across a corpus: clean false-alarm rate, cheat
    detection rate, time-to-detection, and worst-case per-frame latency."""
    mi = manifest.set_index("session_id")
    det = StreamingDetector()
    max_us = 0.0
    clean_fa = clean_n = 0
    cheat_hit = cheat_n = 0
    ttd = []
    for sid, df in sessions.items():
        res = run_stream(det, df)
        max_us = max(max_us, res.max_frame_us)
        if mi.loc[sid, "label"] == "clean":
            clean_n += 1
            clean_fa += int(res.flagged)
        else:
            cheat_n += 1
            if res.flagged:
                cheat_hit += 1
                ttd.append(res.detection_time_s)
    return {
        "max_frame_us": max_us,
        "frame_budget_us": FRAME_BUDGET_US,
        "within_budget": max_us < FRAME_BUDGET_US,
        "clean_false_alarm_rate": clean_fa / max(1, clean_n),
        "cheat_detection_rate": cheat_hit / max(1, cheat_n),
        "median_time_to_detection_s": float(np.median(ttd)) if ttd else None,
        "p90_time_to_detection_s": float(np.percentile(ttd, 90)) if ttd else None,
    }


if __name__ == "__main__":
    import sim
    sessions, manifest = sim.generate_dataset(40, 16, seed=99)
    bm = benchmark(sessions, manifest)
    print("Streaming benchmark")
    for k, v in bm.items():
        print(f"  {k}: {v}")
