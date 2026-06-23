"""
ml/v3/evasion_sweep.py — deterministic evasion-robustness evaluation.

This is the "above a classifier" core: a detector is not graded on accuracy over one
distribution, but on whether it HOLDS UP as the triggerbot's evasion strength increases.
The harness:

  1. generates a seeded corpus spanning evasion levels 1..L (plus clean sessions),
  2. runs the detector on every session,
  3. reports a robustness curve (recall at each evasion level) and the clean
     false-positive rate,
  4. issues a pass/fail verdict against (recall_floor, fpr_ceiling) thresholds.

A naive detector posts high recall at low evasion and collapses at high evasion -> FAIL.
An evasion-robust detector holds recall across the whole sweep -> PASS. Everything is
seeded, so the verdict is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sim


@dataclass
class SweepResult:
    detector_name: str
    fpr: float                         # false-positive rate on clean sessions
    recall_by_level: dict              # evasion_level -> recall
    n_clean: int
    n_cheat_by_level: dict

    @property
    def min_recall(self) -> float:
        return min(self.recall_by_level.values()) if self.recall_by_level else 0.0

    def passes(self, recall_floor: float = 0.90, fpr_ceiling: float = 0.05) -> bool:
        return self.fpr <= fpr_ceiling and self.min_recall >= recall_floor

    def report(self, recall_floor: float = 0.90, fpr_ceiling: float = 0.05) -> str:
        lines = [f"Evasion sweep — detector: {self.detector_name}",
                 f"  clean FPR: {self.fpr:.3f}  (ceiling {fpr_ceiling:.2f})"]
        for lvl in sorted(self.recall_by_level):
            r = self.recall_by_level[lvl]
            bar = "#" * int(round(r * 30))
            flag = "" if r >= recall_floor else "  <- below floor"
            lines.append(f"  L{lvl}  recall {r:5.3f} |{bar:<30}|{flag}")
        verdict = "PASS" if self.passes(recall_floor, fpr_ceiling) else "FAIL"
        lines.append(f"  min recall: {self.min_recall:.3f}  (floor {recall_floor:.2f})"
                     f"   =>  {verdict}")
        return "\n".join(lines)


def run_sweep(detector, n_clean: int = 60, n_cheat_per_level: int = 40,
              evasion_levels=range(1, sim.MAX_EVASION + 1), seed: int = 2024) -> SweepResult:
    evasion_levels = list(evasion_levels)
    sessions, manifest = sim.generate_dataset(
        n_clean=n_clean, n_cheat_per_level=n_cheat_per_level,
        evasion_levels=evasion_levels, seed=seed)
    mi = manifest.set_index("session_id")

    fp = 0
    hits = {lvl: 0 for lvl in evasion_levels}
    totals = {lvl: 0 for lvl in evasion_levels}
    for sid, df in sessions.items():
        pred = detector.predict(df)
        row = mi.loc[sid]
        if row["label"] == "clean":
            fp += int(pred == "cheat")
        else:
            lvl = int(row["evasion_level"])
            totals[lvl] += 1
            hits[lvl] += int(pred == "cheat")

    recall = {lvl: (hits[lvl] / totals[lvl] if totals[lvl] else 0.0) for lvl in evasion_levels}
    return SweepResult(
        detector_name=getattr(detector, "name", type(detector).__name__),
        fpr=fp / max(1, n_clean),
        recall_by_level=recall,
        n_clean=n_clean,
        n_cheat_by_level=totals,
    )


if __name__ == "__main__":
    from detectors import BaselineDetector, RobustDetector
    for det in (BaselineDetector(), RobustDetector()):
        res = run_sweep(det, n_clean=60, n_cheat_per_level=40, seed=2024)
        print(res.report())
        print()
