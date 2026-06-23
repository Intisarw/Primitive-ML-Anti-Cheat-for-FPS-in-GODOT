"""
ml/v3/demo.py — one-shot reproducible demonstration of the v3 suite.

Run:  python ml/v3/demo.py
Builds one seeded corpus and prints the four headline results:
  1. naive baseline collapses under evasion,
  2. robust kinematics detector holds across the whole sweep,
  3. all three leakage guards pass,
  4. the streaming detector stays within the 60 Hz frame budget.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sim
from detectors import BaselineDetector, RobustDetector
from evasion_sweep import run_sweep
from leakage_guards import run_all_guards
from streaming import benchmark

N_CLEAN, N_CHEAT_PER_LEVEL, SEED = 60, 40, 2024


def main():
    print("=" * 68)
    print("v3 adversarial-robustness demo  (seed=%d)" % SEED)
    print("=" * 68)
    sessions, manifest = sim.generate_dataset(N_CLEAN, N_CHEAT_PER_LEVEL, seed=SEED)
    print("corpus: %d sessions  |  %s\n" % (len(sessions),
          manifest.label.value_counts().to_dict()))

    print("1) NAIVE baseline detector (reaction-latency) across the evasion sweep")
    print(run_sweep(BaselineDetector(), N_CLEAN, N_CHEAT_PER_LEVEL, seed=SEED).report())
    print()
    print("2) ROBUST detector (fire-kinematics coupling) across the evasion sweep")
    print(run_sweep(RobustDetector(), N_CLEAN, N_CHEAT_PER_LEVEL, seed=SEED).report())
    print()

    print("3) Leakage guards")
    for g in run_all_guards(sessions, manifest):
        print("   [%s] %s: %s" % ("PASS" if g.passed else "FAIL", g.name, g.detail))
    print()

    print("4) Streaming detector benchmark")
    for k, v in benchmark(sessions, manifest).items():
        print("   %-28s %s" % (k, v))


if __name__ == "__main__":
    main()
