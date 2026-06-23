# v3 — Adversarial Robustness

This branch (`v3-adversarial-robustness`) upgrades the project from *"can a classifier
detect a too-clean cheat?"* to *"does a detector stay robust as the cheat actively
evades — and can it run live, within budget, without leaking?"*

It directly closes the four future-work items called out in [`REPORT.md`](../../REPORT.md):
the *adversarial cheat*, *session-level labels*, *leakage elimination*, and the
*real-time in-engine detector*.

Everything here is **numpy + pandas only** (no scikit-learn / xgboost), fully **seeded**,
and verified by a pytest suite — so the headline results are reproducible bit-for-bit.

---

## Why v3

v2's honest limitations were:

- **"Aimbot is too clean."** It set `mouse_dx == mouse_dy == 0`, so one rolling-std
  feature separated it. Real cheats inject noise to evade exactly that.
- **Rule-based, per-row labels** created circular signal (the label *was* a function of
  three measured columns).
- **Single session per class** let *position* encode the label.
- The detector was **batch-only**; production anti-cheat must run live.

v3 replaces the "too clean" cheat with a **humanized triggerbot** whose evasion strength
is a knob, and grades detectors on **robustness across that knob** rather than accuracy on
one distribution.

---

## The core idea

A triggerbot automates *firing*, not *aiming*. So the simulator draws the **same human
aim** (slews, settles, jitter, fast sweeps) for clean and cheat sessions — the only
difference is *which frames a shot fires on*. That removes any "detect the generator"
artifact: the tell is purely the fire/aim coupling, i.e. the actual cheat behaviour.

Two signals fall out:

| Signal | What it is | Survives humanization? |
|---|---|---|
| **Reaction latency** (naive) | on-target dwell before a shot | ❌ the evasion knob humanizes it |
| **Fire-time crosshair angular velocity** (robust) | shots fired mid-sweep, never settled | ✅ incidental grazes are fast regardless of delay |

A human settles before firing; a triggerbot fires on incidental sweep-grazes at high
angular velocity — shots a human never takes. That coupling is delay-invariant.

---

## What's in `ml/v3/`

| File | Purpose |
|---|---|
| `sim.py` | Deterministic telemetry simulator. Humanized triggerbot with **evasion levels 1–5**, clean + skilled humans, real v2 schema, **no per-row label** (session-level labels in a manifest). |
| `features.py` | Telemetry features: crosshair angular velocity, aimed-shot dwell, session-level feature row. |
| `detectors.py` | `BaselineDetector` (naive reaction-latency) and `RobustDetector` (fire-kinematics) with a shared `score`/`predict` interface. |
| `evasion_sweep.py` | Deterministic robustness harness: recall at each evasion level + clean FPR, pass/fail vs thresholds. The "above-a-classifier" verifier shape. |
| `leakage_guards.py` | Automated guards for the three v2 leakage modes (self-contained numpy logreg + group-aware CV + rank AUC). |
| `streaming.py` | Online, latency-bounded detector: O(1)/frame, time-to-detection, per-frame latency vs the 60 Hz budget. |
| `demo.py` | One-shot reproducible run of all four results. |
| `tests/test_v3.py` | 11 property tests covering determinism, the difficulty gap, leakage guards, and latency. |

---

## Results (seed = 2024)

**1. Naive baseline collapses under evasion** — recall 1.00 → 0.75 → 0.00 as the cheat
humanizes its latency. **FAIL** the sweep.

**2. Robust detector holds** — recall **1.00 at every evasion level**, **0% FPR**. **PASS**.

**3. Leakage guards** — positional/raw features at chance (AUC ≈ 0.52, vs v2's positional
dominance); no single raw feature leaks the label (≈ 0.55, vs v2's ≈ 1.0 rule leak);
behavioural signal genuinely present (AUC 1.00).

**4. Streaming** — worst-case **≈ 0.2 ms/frame** (budget 16.67 ms), **0%** clean
false-alarm, **100%** cheat detection, median time-to-detection **≈ 1.7 s**.

---

## Run it

```bash
python ml/v3/demo.py          # headline results
pytest ml/v3/tests -v         # 11 property tests
```

No data files are written; the corpus is generated in-memory from a seed.

---

## Relationship to the Triton authoring task

This branch is also the seed material for a Triton terminal task: ship the telemetry
corpus + the failing baseline, ask the agent to build a detector that **passes the evasion
sweep** (recall ≥ floor across all levels, FPR ≤ ceiling, inference within the frame
budget), and grade it on a **held-out** set of evasion variants. Difficulty comes from
*discovering the evasion-invariant signal* (latency fails, kinematics holds), not from
curve-fitting one distribution — and the leakage guards keep the spec fair.
