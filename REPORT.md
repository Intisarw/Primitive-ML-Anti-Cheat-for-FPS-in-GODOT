# Project Report

## Behavioral Anti-Cheat for First-Person Shooters

**Version:** v2.0   **Period:** May–June 2026

---

## Executive Summary

End-to-end machine learning pipeline for detecting aimbot cheats in FPS games using gameplay telemetry. A Godot-based FPS (Kenney Starter Kit) is instrumented with a toggleable aimbot, capturing 60 Hz behavioral telemetry. Three classical classifiers (Logistic Regression, Random Forest, XGBoost) are trained to distinguish clean play from cheating, with class balancing to handle severe imbalance. A locally-served Llama 3.1 8B (Ollama) serves as a moderator-facing explainer for flagged events.

**Headline result:** XGBoost trained on leak-free behavioral features achieves **0.81 F1 score and 84% recall on the cheat class**, catching 46 of 55 cheaters in the test set with only 1 false positive across 10,500 clean players. When integrated with the LLM explainer pipeline, all 5 top-confidence flags produced VERDICT_CHEAT with cited evidence from the telemetry.

The project demonstrates the **cheap-triage + expensive-explanation pattern** used in production content moderation: a fast classical model handles every frame; a slow LLM only fires on flagged events.

---

## Problem Statement

Multiplayer FPS games face persistent automated cheating (aimbots, triggerbots). Commercial anti-cheat systems (Riot Vanguard, BattlEye, Easy Anti-Cheat, Valve VAC) combine kernel-level telemetry with classical ML to detect physically-impossible input patterns. This project replicates that pattern at small scale across four stages:

1. **Game instrumentation** for telemetry collection
2. **Behavioral feature engineering** (not just raw inputs)
3. **Multi-class classification** with class balancing
4. **LLM-based reasoning** over flagged events

---

## Methodology

### Data Collection
A Godot autoload (`scripts/Logger.gd`, registered as `MLLogger`) writes a 17-column CSV per session: position (x/y/z for player and enemy), aim (yaw/pitch), mouse delta, FOV-to-target, snap delta, frame events, and a rule-based label.

| Session | Duration | Rows | Labels |
|---|---|---|---|
| `clean_001.csv` | ~10 min | 36,918 | 100% clean |
| `cheat_001.csv` | ~7 min | 20,737 | 76% clean / 23% suspect / 1.3% cheat |

Combined: 57,653 rows after first-row drop per session. Class distribution: 91% clean, 8.5% suspect, 0.5% cheat.

### Features
Six derived features computed per session: 30-frame rolling std of mouse_dx and mouse_dy (jitter), angular velocity (snap_delta / dt), distance to enemy, frame-to-frame FOV rate of change, frame duration. Each targets a specific physical impossibility produced by aimbots:

| Feature | Captures |
|---|---|
| `mouse_dx_std30`, `mouse_dy_std30` | Human hand tremor — present in clean play, absent in aimbot frames |
| `angular_velocity` | Aim rotation speed — bounded for humans (~3000°/sec), unbounded for aimbots |
| `dist_to_enemy` | Context — same snap angle means different things at 2m vs 30m |
| `fov_rate` | Closing speed onto target |

### Models
Three sklearn Pipelines (StandardScaler + classifier): Logistic Regression, Random Forest (200 trees), XGBoost (200 boosting rounds). All trained with class balancing (`class_weight="balanced"` for sklearn models, `compute_sample_weight("balanced")` for XGBoost). Stratified 80/20 row-level split with `random_state=42`.

### LLM Explainer (v2)
For each high-confidence flag (XGBoost cheat probability ≥ 0.8), a 5-second telemetry window is summarized into ~15 lines of structured statistics and submitted to a locally-served Llama 3.1 8B via Ollama's HTTP API. The LLM produces a moderator-readable explanation ending with `VERDICT_CHEAT`, `VERDICT_CLEAN`, or `VERDICT_INCONCLUSIVE`. A 5-second debounce per session prevents redundant LLM calls during sustained cheating.

---

## Key Findings: Three Leakage Discoveries

The most technically interesting outcome was not the headline F1 — it was the iterative discovery and correction of three forms of data leakage through feature-ablation experiments.

### 1. Rule leakage
All-features training produced 100% recall and F1 on the cheat class — the textbook indicator of label leakage. The labeling rule in `aimbot_look()` uses `mouse_dx`, `mouse_dy`, and `fov_to_target`; trees with these inputs reconstructed the rule in three splits without learning generalizable patterns.

*Caught by:* anomalous perfect scores. *Confirmed by:* ablation dropping XGB F1 from 1.00 to 0.85.

### 2. Session-encoding via `is_firing`
After removing rule features, XGBoost still hit 100% recall, with `is_firing` accounting for 71% of feature importance. The `is_firing` field has inconsistent semantics between code paths in `player.gd` (`Input.is_action_pressed()` for clean rows vs `fired_this_frame` for cheat rows), so it encodes which logging branch produced the row rather than whether a shot occurred.

*Caught by:* feature importance analysis. *Confirmed by:* ablation dropping XGB F1 from 0.85 to 0.81.

### 3. Positional session-encoding
After removing `is_firing`, positional features (`player_y`, `aim_yaw`, `enemy_x`) became dominant. With only one session per class, position varies systematically between sessions and acts as a proxy for session identity rather than cheating behavior.

*Caught by:* feature importance plot still showing positional dominance. *Resolution:* requires multi-session data — flagged as v2.1 future work.

---

## Results

### Three-iteration model comparison (cheat-class F1)

| Feature set | Features | LR F1 | RF F1 | XGB F1 |
|---|---|---|---|---|
| All features | 20 | 0.42 | 1.00 | 1.00 |
| No rule features | 17 | 0.07 | 0.77 | 0.85 |
| **No session-encoding** | **15** | **0.06** | **0.76** | **0.81** |

### Final XGBoost confusion matrix (leak-free)

```
              Predicted
            cheat  clean  suspect
True cheat    46     1      8       (recall 84%)
     clean     1   10404   96       (recall 99%)
     suspect  12     67    896      (recall 92%)
```

**46 of 55 cheaters caught with only 1 false positive across 10,500 clean players** — the production-quality framing.

### ROC AUC (binary cheat-vs-not-cheat)

- XGBoost: **0.998**
- Random Forest: 0.990
- Logistic Regression: 0.900

### LLM pipeline demo
Running `ml/detect_and_explain.py` on the full dataset:
- 274 frames above 0.8 cheat probability
- 5 events after debouncing
- **5/5 VERDICT_CHEAT** with cited evidence from telemetry

### LLM hallucination (observed)
Manual inspection revealed Llama occasionally invents numeric values — e.g., reporting "angular velocity exceeds 3000°/sec" when the observed maximum was 813.6°/sec. The verdict was correct but the supporting reasoning contained fabricated numbers. This is a critical lesson in LLM evaluation: even when the output appears authoritative, individual claims need verification. Mitigation strategies (explicit "cite only data values" prompt instructions, lower temperature, post-validation against source data) are documented as v2.1 work.

---

## Limitations

- **Single session per class.** Positional features still encode session identity. Real generalization requires multiple sessions of each class with varied gameplay.
- **Rule-based labeling.** Labels are deterministic functions of measured features, creating circular signal. Session-level labels (whether aimbot was enabled at the session level) would be a stronger ground truth.
- **Aimbot is too "clean."** This project's aimbot sets mouse_dx and mouse_dy to exactly 0; real-world cheats inject noise to evade exactly this signal. Validating against adversarial cheats requires implementing them first.
- **Frame-level random splits.** Adjacent frames are highly correlated; honest evaluation requires session-level cross-validation, which requires multiple sessions per class.
- **LLM hallucination.** Llama occasionally fabricates numeric claims even when reaching correct verdicts.

---

## Future Work

- **Multi-session data collection** (≥5 per class) with varied gameplay to enable group-aware cross-validation and eliminate positional session-encoding leakage.
- **Standalone `detector/CheatDetector.gd`** Godot autoload for real-time in-game inference.
- **ONNX export + Python inference sidecar** for sub-16ms latency real-time inference.
- **Adversarial aimbot** with synthetic mouse jitter and angular smoothing to evade current detectors, then retrain for robustness.
- **LLM post-validation.** Parse cited numeric values from LLM output and verify against source telemetry; flag responses with unverifiable claims.

---

## Lessons Learned

1. **Suspiciously perfect ML results almost always indicate data leakage.** Feature-ablation is a quick, decisive diagnostic.
2. **Feature importance is an evaluation tool, not a "feature ranking."** Where the model looks reveals where it's taking shortcuts.
3. **LLMs add value as explainers, not as primary classifiers** for numeric tabular data. The hybrid architecture (cheap detector triages, expensive LLM explains) is cost-aware and matches production patterns.
4. **Honest documentation of limitations is more impressive than inflated metrics.** A 0.81 F1 with three documented leakage diagnostics is a stronger artifact than 0.99 F1 with no analysis.

---

*v2.0 — Behavioral Anti-Cheat for First-Person Shooters*
