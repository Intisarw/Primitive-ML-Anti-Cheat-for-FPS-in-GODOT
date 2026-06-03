"""ml/detect_and_explain.py — full v2 pipeline: detector flags, LLM explains."""

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from data_loader import load_data
from features import add_features
from train import FEATURE_COLS
from explainer import explain_flag


MODELS_DIR = Path(__file__).parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

CHEAT_THRESHOLD = 0.8       # probability above which we flag for explanation
DEBOUNCE_SECONDS = 5.0       # don't re-flag the same session within this window
WINDOW_FRAMES = 300          # context window for the LLM (~5 sec at 60 fps)
MAX_FLAGS = 5                # demo cap — LLM calls take ~5 sec each

def load_detector():
    model = joblib.load(MODELS_DIR / "xgb.pkl")
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    return model, le

def run_pipeline(df, model, le,
                 threshold=CHEAT_THRESHOLD,
                 debounce=DEBOUNCE_SECONDS,
                 window_frames=WINDOW_FRAMES,
                 max_flags=MAX_FLAGS):
    """Run detector → debounce → explainer on the full DataFrame."""

    cheat_idx = list(le.classes_).index("cheat")

    print(f"Running detector on {len(df):,} rows...")
    probas = model.predict_proba(df[FEATURE_COLS])[:, cheat_idx]
    df = df.copy()
    df["cheat_proba"] = probas

    flagged = df[df["cheat_proba"] >= threshold]
    print(f"Found {len(flagged):,} frames with cheat probability >= {threshold}")
    print(f"After debouncing (max one flag per {debounce}s per session), "
          f"calling LLM on up to {max_flags} of them...\n")

    events = []
    last_flag_time = {}   # session_id -> last flag timestamp

    for _, row in flagged.iterrows():
        session = row["session_id"]
        t = row["timestamp"]

        if t - last_flag_time.get(session, -float("inf")) < debounce:
            continue
        last_flag_time[session] = t

        # Pull 300-frame window leading up to and including the flag
        session_df = df[df["session_id"] == session].reset_index(drop=True)
        flag_pos = session_df.index[session_df["timestamp"] == t][0]
        window_start = max(0, flag_pos - window_frames + 1)
        window = session_df.iloc[window_start:flag_pos + 1]

        print(f"[{len(events) + 1}] Flag in {session} at t={t:.2f}s "
              f"(proba={row['cheat_proba']:.3f}) — calling LLM...")

        explanation = explain_flag(window, flag_time=t)

        events.append({
            "session_id": session,
            "flag_time": float(t),
            "cheat_proba": float(row["cheat_proba"]),
            "window_frames": int(len(window)),
            "explanation": explanation,
        })

        # Print verdict line inline for visibility
        verdict = next(
            (line for line in explanation.split("\n") if line.startswith("VERDICT_")),
            "VERDICT_NOT_FOUND"
        )
        print(f"    → {verdict}\n")

        if len(events) >= max_flags:
            print(f"Reached max_flags={max_flags}, stopping.\n")
            break

    return events

def save_events(events, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_events": len(events),
            "events": events,
        }, f, indent=2)
    print(f"Saved {len(events)} flagged events to {output_path}")


if __name__ == "__main__":
    print("Loading data and features...")
    df = add_features(load_data())

    print("Loading XGBoost detector...")
    model, le = load_detector()

    events = run_pipeline(df, model, le)

    output_path = REPORTS_DIR / "flagged_events.json"
    save_events(events, output_path)

    print("\n=== MODERATOR QUEUE SUMMARY ===")
    for i, e in enumerate(events, 1):
        verdict = next(
            (line for line in e["explanation"].split("\n") if line.startswith("VERDICT_")),
            "VERDICT_NOT_FOUND"
        )
        print(f"  [{i}] {e['session_id']} t={e['flag_time']:.2f}s "
              f"proba={e['cheat_proba']:.3f} → {verdict}")