"""ml/explainer.py — LLM-based explainer for flagged cheat events."""

import pandas as pd
import requests

from data_loader import load_data
from features import add_features


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"


def format_telemetry_stats(window: pd.DataFrame) -> str:
    """Compress a telemetry window into compact stats for the LLM prompt."""
    n_frames = len(window)
    duration = window["timestamp"].max() - window["timestamp"].min()

    lines = [
        f"Window: {n_frames} frames ({duration:.2f} seconds)",
        "",
        "AIM BEHAVIOR:",
        f"- snap_delta range: {window['snap_delta'].min():.2f}° to {window['snap_delta'].max():.2f}° (mean: {window['snap_delta'].mean():.2f}°)",
        f"- max angular_velocity: {window['angular_velocity'].max():.1f}°/sec",
        f"- mouse_dx rolling jitter (std): {window['mouse_dx_std30'].mean():.4f}",
        f"- frames with mouse_dx == 0 exactly: {(window['mouse_dx'] == 0.0).sum()}/{n_frames}",
        "",
        "TARGETING:",
        f"- min angle to target: {window['fov_to_target'].min():.2f}°",
        f"- distance to enemy: {window['dist_to_enemy'].min():.2f}m to {window['dist_to_enemy'].max():.2f}m",
        "",
        "ACTIONS:",
        f"- shots fired: {int(window['is_firing'].sum())}",
        f"- enemies killed: {int(window['enemy_killed'].sum())}",
    ]
    return "\n".join(lines)

def build_prompt(window: pd.DataFrame, flag_time=None) -> str:
    """Build the full prompt to send to the LLM."""
    flag_str = f" at t={flag_time:.2f}s" if flag_time is not None else ""
    stats = format_telemetry_stats(window)

    return f"""You are an FPS anti-cheat reviewer analyzing player telemetry for evidence of aimbot use.

A player was flagged{flag_str}. The following statistics summarize their behavior in the telemetry window leading up to the flag:

{stats}

For reference: human players typically show mouse_dx jitter (rolling std) around 0.05 and angular velocities under 3000°/sec. Aimbots produce zero mouse jitter and angular velocities exceeding 10,000°/sec because they snap the camera directly without mouse input.

Write a brief (3-4 sentence) explanation of whether this looks like cheating, citing specific values from the data. End your response with exactly one of these on its own line:
VERDICT_CHEAT
VERDICT_CLEAN
VERDICT_INCONCLUSIVE"""

def explain_flag(window: pd.DataFrame, flag_time=None, model: str = MODEL_NAME) -> str:
    """Call Ollama to generate a moderator-readable explanation of a flag."""
    prompt = build_prompt(window, flag_time)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"]


if __name__ == "__main__":
    print("Loading data and computing features...")
    df = add_features(load_data())

    # Pick a real cheat-labeled moment to demo on
    cheat_rows = df[df["label"] == "cheat"]
    if cheat_rows.empty:
        print("No cheat-labeled rows found in dataset.")
        exit(1)

    flag_row = cheat_rows.sample(1, random_state=42).iloc[0]
    session_id = flag_row["session_id"]

    # Grab the 300 frames (~5 sec at 60 fps) up to and including the flagged frame
    session_df = df[df["session_id"] == session_id].reset_index(drop=True)
    flag_pos = session_df.index[session_df["timestamp"] == flag_row["timestamp"]][0]
    window_start = max(0, flag_pos - 299)
    window = session_df.iloc[window_start:flag_pos + 1]

    print(f"Flag picked: session={session_id}, t={flag_row['timestamp']:.2f}s")
    print(f"Window: {len(window)} frames")
    print()
    print("=== PROMPT SENT TO OLLAMA ===")
    print(build_prompt(window, flag_time=flag_row["timestamp"]))
    print()
    print("=== CALLING OLLAMA (first call may take 30-60 sec while model warms up) ===")

    explanation = explain_flag(window, flag_time=flag_row["timestamp"])

    print()
    print("=== LLM EXPLANATION ===")
    print(explanation)