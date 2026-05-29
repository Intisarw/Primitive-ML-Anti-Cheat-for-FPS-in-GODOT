from pathlib import Path
import glob
import pandas as pd


def parse_vector3(s):
    nums = str(s).strip("()").split(",")
    return [float(n.strip()) for n in nums]


def load_data():
    raw_dir = Path(__file__).parent / "data" / "raw"
    csv_files = sorted(glob.glob(str(raw_dir / "*.csv")))

    print("Looking for CSV files in:", raw_dir)
    print("Found files:", csv_files)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for path in csv_files:
        df = pd.read_csv(path)
        filename = Path(path).stem
        df["session_id"] = filename
        df["session_type"] = filename.split("_")[0]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Drop the first row of each session (snap_delta artifact from previous_fov bug)
    combined = combined.groupby("session_id", group_keys=False).apply(
        lambda g: g.iloc[1:]
    )

    processed_dir = Path(__file__).parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "combined.csv"
    combined.to_csv(output_path, index=False)

    print(f"Saved {len(combined):,} rows to {output_path}")
    return combined


if __name__ == "__main__":
    df = load_data()