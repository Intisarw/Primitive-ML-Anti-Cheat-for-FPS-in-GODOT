import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features..."""
    df = df.copy()

    WINDOW = 30

    # rolling mouse jitter
    df["mouse_dx_std30"] = (
        df.groupby("session_id")["mouse_dx"]
          .transform(lambda x: x.rolling(WINDOW, min_periods=1).std())
    )
    df["mouse_dy_std30"] = (
        df.groupby("session_id")["mouse_dy"]
          .transform(lambda x: x.rolling(WINDOW, min_periods=1).std())
    )

    # angular velocity
    df["dt"] = df.groupby("session_id")["timestamp"].diff().fillna(0.0)
    df["angular_velocity"] = np.where(df["dt"] > 0, df["snap_delta"] / df["dt"], 0.0)

    # distance to enemy
    df["dist_to_enemy"] = np.sqrt(
        (df["player_x"] - df["enemy_x"]) ** 2
        + (df["player_y"] - df["enemy_y"]) ** 2
        + (df["player_z"] - df["enemy_z"]) ** 2
    )

    # fov rate of change
    df["fov_rate"] = (
        df.groupby("session_id")["fov_to_target"]
          .diff()
          .fillna(0.0)
          .abs()
    )
    df["mouse_dx_std30"] = (
    df.groupby("session_id")["mouse_dx"]
      .transform(lambda x: x.rolling(WINDOW, min_periods=1).std())
      .fillna(0.0)
    )
       
    df["mouse_dy_std30"] = (
    df.groupby("session_id")["mouse_dy"]
      .transform(lambda x: x.rolling(WINDOW, min_periods=1).std())
      .fillna(0.0)
    )
    

    return df


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    df = add_features(df)
    print(df.shape)
    print(df.columns.tolist())
    print(df.head())