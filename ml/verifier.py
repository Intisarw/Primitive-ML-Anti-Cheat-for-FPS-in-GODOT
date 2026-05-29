import pandas as pd
df = pd.read_csv("ml/data/processed/combined.csv")
print(df.shape)              # (57653, 19) — 17 original cols + session_id + session_type
print(df["label"].value_counts())
print(df["session_type"].value_counts())
print(df.isna().sum().sum()) # 0