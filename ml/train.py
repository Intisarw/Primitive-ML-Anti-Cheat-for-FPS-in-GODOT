"""ml/train.py — train and save baseline cheat detection models."""

import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from data_loader import load_data
from features import add_features


# Numeric features the models will train on. Note exclusions:
# - timestamp (sequence number — would just memorize order)
# - time_to_kill (always -1 in current data, no signal)
# - session_id, session_type, label (metadata / target)
FEATURE_COLS = [
    "player_x", "player_y", "player_z",
    "enemy_x", "enemy_y", "enemy_z",
    "aim_yaw", "aim_pitch",
    #"mouse_dx", "mouse_dy",
    #"fov_to_target", 
    "snap_delta",
    #"is_firing", "enemy_killed",
    "mouse_dx_std30", "mouse_dy_std30",
    "dt", "angular_velocity",
    "dist_to_enemy", "fov_rate",
]

# Why explicit FEATURE COL? Why not df.select("number").columns, but listing by hand forces to think about each one and prevents leaks. Its self documenting as well

RANDOM_STATE = 42
MODELS_DIR = Path(__file__).parent / "models"

def build_models():
    """Return a dict of named sklearn-compatible Pipelines."""
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "xgb": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="mlogloss",
            )),
        ]),
    }


def train_and_evaluate():
    # Load and feature-engineer
    print("Loading data and computing features...")
    df = add_features(load_data())
    print(f"Total rows: {len(df):,}  |  Features: {len(FEATURE_COLS)}")

    X = df[FEATURE_COLS]
    y = df["label"]

    # Encode labels (xgboost requires numeric)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Classes: {dict(enumerate(le.classes_))}")  # e.g. {0: 'cheat', 1: 'clean', 2: 'suspect'}

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")

    # Train each model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = build_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == "xgb":
            # XGBoost needs sample weights for class balancing
            weights = compute_sample_weight("balanced", y_train)
            model.fit(X_train, y_train, clf__sample_weight=weights)
        else:
            model.fit(X_train, y_train)

        # Evaluate on test
        y_pred = model.predict(X_test)
        cheat_idx = list(le.classes_).index("cheat")

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "cheat_recall": recall_score(y_test, y_pred, labels=[cheat_idx], average="macro", zero_division=0),
            "cheat_f1": f1_score(y_test, y_pred, labels=[cheat_idx], average="macro", zero_division=0),
        }

        # Save the trained pipeline
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
        print(f"  Saved to {MODELS_DIR / f'{name}.pkl'}")

    # Save label encoder so evaluate.py can decode predictions
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    return results, le

def print_comparison(results):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON ON TEST SET")
    print("=" * 60)
    print(f"{'Model':<12} {'Accuracy':<12} {'Cheat Recall':<16} {'Cheat F1':<10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<12} {r['accuracy']:<12.4f} {r['cheat_recall']:<16.4f} {r['cheat_f1']:<10.4f}")


if __name__ == "__main__":
    results, le = train_and_evaluate()
    print_comparison(results)