"""ml/evaluate.py — generate evaluation plots and metrics for trained models."""

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split

from data_loader import load_data
from features import add_features
from train import FEATURE_COLS, RANDOM_STATE


MODELS_DIR = Path(__file__).parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"
MODEL_NAMES = ["logreg", "rf", "xgb"]


def load_test_set():
    """Rebuild the exact same test set the models were trained against."""
    df = add_features(load_data())
    X = df[FEATURE_COLS]
    y = df["label"]

    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    y_enc = le.transform(y)

    _, X_test, _, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
    )
    return X_test, y_test, le

def plot_confusion_matrix(model, X_test, y_test, le, name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    out_path = REPORTS_DIR / f"cm_{name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")


def print_classification_reports(models, X_test, y_test, le):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

def plot_roc_overlay(models, X_test, y_test, le):
    cheat_idx = list(le.classes_).index("cheat")
    y_test_binary = (y_test == cheat_idx).astype(int)

    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        y_score = model.predict_proba(X_test)[:, cheat_idx]
        fpr, tpr, _ = roc_curve(y_test_binary, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Cheat vs Not-Cheat (binary)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out_path = REPORTS_DIR / "roc_overlay.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")

def plot_feature_importance(model, name):
    clf = model.named_steps["clf"]   # unwrap the Pipeline to get the classifier

    if not hasattr(clf, "feature_importances_"):
        print(f"  {name} has no feature_importances_, skipping")
        return

    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]   # descending

    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_features = [FEATURE_COLS[i] for i in order][::-1]
    sorted_importances = importances[order][::-1]
    ax.barh(sorted_features, sorted_importances, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"{name} — Feature Importance")
    plt.tight_layout()
    out_path = REPORTS_DIR / f"importance_{name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")


def save_summary_csv(models, X_test, y_test, le):
    cheat_idx = list(le.classes_).index("cheat")
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "cheat_precision": precision_score(y_test, y_pred, labels=[cheat_idx], average="macro", zero_division=0),
            "cheat_recall": recall_score(y_test, y_pred, labels=[cheat_idx], average="macro", zero_division=0),
            "cheat_f1": f1_score(y_test, y_pred, labels=[cheat_idx], average="macro", zero_division=0),
        })

    summary = pd.DataFrame(rows)
    out_path = Path(__file__).parent.parent / "reports" / "model_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    return summary


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test set and models...")
    X_test, y_test, le = load_test_set()
    print(f"Test set: {X_test.shape[0]:,} rows")

    models = {name: joblib.load(MODELS_DIR / f"{name}.pkl") for name in MODEL_NAMES}

    print("\n=== Confusion matrices ===")
    for name, model in models.items():
        plot_confusion_matrix(model, X_test, y_test, le, name)

    print("\n=== Classification reports ===")
    print_classification_reports(models, X_test, y_test, le)

    print("\n=== ROC overlay ===")
    plot_roc_overlay(models, X_test, y_test, le)

    print("\n=== Feature importance (tree models only) ===")
    for name in ["rf", "xgb"]:
        plot_feature_importance(models[name], name)

    print("\n=== Summary ===")
    summary = save_summary_csv(models, X_test, y_test, le)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()