"""
Train a meta-classifier that predicts the best AutoTS model from time-series features.
Reads training_features.csv, trains RandomForestClassifier, saves model and encoder.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_CSV = os.path.join(_SCRIPT_DIR, "training_features.csv")
META_MODEL_PATH = os.path.join(_SCRIPT_DIR, "meta_model.pkl")
LABEL_ENCODER_PATH = os.path.join(_SCRIPT_DIR, "label_encoder.pkl")

# Feature columns (all except best_model)
FEATURE_COLS = [
    "mean", "variance", "skewness", "kurtosis",
    "rolling_mean_7", "rolling_std_7",
    "autocorr_lag_1", "autocorr_lag_7", "autocorr_lag_12",
    "adf_pvalue", "linear_trend_slope", "seasonality_strength",
]


def train_meta_classifier() -> None:
    """
    Load training_features.csv, encode labels, train RandomForestClassifier,
    save meta_model.pkl and label_encoder.pkl. Print accuracy.
    """
    if not os.path.isfile(TRAINING_CSV):
        raise FileNotFoundError(
            f"Training data not found: {TRAINING_CSV}. Run training_pipeline.py first."
        )

    df = pd.read_csv(TRAINING_CSV)

    # Determine feature columns (may have different names if CSV was saved with variations)
    target_col = "best_model"
    if target_col not in df.columns:
        raise ValueError(f"Expected column '{target_col}' in {TRAINING_CSV}")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        raise ValueError("No feature columns found.")

    X = df[feature_cols].copy()
    # Fill NaN with median for robustness
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    y_raw = df[target_col].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Basic accuracy (train)
    train_acc = (clf.predict(X) == y).mean()
    print(f"Train accuracy: {train_acc:.4f}")

    # Cross-validation if enough samples
    if len(y) >= 5:
        cv_scores = cross_val_score(clf, X, y, cv=min(5, len(y)), scoring="accuracy")
        print(f"CV accuracy (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    os.makedirs(os.path.dirname(META_MODEL_PATH) or ".", exist_ok=True)
    with open(META_MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"Saved {META_MODEL_PATH} and {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    train_meta_classifier()
