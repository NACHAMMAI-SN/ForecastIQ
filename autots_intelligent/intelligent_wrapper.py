"""
Intelligent forecasting: use meta-classifier to select model, then run AutoTS
restricted to that model and return forecast and MAPE (when available).
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTOTS_PKG_ROOT = os.path.dirname(_SCRIPT_DIR)  # AutoTS/ folder containing autots/
if _AUTOTS_PKG_ROOT not in sys.path:
    sys.path.insert(0, _AUTOTS_PKG_ROOT)

from autots import AutoTS

from feature_extraction import extract_features
from meta_classifier import FEATURE_COLS

META_MODEL_PATH = os.path.join(_SCRIPT_DIR, "meta_model.pkl")
LABEL_ENCODER_PATH = os.path.join(_SCRIPT_DIR, "label_encoder.pkl")
FORECAST_LENGTH = 30


def _ensure_value_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has datetime index and a single column named 'value'."""
    if isinstance(df.index, pd.DatetimeIndex) and "value" in df.columns:
        return df[["value"]].copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.shape[1] == 1:
        df = df.copy()
        df.columns = ["value"]
        return df
    raise ValueError("df must have DatetimeIndex and one value column.")


def predict_with_intelligence(df: pd.DataFrame) -> dict:
    """
    Extract features, predict best model via meta-classifier, run AutoTS with that
    model only, generate forecast, and compute MAPE when possible.

    Args:
        df: DataFrame with DatetimeIndex and one value column (e.g. 'value').

    Returns:
        {
            "selected_model": str,
            "mape": float | None,
            "forecast": list of floats,
        }
    """
    df = _ensure_value_column(df)
    series = df["value"]

    # Load meta-model and predict which model to use
    if not os.path.isfile(META_MODEL_PATH) or not os.path.isfile(LABEL_ENCODER_PATH):
        raise FileNotFoundError(
            "meta_model.pkl and/or label_encoder.pkl not found. "
            "Run training_pipeline.py then meta_classifier.py first."
        )
    with open(META_MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    features = extract_features(series)
    feature_cols = [c for c in FEATURE_COLS if c in features]
    X = pd.DataFrame([features])[feature_cols]
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    pred_label = clf.predict(X)[0]
    selected_model = le.inverse_transform([pred_label])[0]

    # Run AutoTS restricted to the selected model
    model = AutoTS(
        forecast_length=FORECAST_LENGTH,
        frequency="infer",
        ensemble=None,
        model_list=[selected_model],
        max_generations=1,
        num_validations=0,
        verbose=0,
    )
    model.fit(df)
    result = model.predict(forecast_length=FORECAST_LENGTH, just_point_forecast=True)

    # result is DataFrame (one column for one series); convert to list
    if isinstance(result, pd.DataFrame):
        forecast = result.iloc[:, 0].tolist()
    else:
        forecast = result.tolist() if hasattr(result, "tolist") else list(result)

    # MAPE: we don't have future actuals; optionally use validation score if available
    mape = None
    try:
        res = model.results()
        if res is not None and not isinstance(res, str) and "Score" in res.columns:
            best_row = res.iloc[0]
            # Score might be SMAPE or similar; use as proxy for MAPE if numeric
            score = best_row.get("Score", np.nan)
            if pd.notna(score) and isinstance(score, (int, float)):
                mape = float(score)
    except Exception:
        pass

    return {
        "selected_model": str(selected_model),
        "mape": mape,
        "forecast": forecast,
    }
