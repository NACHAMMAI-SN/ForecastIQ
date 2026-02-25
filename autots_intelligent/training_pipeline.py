"""
Training pipeline: for each CSV in data/stock_data and data/synthetic_data,
extract features, run AutoTS to get the best model, and save a training dataset
for the meta-classifier.
"""

import os
import sys
import traceback
import pandas as pd

# Add AutoTS package root so we can import autots (from the unmodified library)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTOTS_PKG_ROOT = os.path.dirname(_SCRIPT_DIR)  # AutoTS/ folder containing autots/
if _AUTOTS_PKG_ROOT not in sys.path:
    sys.path.insert(0, _AUTOTS_PKG_ROOT)

from autots import AutoTS

from feature_extraction import extract_features
from data_loader import load_csv_to_df


# Paths relative to autots_intelligent/
DATA_STOCK = os.path.join(_SCRIPT_DIR, "data", "stock_data")
DATA_SYNTHETIC = os.path.join(_SCRIPT_DIR, "data", "synthetic_data")
OUTPUT_CSV = os.path.join(_SCRIPT_DIR, "training_features.csv")

FORECAST_LENGTH = 30
FREQUENCY = "infer"
ENSEMBLE = None


def _collect_csv_paths() -> list:
    """Return list of all CSV paths under data/stock_data and data/synthetic_data."""
    paths = []
    for folder in (DATA_STOCK, DATA_SYNTHETIC):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(".csv"):
                paths.append(os.path.join(folder, name))
    return paths


def _process_one(path: str) -> dict | None:
    """
    Load one CSV, extract features, run AutoTS, return one row dict:
    { feature_1: v1, ..., best_model: name } or None on failure.
    """
    try:
        df = load_csv_to_df(path)
    except Exception as e:
        print(f"[SKIP] {path}: load failed: {e}")
        return None

    if df.empty or len(df) < 50:
        print(f"[SKIP] {path}: too few rows ({len(df)})")
        return None

    # Single series for feature extraction and AutoTS
    series = df["value"]
    try:
        features = extract_features(series)
    except Exception as e:
        print(f"[SKIP] {path}: feature extraction failed: {e}")
        return None

    # AutoTS: broader model search via superfast list; forecast_length=30, frequency="infer", ensemble=None
    try:
        model = AutoTS(
            forecast_length=FORECAST_LENGTH,
            frequency=FREQUENCY,
            ensemble=ENSEMBLE,
            model_list="superfast",
            max_generations=2,
            num_validations=1,
            verbose=0,
        )
        model.fit(df)
        best_model = model.best_model_name
    except Exception as e:
        print(f"[SKIP] {path}: AutoTS failed: {e}")
        traceback.print_exc()
        return None

    row = dict(features)
    row["best_model"] = best_model
    return row


def run_training_pipeline() -> None:
    """
    Process all CSVs in data/stock_data and data/synthetic_data;
    build training dataset and save to training_features.csv.
    """
    paths = _collect_csv_paths()
    if not paths:
        print("No CSV files found under data/stock_data or data/synthetic_data.")
        return

    rows = []
    for path in paths:
        row = _process_one(path)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No datasets could be processed. Cannot create training_features.csv.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out_df)} rows to {OUTPUT_CSV}")
    print("Model diversity (best_model unique):", sorted(out_df["best_model"].unique().tolist()))


if __name__ == "__main__":
    run_training_pipeline()
