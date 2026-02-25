"""
Load CSV datasets (date, value) into a DataFrame with DatetimeIndex.
Handles both standard date,value and stock-style CSVs with header rows.
"""

import os
import pandas as pd


def load_csv_to_df(filepath: str) -> pd.DataFrame:
    """
    Load a CSV with date and value columns into a DataFrame with DatetimeIndex.

    Supports:
    - Standard format: first row is "date,value", then data rows.
    - Stock format: 3 header lines (Price/Close, Ticker/Name, Date/empty), then date,value.

    Args:
        filepath: Path to CSV file.

    Returns:
        DataFrame with DatetimeIndex and a single column (name 'value').
        The column used is the numeric one; the index is the date column.

    Raises:
        ValueError: If file cannot be parsed or has no valid data.
    """
    # Try standard date,value format first (e.g. synthetic_data)
    try:
        df = pd.read_csv(filepath)
        if df.shape[1] < 2 or df.shape[0] == 0:
            raise ValueError("Need at least 2 columns and 1 row")
        col0, col1 = df.columns[0], df.columns[1]
        df = df.rename(columns={col0: "date", col1: "value"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        if len(df) == 0:
            raise ValueError("No valid rows after parsing dates/values")
        df = df.set_index("date").sort_index()
        return df[["value"]]
    except (ValueError, TypeError) as e:
        # Fall through to stock format
        pass

    # Stock format: 3 header lines (Price/Close, Ticker/Name, Date,), then date,value
    try:
        df = pd.read_csv(filepath, skiprows=3, header=None)
        if df.shape[1] < 2 or df.shape[0] == 0:
            raise ValueError(f"No data in {filepath}")
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df = df.set_index("date").sort_index()
        return df[["value"]]
    except Exception as e:
        raise ValueError(f"Could not load {filepath}: {e}") from e
