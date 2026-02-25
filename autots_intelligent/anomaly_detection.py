"""
Anomaly detection for forecast vs actuals.
Marks points where residual exceeds 2 * rolling standard deviation.
"""

import numpy as np


def detect_anomalies(actual, predicted) -> list:
    """
    Mark anomalies where |actual - predicted| > 2 * rolling_std.

    rolling_std is computed along the residuals (actual - predicted)
    so the threshold adapts to local variability.

    Args:
        actual: Array-like of actual values (same length as predicted).
        predicted: Array-like of predicted values.

    Returns:
        List of booleans, one per point: True if that point is an anomaly.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)
    if len(predicted) != n:
        raise ValueError("actual and predicted must have the same length")

    residuals = np.abs(actual - predicted)
    # Rolling std of actuals (window 7) for threshold
    window = min(7, n)
    rolling_std = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = actual[start : i + 1]
        if len(chunk) >= 2:
            rolling_std[i] = np.nanstd(chunk)
        else:
            rolling_std[i] = 0.0 if len(chunk) == 0 else np.nan

    # Where rolling_std is NaN or 0, use global std of actuals
    global_std = np.nanstd(actual)
    if np.isnan(global_std) or global_std <= 0:
        global_std = 1e-10
    threshold = 2 * np.where(np.isnan(rolling_std) | (rolling_std <= 0), global_std, rolling_std)

    anomaly = residuals > threshold
    return anomaly.tolist()
