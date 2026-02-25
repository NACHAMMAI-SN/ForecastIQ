"""
Feature extraction for time series.
Used to build inputs for the meta-classifier that selects the best AutoTS model.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller


def extract_features(series: pd.Series) -> dict:
    """
    Extract statistical and structural features from a univariate time series.

    Args:
        series: Univariate time series (numeric, index can be datetime or int).

    Returns:
        Dictionary of feature name -> value. Uses np.nan for undefined values.
    """
    # Drop NaN for most stats; keep series as 1d array
    clean = series.dropna()
    if len(clean) < 2:
        return _empty_features()

    x = clean.values.astype(float)
    n = len(x)

    features = {}

    # --- Basic moments ---
    features["mean"] = float(np.mean(x))
    features["variance"] = float(np.var(x))
    if features["variance"] <= 0:
        features["skewness"] = np.nan
        features["kurtosis"] = np.nan
    else:
        features["skewness"] = float(stats.skew(x, nan_policy="omit"))
        features["kurtosis"] = float(stats.kurtosis(x, nan_policy="omit"))

    # --- Rolling stats (window=7, last value) ---
    window = 7
    if n >= window:
        rolling = pd.Series(x).rolling(window=window, min_periods=window)
        features["rolling_mean_7"] = float(rolling.mean().iloc[-1])
        features["rolling_std_7"] = float(rolling.std().iloc[-1])
    else:
        features["rolling_mean_7"] = float(np.mean(x))
        features["rolling_std_7"] = float(np.nan) if n < 2 else float(np.std(x))

    # --- Autocorrelation (lag 1, 7, 12) ---
    for lag in [1, 7, 12]:
        if n > lag and features["variance"] > 0:
            ac = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            features[f"autocorr_lag_{lag}"] = float(ac)
        else:
            features[f"autocorr_lag_{lag}"] = np.nan

    # --- Augmented Dickey-Fuller p-value ---
    try:
        adf_result = adfuller(x, autolag="AIC", maxlag=None)
        features["adf_pvalue"] = float(adf_result[1])
    except Exception:
        features["adf_pvalue"] = np.nan

    # --- Linear regression slope (trend) ---
    try:
        t = np.arange(n, dtype=float)
        slope, _, _, _, _ = stats.linregress(t, x)
        features["linear_trend_slope"] = float(slope)
    except Exception:
        features["linear_trend_slope"] = np.nan

    # --- Simple seasonality strength (variance of seasonal component via rolling diff) ---
    # Use period=7 as default seasonal period; variance of (x - rolling_mean_7) as seasonal component proxy
    if n >= window:
        rolling_mean = pd.Series(x).rolling(window=window, min_periods=1).mean()
        seasonal = x - rolling_mean.values
        features["seasonality_strength"] = float(np.var(seasonal))
    else:
        features["seasonality_strength"] = np.nan

    return features


def _empty_features() -> dict:
    """Return a feature dict with NaN for all keys (used when series too short)."""
    return {
        "mean": np.nan,
        "variance": np.nan,
        "skewness": np.nan,
        "kurtosis": np.nan,
        "rolling_mean_7": np.nan,
        "rolling_std_7": np.nan,
        "autocorr_lag_1": np.nan,
        "autocorr_lag_7": np.nan,
        "autocorr_lag_12": np.nan,
        "adf_pvalue": np.nan,
        "linear_trend_slope": np.nan,
        "seasonality_strength": np.nan,
    }
