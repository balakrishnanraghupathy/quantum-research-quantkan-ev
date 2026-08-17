"""
data_pipeline.py
=================
Stage II preprocessing pipeline (Section 3.2): the 6-step procedure
applied identically to all three datasets (NASA, CALCE, EV) before
windowing:

    1. Data cleaning       - remove duplicate and non-physical readings
    2. Missing-value imputation - piecewise cubic interpolation over
                                    short missing-data periods
    3. Outlier rejection   - rolling median-absolute-deviation (MAD) filter
    4. Normalization        - per-dataset Z-scoring (kept separate per
                                dataset to preserve scale differences for
                                the Eq. 19 alignment module, Section 3.8)
    5. Derived feature       - DeltaCapacity health indicator appended,
                                d = 4 raw channels -> d' = 5 features
    6. Windowing             - sliding window of length w (Section 3.10),
                                producing a (w, d') tensor per sample

IMPORTANT (manuscript limitation, see Section 5 Future Directions): the
exact computation of DeltaCapacity for the EV (target) domain -- e.g.
whether derived from BMS-reported SoH or from partial coulomb counting
over observed charge/discharge segments -- is not specified in the
manuscript, and no field-measurement error characterization for this
indicator currently exists. `compute_delta_capacity()` below therefore
exposes an explicit, swappable `method` argument rather than hard-coding
an assumption; users should supply the method that matches their actual
data-acquisition pipeline and are encouraged to report its error
characteristics when they do.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def clean_readings(df: pd.DataFrame, value_cols: list) -> pd.DataFrame:
    """Step 1: remove duplicate timestamps and physically-impossible readings
    (e.g. negative voltage, out-of-range temperature)."""
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    for col in value_cols:
        if col == "voltage":
            df = df[(df[col] > 0) & (df[col] < 6.0)]          # plausible per-cell voltage bounds
        elif col == "temperature":
            df = df[(df[col] > -40) & (df[col] < 100)]        # plausible operating temperature (C)
    return df.reset_index(drop=True)


def impute_missing(df: pd.DataFrame, value_cols: list, max_gap: int = 5) -> pd.DataFrame:
    """Step 2: piecewise cubic interpolation for short gaps (<= max_gap samples)."""
    df = df.copy()
    for col in value_cols:
        series = df[col]
        is_missing = series.isna()
        if is_missing.sum() == 0:
            continue
        valid_idx = np.where(~is_missing)[0]
        if len(valid_idx) < 4:
            df[col] = series.interpolate(limit=max_gap)
            continue
        cs = CubicSpline(valid_idx, series.values[valid_idx], extrapolate=False)
        interpolated = cs(np.arange(len(series)))
        df[col] = np.where(is_missing, interpolated, series.values)
        df[col] = df[col].interpolate(limit=max_gap)  # fallback for any remaining edge gaps
    return df


def reject_outliers(df: pd.DataFrame, value_cols: list, window: int = 15, thresh: float = 4.0) -> pd.DataFrame:
    """Step 3: rolling median-absolute-deviation (MAD) filter."""
    df = df.copy()
    for col in value_cols:
        rolling_median = df[col].rolling(window, center=True, min_periods=1).median()
        mad = (df[col] - rolling_median).abs().rolling(window, center=True, min_periods=1).median()
        mad_scaled = 1.4826 * mad  # consistent estimator for normal-distributed data
        z = (df[col] - rolling_median).abs() / (mad_scaled + 1e-8)
        df.loc[z > thresh, col] = np.nan
    return impute_missing(df, value_cols)  # re-impute any newly-created gaps


def zscore_normalize(df: pd.DataFrame, value_cols: list) -> (pd.DataFrame, dict):
    """
    Step 4: per-DATASET Z-scoring (i.e. fit mean/std once per dataset --
    NASA, CALCE, or EV -- not globally), so that scale differences
    between datasets are preserved for the Eq. 19 alignment module.

    Returns the normalized dataframe and the {mean, std} used, so the
    same statistics can be reused for Eq. 19's (mu, sigma) terms.
    """
    df = df.copy()
    stats = {}
    for col in value_cols:
        mu, sigma = df[col].mean(), df[col].std() + 1e-8
        df[col] = (df[col] - mu) / sigma
        stats[col] = {"mean": mu, "std": sigma}
    return df, stats


def compute_delta_capacity(
    df: pd.DataFrame,
    capacity_col: str,
    method: str = "diff",
) -> pd.Series:
    """
    Step 5: computes the derived DeltaCapacity health indicator.

    Parameters
    ----------
    method : {"diff", "bms_soh", "partial_coulomb_count"}
        "diff"                    -- cycle-to-cycle difference of a directly
                                      measured/reference capacity column
                                      (appropriate for NASA/CALCE lab data,
                                      where full discharge capacity is
                                      directly measured each cycle).
        "bms_soh"                 -- derived from a BMS-reported SoH
                                      column; caller must characterize the
                                      noise of this signal separately
                                      (see module docstring / Section 5).
        "partial_coulomb_count"   -- derived from integrating current over
                                      observed (possibly partial)
                                      charge/discharge segments; caller
                                      must supply a segmented coulomb-count
                                      column and characterize its error.

    NOTE: this manuscript does not specify which method was used for the
    EV (target) domain -- see module docstring. Callers targeting the EV
    domain should explicitly choose "bms_soh" or "partial_coulomb_count"
    (as appropriate to their acquisition pipeline) rather than relying on
    the "diff" default, which assumes a directly-measured reference
    capacity of the kind only available in controlled lab cycling.
    """
    if method == "diff":
        return df[capacity_col].diff().fillna(0.0)
    elif method in ("bms_soh", "partial_coulomb_count"):
        raise NotImplementedError(
            f"'{method}' requires a pipeline-specific implementation supplying "
            f"the appropriate source column(s) and error characterization; "
            f"this is intentionally left unimplemented pending the "
            f"characterization identified in the manuscript's Future "
            f"Directions (Section 5)."
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def make_sliding_windows(
    features: np.ndarray, window_length: int, stride: int = 1
) -> np.ndarray:
    """
    Step 6: segments a (T, d') feature array into overlapping sliding
    windows of length `window_length`, producing an (N, w, d') tensor,
    consistent with Section 3.2's "output is a w x d' Tensor per sample."
    """
    T, d = features.shape
    n_windows = (T - window_length) // stride + 1
    if n_windows <= 0:
        raise ValueError("Sequence shorter than window_length.")
    windows = np.stack(
        [features[i : i + window_length] for i in range(0, n_windows * stride, stride)]
    )
    return windows


def preprocess_pipeline(
    df: pd.DataFrame,
    value_cols: list,
    capacity_col: str,
    window_length: int,
    delta_capacity_method: str = "diff",
) -> np.ndarray:
    """
    Runs the full 6-step Stage II pipeline end-to-end on one dataset's
    raw dataframe and returns the final (N, w, d') windowed tensor ready
    for the QuantKANEV.backbone() input (after appending DeltaCapacity,
    d' = len(value_cols) + 1).
    """
    df = clean_readings(df, value_cols)
    df = impute_missing(df, value_cols)
    df = reject_outliers(df, value_cols)
    df, _stats = zscore_normalize(df, value_cols)
    df["delta_capacity"] = compute_delta_capacity(df, capacity_col, method=delta_capacity_method)

    feature_cols = value_cols + ["delta_capacity"]
    features = df[feature_cols].to_numpy(dtype=np.float32)
    windows = make_sliding_windows(features, window_length)
    return windows
