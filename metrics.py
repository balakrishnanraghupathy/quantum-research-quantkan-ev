"""
metrics.py
==========
Evaluation metrics, Section 4.2, Equations 20-23. All metrics are
computed on [0,1]-normalized SoH values, consistent with the training
loss formulation (Section 3.6).

    Accuracy = 100 * (1 - (1/N) sum_i |y_hat_i - y_i|)                 (20)
    R^2      = 1 - sum_i(y_hat_i - y_i)^2 / sum_i(y_i - y_bar)^2       (21)
    RMSE     = sqrt( (1/N) sum_i (y_hat_i - y_i)^2 )                    (22)
    MAE      = (1/N) sum_i |y_hat_i - y_i|                              (23)

Note (see manuscript's Future Directions, Section 5): mean absolute
percentage error (MAPE) is named as a planned addition to this metric
suite, together with confidence intervals computed across individual
vehicles and cells, but is not part of the results reported in the
current study. No MAPE function is included here for that reason --
adding one prematurely, without recomputing it under the same K=5
rolling-origin protocol for every baseline in Table 18, would break the
fairness protocol described in Section 4.7.
"""

from typing import Dict

import numpy as np


def accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Eq. 20."""
    return 100.0 * (1.0 - np.mean(np.abs(y_hat - y)))


def r_squared(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Eq. 21."""
    y_bar = np.mean(y)
    ss_res = np.sum((y_hat - y) ** 2)
    ss_tot = np.sum((y - y_bar) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rmse(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Eq. 22."""
    return float(np.sqrt(np.mean((y_hat - y) ** 2)))


def mae(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Eq. 23."""
    return float(np.mean(np.abs(y_hat - y)))


def evaluate_all(y_hat: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Convenience wrapper computing all four metrics at once, matching the
    column layout of Tables 15, 16, and 18.
    """
    return {
        "Accuracy (%)": accuracy(y_hat, y),
        "R2": r_squared(y_hat, y),
        "RMSE": rmse(y_hat, y),
        "MAE": mae(y_hat, y),
    }


def macro_average(*metric_dicts: Dict[str, float]) -> Dict[str, float]:
    """
    Unweighted macro-average across multiple test-set metric dicts, used
    to reproduce the headline Abstract numbers (Section 4.5):
        Accuracy = mean(NASA, CALCE, EV-post-adaptation) = 98.63%, etc.
    """
    keys = metric_dicts[0].keys()
    return {k: float(np.mean([d[k] for d in metric_dicts])) for k in keys}
