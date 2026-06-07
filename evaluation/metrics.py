"""Evaluation metrics: MAPE, RMSE, DA, Theil's U, CTR (Section 4.2)."""

import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return 100.0 * float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in normalized units."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> float:
    """Fraction of correctly predicted price-movement directions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    return float(np.mean(true_dir == pred_dir))


def theils_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Theil's U1: ratio of forecast RMSE to naive random-walk RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    forecast_rmse = np.sqrt(np.mean((y_true[1:] - y_pred[1:]) ** 2))
    naive_rmse = np.sqrt(np.mean((y_true[1:] - y_true[:-1]) ** 2))
    return float(forecast_rmse / max(naive_rmse, 1e-12))


def ctr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> float:
    """Confidence Tracking Rate: concordance of high confidence with low error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    errors = np.abs(y_true - y_pred)
    median_conf = np.median(confidence)
    median_err = np.median(errors)
    high_conf = confidence > median_conf
    low_err = errors < median_err
    return float(np.mean(high_conf == low_err))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: np.ndarray,
    confidence: np.ndarray | None = None,
) -> dict[str, float]:
    out = {
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "da": directional_accuracy(y_true, y_pred, y_prev) * 100.0,
        "theils_u": theils_u(y_true, y_pred),
    }
    if confidence is not None:
        out["ctr"] = ctr(y_true, y_pred, confidence) * 100.0
    return out
