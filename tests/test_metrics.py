"""Unit tests for the evaluation metrics."""

import numpy as np

from evaluation import compute_all_metrics, directional_accuracy, mape, rmse, theils_u


def test_mape_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert mape(y, y) == 0.0


def test_rmse_basic():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.0, 2.5, 3.0])
    assert abs(rmse(y, yhat) - np.sqrt(0.25 / 3)) < 1e-9


def test_directional_accuracy_all_correct():
    y_prev = np.array([100.0, 100.0])
    y_true = np.array([105.0, 95.0])
    y_pred = np.array([110.0, 90.0])
    assert directional_accuracy(y_true, y_pred, y_prev) == 1.0


def test_theils_u_naive_is_one():
    # If the prediction is the previous value (naive random walk), Theil's U == 1.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    naive = y.copy()
    naive[1:] = y[:-1]
    assert abs(theils_u(y, naive) - 1.0) < 1e-9


def test_compute_all_metrics_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([1.1, 1.9, 3.05, 3.95])
    yprev = np.array([0.9, 2.1, 2.9, 4.1])
    conf = np.array([0.8, 0.2, 0.9, 0.1])
    metrics = compute_all_metrics(y, yhat, yprev, conf)
    assert set(metrics.keys()) == {"mape", "rmse", "da", "theils_u", "ctr"}
