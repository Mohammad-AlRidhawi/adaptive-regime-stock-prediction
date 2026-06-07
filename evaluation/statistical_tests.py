"""Statistical tests for model comparison: paired t-test and Diebold-Mariano."""

import numpy as np
from scipy import stats


def paired_t_test(errors_a: np.ndarray, errors_b: np.ndarray) -> dict[str, float]:
    """Two-sided paired t-test on daily squared errors."""
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    t_stat, p_value = stats.ttest_rel(errors_a, errors_b)
    diff = errors_a - errors_b
    cohen_d = float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-12))
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohen_d,
        "n": int(len(errors_a)),
    }


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    horizon: int = 1,
    loss: str = "squared",
) -> dict[str, float]:
    """Diebold-Mariano test of equal predictive accuracy (Newey-West HAC variance)."""
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)

    if loss == "squared":
        d = errors_a ** 2 - errors_b ** 2
    elif loss == "absolute":
        d = np.abs(errors_a) - np.abs(errors_b)
    else:
        raise ValueError(f"Unsupported loss: {loss}")

    n = len(d)
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))

    # Newey-West autocovariance up to lag h - 1
    h = max(horizon, 1)
    for k in range(1, h):
        weight = 1.0 - k / h
        cov = float(np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)))
        var_d += 2.0 * weight * cov

    dm_stat = mean_d / np.sqrt(max(var_d / n, 1e-12))
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "n": int(n),
        "horizon": int(horizon),
        "loss": loss,
    }
