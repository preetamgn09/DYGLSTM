"""Evaluation metrics and significance test. Operate on z-score-normalised units
unless inverse-transformed first (RMSE/MAE reported in z-units per the manuscript)."""
import numpy as np
from scipy.stats import wilcoxon


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def smape(y, yhat):
    return float(np.mean(200.0 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-8)))


def all_metrics(y, yhat):
    return {"RMSE": rmse(y, yhat), "MAE": mae(y, yhat), "sMAPE": smape(y, yhat)}


def paired_wilcoxon(sq_err_a, sq_err_b):
    """Two-sided Wilcoxon signed-rank on per-observation squared errors.
    NOTE: the 'observation' unit must be defined and independent. Pass one value
    per independent test window (aggregated over horizon/flows) rather than every
    element, or overlapping windows will inflate significance. See README."""
    a = np.asarray(sq_err_a).ravel()
    b = np.asarray(sq_err_b).ravel()
    stat, p = wilcoxon(a, b)
    return float(stat), float(p)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.normal(size=(100, 12, 30))
    yhat = y + rng.normal(scale=0.2, size=y.shape)
    print(all_metrics(y, yhat))
    base = y + rng.normal(scale=0.5, size=y.shape)
    # per-window mean squared error as the independent unit
    sa = ((y - yhat) ** 2).mean(axis=(1, 2))
    sb = ((y - base) ** 2).mean(axis=(1, 2))
    print("wilcoxon p:", round(paired_wilcoxon(sa, sb)[1], 6))
