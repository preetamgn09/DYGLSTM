"""Windowing and chronological splits (no leakage; test set frozen)."""
import numpy as np


def make_windows(series, seq_len=12, horizon=12):
    """series [T, F] -> X [n, seq_len, F], Y [n, horizon, F] with NON-overlapping
    windows so test observations are independent (each window used once)."""
    T, F = series.shape
    step = seq_len + horizon
    xs, ys, idx = [], [], []
    for s in range(0, T - step + 1, step):
        xs.append(series[s:s + seq_len])
        ys.append(series[s + seq_len:s + step])
        idx.append(s + seq_len)          # forecast-origin time (for causal adjacency)
    return np.asarray(xs), np.asarray(ys), np.asarray(idx)


def chrono_split(T, frac=(0.7, 0.1, 0.2)):
    a = int(frac[0] * T)
    b = int((frac[0] + frac[1]) * T)
    return a, b   # train_end, val_end


if __name__ == "__main__":
    s = np.arange(600 * 4).reshape(600, 4).astype(float)
    X, Y, idx = make_windows(s, 12, 12)
    print("windows:", X.shape, Y.shape, "| first origin idx:", int(idx[0]),
          "| non-overlap gap:", int(idx[1] - idx[0]))
