"""Abilene real traffic-matrix loader.

WHERE TO GET THE DATA (public):
- Yin Zhang's Abilene TM dataset ("Abilene-TM"): files X01..X24, one per week,
  5-minute sampling, 2016 rows/week. Columns are OD-flow volumes (some
  distributions store several measurement columns per flow -- see values_per_flow).
- Or any cleaned OD matrix as CSV / whitespace-txt / .npy shaped [T, F] or
  [T, N*N] (full matrix incl. diagonal) or [T, N*(N-1)] (off-diagonal only).

The loader auto-detects the layout and returns a clean [T, F] float array plus
meta. Graph vertices = OD flows (matching the rebuild's design). Run with
--inspect FIRST to confirm the detected shape before training.

Node-count note: the classic Abilene TM has 12 PoPs (144 OD pairs, 132 off-diag).
The manuscript uses an 11-node / 110-flow variant (one PoP dropped). Use
--drop-node to reproduce that; otherwise the full 12-node graph is used. State
whichever you pick in the paper -- reviewers will check N.
"""
import os, glob
import numpy as np


def _read_matrix(path, values_per_flow=1, value_col=0):
    """Read one file into [T, C]. Concatenate X01..XNN if a directory is given."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "X*")) +
                       glob.glob(os.path.join(path, "*.txt")) +
                       glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No X*/.txt/.csv files in {path}")
        mats = [_read_matrix(f, values_per_flow, value_col) for f in files]
        return np.vstack(mats)
    if path.endswith(".npy"):
        M = np.load(path)
    else:
        with open(path) as fh:
            first = fh.readline()
        delim = "," if first.count(",") > first.count(" ") else None
        # genfromtxt tolerates trailing delimiters / ragged rows (-> NaN columns)
        M = np.genfromtxt(path, delimiter=delim)
        M = np.atleast_2d(M)
        # drop columns that are entirely NaN (e.g. from a trailing comma)
        M = M[:, ~np.all(np.isnan(M), axis=0)]
    M = np.atleast_2d(np.asarray(M, dtype=np.float64))
    if values_per_flow > 1:                       # keep one column per flow
        M = M[:, value_col::values_per_flow]
    return M


def _infer_nodes(F, has_diagonal):
    """Return N such that F == N*N (has_diagonal) or F == N*(N-1)."""
    for N in range(2, 200):
        if has_diagonal and N * N == F:
            return N
        if (not has_diagonal) and N * (N - 1) == F:
            return N
    return None


def load_abilene(path, values_per_flow=1, value_col=0, has_diagonal=None,
                 drop_node=None, min_active_frac=0.5, clip_negative=True, sample_min=5,
                 log_transform=False):
    """Return (TM [T, F_clean], meta).

    has_diagonal: True if columns include self-flows (i==j). If None, we try to
      infer: a perfect square column count is treated as full N*N with diagonal.
    drop_node: optionally remove one node (index) to get the 11-node variant.
    min_active_frac: drop flows whose fraction of non-near-zero samples is below
      this (the standard 'unstable flow' filter). Reported in meta.
    """
    M = _read_matrix(path, values_per_flow, value_col)   # [T, C]
    T, C = M.shape
    if has_diagonal is None:
        n_sq = int(round(C ** 0.5))
        has_diagonal = (n_sq * n_sq == C)
    N = _infer_nodes(C, has_diagonal)
    if N is None:
        raise ValueError(f"Cannot infer node count from {C} columns. "
                         f"Pass values_per_flow / has_diagonal explicitly.")
    # reshape to [T, N, N] if diagonal present, so we can drop self-flows / a node
    if has_diagonal:
        G = M.reshape(T, N, N)
    else:
        # rebuild an N x N with zeros on diagonal
        G = np.zeros((T, N, N))
        off = [(i, j) for i in range(N) for j in range(N) if i != j]
        for k, (i, j) in enumerate(off):
            G[:, i, j] = M[:, k]
    if drop_node is not None:
        keep = [i for i in range(N) if i != drop_node]
        G = G[:, keep][:, :, keep]
        N -= 1
    # flatten off-diagonal OD flows
    off = [(i, j) for i in range(N) for j in range(N) if i != j]
    TM = np.stack([G[:, i, j] for (i, j) in off], axis=1)    # [T, N*(N-1)]
    if clip_negative:
        TM = np.clip(TM, 0, None)
    TM = np.nan_to_num(TM)
    if log_transform:
        # traffic volumes are ~log-normal with large spikes; log1p tames them so
        # z-scored RMSE is not dominated by a few bursts. Standard for TM data.
        TM = np.log1p(TM)
    # filter unstable/dead flows
    active = (np.abs(TM) > 1e-9).mean(axis=0)
    keep = active >= min_active_frac
    dropped = int((~keep).sum())
    TM = TM[:, keep]
    meta = {"N_nodes": N, "F": TM.shape[1], "T": T, "log_transform": log_transform,
            "flows_dropped_unstable": dropped, "has_diagonal": has_diagonal,
            "steps_per_day": int(round(1440 / sample_min))}  # 5-min->288, 15-min->96
    return TM, meta


def inspect(path, **kw):
    TM, meta = load_abilene(path, **kw)
    print("Abilene loaded:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"  TM shape: {TM.shape}")
    print(f"  value range: [{TM.min():.3g}, {TM.max():.3g}]  mean {TM.mean():.3g}")
    print(f"  %near-zero: {(np.abs(TM) < 0.1 * TM.mean()).mean()*100:.1f}")
    return TM, meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--values-per-flow", type=int, default=1)
    ap.add_argument("--value-col", type=int, default=0)
    ap.add_argument("--drop-node", type=int, default=None)
    ap.add_argument("--sample-min", type=int, default=5)
    ap.add_argument("--min-active-frac", type=float, default=0.5)
    ap.add_argument("--log-transform", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    inspect(a.path, values_per_flow=a.values_per_flow, value_col=a.value_col,
            drop_node=a.drop_node, sample_min=a.sample_min, min_active_frac=a.min_active_frac,
            log_transform=a.log_transform)
