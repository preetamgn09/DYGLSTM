"""Per-time-step dynamic adjacency via distance correlation.

Graph vertices = OD FLOWS (design decision, resolving the manuscript ambiguity):
so the adjacency has shape [F, F] where F is the number of directed OD flows.
If you switch to a router-level graph later, change only what feeds this module
(cluster/aggregate flows to routers) and the shape becomes [N_routers, N_routers].

Pipeline per (cached) step t:
  1) distance correlation over the causal window [t-W, t) of de-noised series
  2) top-K sparsification per row
  3) symmetric degree normalisation  D^-1/2 (A+A^T)/2 D^-1/2
  4) flat-hold cache for p steps; static train adjacency during warm-start (t<W)

Causality: only observations strictly before the forecast point enter the window.
"""
import numpy as np


def _double_center(D):
    rm = D.mean(axis=1, keepdims=True)
    cm = D.mean(axis=0, keepdims=True)
    return D - rm - cm + D.mean()


def distance_correlation_matrix(win):
    """dCor between every pair of columns of `win` ([W, F]). Returns [F, F] in [0,1].
    Vectorised over pairs via precomputed double-centred distance tensors."""
    W, F = win.shape
    # per-flow double-centred pairwise-distance matrices: A[f] is [W, W]
    A = np.empty((F, W, W))
    for f in range(F):
        d = np.abs(win[:, f][:, None] - win[:, f][None, :])
        A[f] = _double_center(d)
    # dCov^2(f,g) = mean(A[f]*A[g]); dVar(f) = mean(A[f]*A[f])
    Aflat = A.reshape(F, -1)                       # [F, W*W]
    dcov2 = (Aflat @ Aflat.T) / (W * W)            # [F, F]
    dvar = np.clip(np.diag(dcov2), 1e-12, None)    # [F]
    denom = np.sqrt(np.sqrt(np.outer(dvar, dvar)))
    dcor = np.sqrt(np.clip(dcov2, 0, None)) / (denom + 1e-12)
    return np.clip(dcor, 0.0, 1.0)


def topk_sparsify(A, k):
    F = A.shape[0]
    out = np.zeros_like(A)
    k = min(k, F - 1)
    for i in range(F):
        row = A[i].copy()
        row[i] = -np.inf                    # exclude self
        idx = np.argpartition(row, -k)[-k:]
        out[i, idx] = A[i, idx]
    return out


def sym_normalize(A):
    """Â = D^-1/2 (A+A^T)/2 D^-1/2 on the symmetrised matrix."""
    S = (A + A.T) / 2.0
    deg = S.sum(axis=1)
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    return (dinv[:, None] * S) * dinv[None, :]


def build_adjacency(win, k=6):
    return sym_normalize(topk_sparsify(distance_correlation_matrix(win), k))


class DynamicAdjacency:
    """Causal, cached per-step adjacency provider."""
    def __init__(self, series, W=48, k=6, p=4, static_train=None):
        self.x = series          # [T, F] de-noised
        self.W, self.k, self.p = W, k, p
        self.static = static_train
        self._cache, self._cache_t = None, -10**9

    def at(self, t):
        """Adjacency to use when forecasting at time t (uses window strictly < t)."""
        if t < self.W:
            return self.static
        if self._cache is not None and (t - self._cache_t) < self.p:
            return self._cache
        win = self.x[t - self.W:t]
        self._cache = build_adjacency(win, self.k)
        self._cache_t = t
        return self._cache


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W = 48
    x = np.cumsum(rng.normal(size=(W,)))          # a random walk
    win = np.stack([x, x, -x, rng.normal(size=W)], axis=1)  # f0=f1 identical, f2=-f0, f3 indep
    D = distance_correlation_matrix(win)
    print("dCor(identical f0,f1) =", round(D[0, 1], 3), "(expect ~1.0)")
    print("dCor(f0, -f0)         =", round(D[0, 2], 3), "(expect ~1.0; dCor sees dependence)")
    print("dCor(f0, independent) =", round(D[0, 3], 3), "(expect small)")
    A = build_adjacency(win, k=2)
    ev = np.linalg.eigvalsh(A)
    print("symmetric:", np.allclose(A, A.T), "| spectral radius:", round(float(np.abs(ev).max()), 3),
          "(<=1 for stability)")
