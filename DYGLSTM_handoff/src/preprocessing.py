"""FCM-guided weighted exponential smoothing (WES) preprocessing.

Order (causal; train-split statistics only):
  1) z-score per flow using TRAIN mean/std
  2) Fuzzy C-Means over flows -> soft membership U [F, C]
  3) per-flow WES de-noising with alpha_i tied to cluster-assignment strength
  4) keep (mu, sigma) for inverse transform at evaluation

IMPORTANT (corrects a claim in the manuscript):
Standard exponential smoothing is  s_t = alpha * x_t + (1-alpha) * s_{t-1}.
Here a HIGHER alpha puts MORE weight on the new observation, i.e. LESS smoothing.
The manuscript's prose said higher alpha => "more aggressive" smoothing, which is
the opposite of this recurrence. We keep the standard recurrence and document it;
if you intend higher-confidence clusters to be smoothed *harder*, invert the map
(use alpha_i = alpha0 * (1 - 0.5*max_membership)) and update the paper text.
"""
import numpy as np


def zscore_fit(train_x):
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0) + 1e-8
    return mu, sigma


def zscore_apply(x, mu, sigma):
    return (x - mu) / sigma


def zscore_invert(z, mu, sigma):
    return z * sigma + mu


def _kpp_init(X, c, rng):
    """k-means++ style seeding: pick c well-spread data points as initial centers.
    Breaks the symmetry that makes random-membership FCM collapse to uniform."""
    F = X.shape[0]
    idx = [int(rng.integers(F))]
    d2 = np.full(F, np.inf)
    for _ in range(1, c):
        diff = X - X[idx[-1]]
        d2 = np.minimum(d2, np.einsum("ij,ij->i", diff, diff))
        probs = d2 / (d2.sum() + 1e-12)
        idx.append(int(rng.choice(F, p=probs)))
    return X[idx].copy()   # [c, T]


def fuzzy_cmeans(X, c=4, m=2.0, tol=1e-6, max_iter=300, seed=42):
    """FCM over rows of X ([F, T]): each flow is a point. Returns U [F, c].
    Centers are seeded from spread-out data points (k-means++), then the standard
    FCM alternation runs to convergence.
    """
    rng = np.random.default_rng(seed)
    centers = _kpp_init(X, c, rng)                                 # [c, T]
    U = None
    for _ in range(max_iter):
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2) + 1e-12
        inv = dist ** (-2.0 / (m - 1))
        U_new = inv / inv.sum(axis=1, keepdims=True)               # [F, c]
        if U is not None and np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new
        Um = U ** m
        centers = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-12)   # [c, T]
    return U


def wes_denoise(z, alpha):
    """Weighted exponential smoothing per flow.
    z: [T, F] z-scored series; alpha: [F] per-flow coefficients in (0,1].
    s_t = alpha*x_t + (1-alpha)*s_{t-1}. Higher alpha => less smoothing.
    """
    T, F = z.shape
    s = np.empty_like(z)
    s[0] = z[0]
    for t in range(1, T):
        s[t] = alpha * z[t] + (1.0 - alpha) * s[t - 1]
    return s


def flow_features(raw_train, steps_per_day):
    """Per-flow interpretable shape features for FCM (robust to per-step noise;
    Euclidean distance on the raw high-dimensional series degenerates). Features:
    log burstiness (peak/mean), diurnal energy (daily-profile var / total var),
    and lag-1 autocorrelation. Standardised across flows. Returns [F, 3].
    WES de-noising below still runs on the full series."""
    T, F = raw_train.shape
    d = steps_per_day
    nd = (T // d) * d
    feats = np.zeros((F, 3))
    for f in range(F):
        s = raw_train[:, f]
        mu = s.mean() + 1e-9
        burst = np.log1p(s.max() / mu)
        prof = s[:nd].reshape(-1, d).mean(axis=0)
        diurnal = prof.var() / (s.var() + 1e-9)
        ac1 = np.corrcoef(s[:-1], s[1:])[0, 1] if s.std() > 0 else 0.0
        feats[f] = [burst, diurnal, ac1]
    feats = np.nan_to_num(feats)
    return (feats - feats.mean(0)) / (feats.std(0) + 1e-9)


def preprocess(raw, train_end, c=4, alpha0=0.3, seed=42, steps_per_day=96):
    """Full pipeline. raw: [T, F]. train_end: index where train split ends.
    Returns (denoised_z [T,F], info dict with mu/sigma/U/alpha).
    """
    mu, sigma = zscore_fit(raw[:train_end])
    z = zscore_apply(raw, mu, sigma)
    # FCM over flows using TRAIN portion only, on interpretable shape features
    feats = flow_features(raw[:train_end], steps_per_day)  # [F, 3]
    U = fuzzy_cmeans(feats, c=c, seed=seed)                # [F, c]
    max_mem = U.max(axis=1)                               # [F]
    alpha = alpha0 * (1.0 + max_mem) / 2.0                # [F], in (alpha0/2, alpha0]
    s = wes_denoise(z, alpha)
    info = {"mu": mu, "sigma": sigma, "U": U, "alpha": alpha, "max_mem": max_mem}
    return s, info


if __name__ == "__main__":
    from synthetic import generate_geant_like
    TM, meta = generate_geant_like(n_nodes=6, weeks=1, seed=0)
    train_end = int(0.7 * TM.shape[0])
    s, info = preprocess(TM, train_end, c=4, alpha0=0.3, steps_per_day=meta["steps_per_day"])
    print("denoised shape", s.shape)
    print("alpha range", round(info["alpha"].min(), 3), "-", round(info["alpha"].max(), 3))
    print("membership range", round(info["max_mem"].min(), 3), "-", round(info["max_mem"].max(), 3))
    print("finite:", bool(np.isfinite(s).all()))
