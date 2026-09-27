"""GÉANT-like synthetic traffic-matrix generator.

Produces a traffic matrix stream with the statistical properties described in the
manuscript: diurnal + weekly seasonality, per-node heterogeneity, Gaussian
measurement noise, and rare large spikes. Output is the set of directed OD-flow
time series (N_nodes * (N_nodes-1) flows), each a column over T time steps.

This is fully self-contained: no download required, so it is the fastest way to
get a real end-to-end result and verify the pipeline.
"""
import numpy as np


def generate_geant_like(n_nodes=23, weeks=3, sample_min=15, noise_sigma=0.05,
                        spike_prob=0.02, spike_amp=(50, 150), seed=42):
    """Return (TM, meta).

    TM has shape [T, F] where F = n_nodes*(n_nodes-1) directed OD flows.
    Values are non-negative simulated volumes (arbitrary Mbps-like units).
    """
    rng = np.random.default_rng(seed)
    steps_per_day = int(24 * 60 / sample_min)
    T = int(weeks * 7 * steps_per_day)
    t = np.arange(T)

    # diurnal (1 cycle/day) + weekly (1 cycle/7 days) shared seasonality
    diurnal = 0.5 * (1 + np.sin(2 * np.pi * t / steps_per_day - np.pi / 2))
    weekly = 0.2 * (1 + np.sin(2 * np.pi * t / (7 * steps_per_day)))
    season = 1.0 + diurnal + weekly  # >0

    flows = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
    F = len(flows)
    TM = np.zeros((T, F), dtype=np.float64)

    # four flow archetypes so flows are genuinely heterogeneous in SHAPE (not just
    # scale) -> FCM finds real structure. Archetype sets the diurnal weight and
    # burstiness; this mirrors "per-node heterogeneity / varied application mix".
    archetypes = {
        0: dict(diurnal_w=0.9, weekly_w=0.3, burst=1.0, phase=0.0),          # business-hours
        1: dict(diurnal_w=0.2, weekly_w=0.1, burst=0.3, phase=np.pi),        # near-flat / inverted
        2: dict(diurnal_w=0.5, weekly_w=0.6, burst=0.5, phase=np.pi/2),      # weekly-dominant
        3: dict(diurnal_w=0.6, weekly_w=0.2, burst=2.0, phase=3*np.pi/2),    # bursty
    }
    flow_arch = rng.integers(0, 4, size=F)

    for f, (i, j) in enumerate(flows):
        a = archetypes[int(flow_arch[f])]
        wi = 0.5 + rng.random(); wj = 0.5 + rng.random()
        base = wi * wj
        phase = a["phase"] + rng.normal(0, 0.15)  # archetype phase + small jitter
        diur = a["diurnal_w"] * (1 + np.sin(2 * np.pi * t / steps_per_day + phase))
        week = a["weekly_w"] * (1 + np.sin(2 * np.pi * t / (7 * steps_per_day)))
        shape = 1.0 + diur + week
        series = base * shape
        series = series * (1.0 + rng.normal(0, noise_sigma, size=T))
        spikes = rng.random(T) < (spike_prob * a["burst"])
        amp = rng.uniform(spike_amp[0], spike_amp[1], size=T)
        series = series + spikes * amp * base
        TM[:, f] = np.clip(series, 0, None)

    meta = {"n_nodes": n_nodes, "flows": flows, "T": T,
            "steps_per_day": steps_per_day, "F": F}
    return TM, meta


if __name__ == "__main__":
    TM, meta = generate_geant_like(n_nodes=6, weeks=1, seed=0)
    print("TM shape", TM.shape, "| F =", meta["F"], "| T =", meta["T"])
    print("mean", round(TM.mean(), 3), "| max", round(TM.max(), 2),
          "| %near-zero", round(float((np.abs(TM) < 0.1).mean()) * 100, 1))
