# Results so far (rebuilt pipeline)

All numbers below are from THIS repo (the original code was lost; this is a clean
rebuild). They are RMSE on log-transformed, z-scored traffic, lower = better.

## Abilene real data — CONFIRMED, reproducible
Config: 11 nodes / 110 flows (one PoP dropped), 6 weeks (12096 steps, 5-min),
5 seeds, log-transform, 30 epochs.

| Model            | RMSE (mean ± std) |
|------------------|-------------------|
| LSTM             | 1.012 ± 0.003     |
| AC-LSTM          | 1.002 ± 0.003     |
| DyGCN-AC-LSTM    | **0.802 ± 0.005** |

~20% improvement over the temporal baselines, stable across all 5 seeds (gap is
~40× the seed noise). A trivial mean-predictor scores ~1.16 here, so the baselines
barely beat it while DyGCN clearly does.

## Synthetic GEANT-like — reproduces the original paper's Table 3 scale
LSTM 0.277 · AC-LSTM 0.247 · DyGCN-AC-LSTM 0.214 (single seed sanity run).

## NOT yet run (these are the remaining work — see HANDOFF.md)
- Tuned graph baselines (STGCN / DCRNN / Graph WaveNet)  <-- the rejection reason
- Ablations, GEANT real data, Transformer baseline
