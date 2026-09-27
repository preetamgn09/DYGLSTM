# DyGCN-AC-LSTM — clean-room rebuild

Rebuild of the network traffic-matrix prediction model from the manuscript, from
scratch, in a proper repo so results are reproducible. Original code was lost; this
regenerates it honestly rather than reusing unverifiable numbers.

## What runs today
- **Synthetic GÉANT-like benchmark** (no download) — generate data, train, evaluate.
- **Full model** DyGCN-AC-LSTM + **temporal baselines** LSTM and AC-LSTM.
- End-to-end: synthetic data → FCM+WES preprocessing → causal dynamic dCor adjacency
  → clockwork diffusion-gated encoder/decoder → metrics.

## Quick start (Colab or any machine with Python 3.10+)
```bash
pip install -r requirements.txt        # torch installs cleanly here
python smoke_test.py                   # ~1-2 min, proves the pipeline runs
python run_experiment.py               # multi-seed synthetic benchmark table
```
If `smoke_test.py` prints "SMOKE TEST PASSED", the pipeline is working. If it throws,
paste the full traceback back and it gets fixed fast.

## Design decision that resolves the reviewer's ambiguity
**Graph vertices = OD flows.** The adjacency Â_t has shape **[F, F]** where F is the
number of directed OD flows (30 for 6 nodes, 110 for Abilene's 11, 506 for GÉANT's 23).
Every module uses this definition. Consequence: the O(N²·W) adjacency cost is in
**N = flows**, not routers — so the manuscript's compute section must be rewritten in
terms of flows if you keep this choice. To switch to a router-level graph instead,
change only the feature that feeds `adjacency.py` (aggregate flows→routers) and the
shape becomes [N_routers, N_routers]; nothing else changes.

## What was tested vs. what you must smoke-test
- **Tested here (NumPy core, all passing):** synthetic generator; FCM (verified crisp
  on separable data) + WES; distance-correlation adjacency (dCor=1 for identical/negated,
  ~0 for independent; symmetric; spectral radius ≤ 1); metrics; non-overlapping windows.
- **Written + syntax-checked, run it yourself:** the PyTorch model, baselines, and
  training loop. torch could not be installed in the build sandbox (CUDA/disk), so the
  smoke test is your first execution. This is expected — run it on Colab.

## Corrections already baked in (from the two reviews)
- **WES direction:** standard EWMA `s_t = α·x_t + (1-α)·s_{t-1}` — higher α = LESS
  smoothing. The manuscript's "higher α ⇒ more aggressive smoothing" is the opposite;
  code + docstring state the correct interpretation. Flip the α-map if you meant the
  other behaviour, and fix the paper text to match.
- **FCM feature:** clustering the raw high-dim noisy series degenerates (curse of
  dimensionality). We cluster on interpretable per-flow shape features (burstiness,
  diurnal energy, lag-1 autocorrelation) — robust and non-degenerate. Document this in
  Methods (it's a deviation from "cluster the time series").
- **Causality:** adjacency at forecast origin t uses window strictly before t; decoder
  reuses the most recent cached matrix (≤ t). No future data enters.
- **Independent unit for significance:** windows are non-overlapping; use per-window
  MSE as the Wilcoxon unit (`metrics.paired_wilcoxon`), not every element.
- **Clockwork compute:** active-update fraction for periods {1,2,4,8} is
  (1+½+¼+⅛)/4 ≈ 0.47 (~53% fewer cell updates), an operation count — report measured
  wall-clock separately (it's higher than plain LSTM due to graph cost).

## Roadmap (next chunks, in order)
1. **Real datasets** — Abilene + GÉANT Totem loaders (map their TM files into `[T, F]`),
   swap `generate_geant_like` for a loader. Same pipeline downstream.
2. **Graph baselines** — wrap reference STGCN / DCRNN / Graph WaveNet, tuned on val.
3. **Transformer baseline** — encoder-decoder forecaster under the same protocol.
4. **Tuning harness** — equal budget, frozen test set, search spaces logged.
5. **Ablations** — Variant A–D (static adj / linear gates / no-FCM / uniform clock),
   plus the novelty-isolating (a)-(d) matched ablation and graph-construction robustness.
6. **Adjacency-evolution analysis** — snapshots, change-rate, shuffled-graph control.

## Files
```
src/synthetic.py      GÉANT-like generator (tested)
src/preprocessing.py  z-score + FCM + WES (tested)
src/adjacency.py      dynamic dCor adjacency, top-K, sym-norm, cache (tested)
src/datasets.py       non-overlapping windows + chronological splits (tested)
src/metrics.py        RMSE/MAE/sMAPE + Wilcoxon (tested)
src/diffconv.py       bidirectional diffusion conv (torch; smoke-test it)
src/dygcn_ac_lstm.py  clockwork diffusion-gated cell + enc/dec (torch)
src/baselines.py      LSTM + AC-LSTM (torch)
src/train.py          training/eval loop + adjacency precompute (torch)
smoke_test.py         tiny end-to-end check — RUN FIRST
run_experiment.py     multi-seed synthetic benchmark
```
