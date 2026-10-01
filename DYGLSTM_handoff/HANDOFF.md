# DyGCN-AC-LSTM — handoff

## What this is
Clean rebuild of the network traffic-matrix prediction pipeline (the original
experiment code was lost). It runs end-to-end on synthetic + real Abilene data,
with the full DyGCN-AC-LSTM model and LSTM / AC-LSTM baselines. The graph forward
pass is batched and GPU-enabled. Current results are in RESULTS.md.

Context: the paper was rejected by Scientific Reports. Two reviews said the core
issues are EXPERIMENTS, not writing — mainly that the graph baselines were untuned.
This rebuild produces honest, reproducible numbers to address that.

## Setup
```bash
pip install -r requirements.txt        # numpy, scipy, torch
```
Data is NOT in the repo (too large for Git). Download the two CSVs from the
DDPM-TME repo's Dataset/ folder: https://github.com/Y-debug-sys/DDPM-TME
  - Abilene_TM.csv  (Dataset/...)
  - Geant_TM.csv    (Dataset/GEANT/...)
Put them in the project root (or pass a full --path).

On Colab: set Runtime -> T4 GPU. It should print `device: cuda` when it runs.

## Verify it works (reproduces RESULTS.md ~0.80, runs in a few minutes on GPU)
```bash
python run_abilene.py --path Abilene_TM.csv --drop-node 0 --sample-min 5 \
    --max-steps 12096 --seeds 42 123 456 789 1024 --log-transform
```
GEANT (already loads; 23 nodes / 506 flows, 15-min sampling):
```bash
python run_abilene.py --path Geant_TM.csv --sample-min 15 --min-active-frac 0 \
    --max-steps 12000 --seeds 42 --log-transform
```

## What's left, in priority order (this is the real work)
1. **Tuned graph baselines: STGCN, DCRNN, Graph WaveNet.**  THE decider.
   - DCRNN: **DONE** — implemented (src/dcrnn.py), tuned on the validation split
     (small grid over hidden/K, frozen test set), and runs automatically as part
     of `run_abilene.py`. The output now includes a DCRNN row.
   - STGCN, Graph WaveNet: still TO DO — same pattern (wrap reference, tune on
     val, add to run_abilene.py next to DCRNN).
2. **Ablations.** Turn off one component at a time and show each matters:
   (A) static vs dynamic adjacency, (B) linear gates vs DiffConv-in-gates,
   (C) no FCM/WES, (D) uniform clock vs clockwork. This was the paper's strongest
   evidence.
3. **GEANT real data.** Loader is done; just run + tune like Abilene.
4. **Transformer baseline.** Reviewer asked "why LSTM not Transformer" — one more
   baseline under the same protocol. Lower priority.
5. **Scale up.** Once tuned baselines exist, run on more of the data (currently
   6 weeks for tractability; full Abilene is 24 weeks) for final numbers.

## Design decisions already made (keep consistent in the paper)
- **Graph vertices = OD flows** (adjacency is [F, F], F = number of flows). The
  compute cost O(N^2 * W) is therefore in flows, not routers — update the paper's
  compute section accordingly.
- **Abilene = 11 nodes / 110 flows** (one of 12 PoPs dropped via --drop-node).
  State which node and why. GEANT = 23 nodes / 506 flows (matches paper).
- **WES smoothing:** standard EWMA, higher alpha = LESS smoothing (the submitted
  paper's text had this backwards — fix it). See src/preprocessing.py docstring.
- **FCM** clusters interpretable per-flow shape features (burstiness, diurnal
  energy, autocorrelation), not the raw high-dim series (which degenerates).
- **Log-transform** applied to raw traffic (~log-normal with big spikes) before
  z-scoring, so RMSE isn't dominated by bursts.

## Repo map
```
src/synthetic.py      GEANT-like generator (tested)
src/abilene.py        real Abilene/GEANT loader (tested on the actual files)
src/preprocessing.py  z-score + FCM + WES (tested)
src/adjacency.py      dynamic dCor adjacency, top-K, sym-norm, cache (tested)
src/datasets.py       non-overlapping windows + chronological splits (tested)
src/metrics.py        RMSE/MAE/sMAPE + Wilcoxon (tested)
src/diffconv.py       batched bidirectional diffusion conv
src/dygcn_ac_lstm.py  clockwork diffusion-gated cell + enc/dec
src/baselines.py      LSTM + AC-LSTM
src/train.py          batched training/eval loop (+ GPU)
run_abilene.py        real-data benchmark runner
run_experiment.py     synthetic benchmark runner
smoke_test.py         tiny end-to-end check
```
