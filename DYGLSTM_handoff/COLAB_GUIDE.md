# DyGCN-AC-LSTM — Colab run guide

Run these cells top to bottom in a fresh Google Colab notebook
(colab.research.google.com -> New notebook). Each block = one code cell.
No files need to be uploaded manually — code and data are both pulled from GitHub.

------------------------------------------------------------------------
## STEP 0 — Turn on the GPU (do this first, in the menu)
Runtime -> Change runtime type -> Hardware accelerator: T4 GPU -> Save.

Then run this cell to confirm:
```python
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE — go set Runtime > Change runtime type > T4 GPU")
```
Expect a GPU name (e.g. "Tesla T4"). If it says NONE, fix the runtime first.

------------------------------------------------------------------------
## STEP 1 — Get the code
```python
!git clone https://github.com/preetamgn09/DYGLSTM.git
%cd /content/DYGLSTM
!ls
```
You should see run_abilene.py, run_experiment.py, src/, README.md, etc.

------------------------------------------------------------------------
## STEP 2 — Get the data (auto-downloaded, no manual upload)
```python
import shutil, glob
!git clone --depth 1 https://github.com/Y-debug-sys/DDPM-TME.git /content/ddpm
for name in ["Abilene_TM.csv", "Geant_TM.csv"]:
    hits = glob.glob(f"/content/ddpm/**/{name}", recursive=True)
    assert hits, f"{name} not found — check the DDPM-TME repo layout"
    shutil.copy(hits[0], f"/content/DYGLSTM/{name}")
    print("copied", hits[0])
```

------------------------------------------------------------------------
## STEP 3 — Install the one extra dependency
```python
%cd /content/DYGLSTM
!pip install -q scipy
```
(torch and numpy are already on Colab.)

------------------------------------------------------------------------
## STEP 4 — Quick verify (1 seed, ~a few min). Confirms setup works.
```python
!python run_abilene.py --path Abilene_TM.csv --drop-node 0 --sample-min 5 \
    --max-steps 12096 --seeds 42 --log-transform
```
SUCCESS CHECK: prints `device: cuda`, finishes in minutes, and DyGCN-AC-LSTM
RMSE is ~0.80. If so, everything works. If it errors, copy the FULL error.

------------------------------------------------------------------------
## STEP 5 — Full Abilene run (5 seeds) — a real result
```python
!python run_abilene.py --path Abilene_TM.csv --drop-node 0 --sample-min 5 \
    --max-steps 12096 --seeds 42 123 456 789 1024 --log-transform
```
Expected: DyGCN ~0.80 vs AC-LSTM ~1.00 vs LSTM ~1.01, PLUS a tuned DCRNN row.
The run first prints a short "Tuning DCRNN on validation" section (it tries a few
configs and picks the best on val), then the per-seed table including DCRNN.
THE KEY NUMBER: does DyGCN-AC-LSTM still beat the TUNED DCRNN? That is the
comparison the paper was rejected for.

------------------------------------------------------------------------
## STEP 6 — GEANT real data (5 seeds)
```python
!python run_abilene.py --path Geant_TM.csv --sample-min 15 --min-active-frac 0 \
    --max-steps 12000 --seeds 42 123 456 789 1024 --log-transform
```

------------------------------------------------------------------------
## STEP 7 — Save the results so they aren't lost
Copy the printed "RMSE mean +/- std" tables into a file and download it:
```python
results_text = """PASTE the printed Abilene and GEANT tables here"""
with open("/content/DYGLSTM/RESULTS_new.txt", "w") as f:
    f.write(results_text)
from google.colab import files
files.download("/content/DYGLSTM/RESULTS_new.txt")
```
(Or just screenshot the output cells.)

------------------------------------------------------------------------
## NOTES / GOTCHAS
- Colab disconnects after ~90 min idle. For long runs keep the tab open/active.
- First GPU run is also the first real test of the batched code. If it throws a
  device or dtype error, that's a small fix — send the full traceback.
- These runs use 6 weeks of data (--max-steps ~12000) for speed. For final paper
  numbers, raise --max-steps (full Abilene = 48384) once everything checks out.

------------------------------------------------------------------------
## WHAT STILL NEEDS TO BE BUILT (not runnable yet — this is the real work)
The commands above run the proposed model + LSTM/AC-LSTM baselines only.
Still TO DO (see HANDOFF.md):
1. Tuned graph baselines: DCRNN is DONE (runs automatically, tuned on val).
   STILL TO BUILD: STGCN, Graph WaveNet (same pattern as DCRNN).
2. Ablations (static adj / linear gates / no-FCM / uniform clock)
3. Transformer baseline
Items 2-3 and the two remaining baselines need new code before they can be run.
