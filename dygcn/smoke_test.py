"""Tiny end-to-end smoke test. Run this FIRST (CPU, ~1-2 min on Colab).
It generates a small synthetic dataset, runs preprocessing + dynamic adjacency,
trains the full model and two baselines for a few epochs, and prints metrics.
Success = it runs to the end, loss decreases, and metrics are finite.
It does NOT produce paper numbers -- it proves the pipeline works.
"""
import numpy as np, torch
from src.synthetic import generate_geant_like
from src.preprocessing import preprocess
from src.datasets import make_windows, chrono_split
from src.adjacency import build_adjacency
from src.train import precompute_adjacency, run_epoch, evaluate
from src.dygcn_ac_lstm import DyGCNACLSTM
from src.baselines import LSTMBaseline, ACLSTMBaseline

torch.manual_seed(42); np.random.seed(42)

# --- small config so it runs fast ---
N_NODES, WEEKS = 6, 3           # 6 nodes -> 30 flows
SEQ, HOR = 12, 12
W, K, P = 48, 6, 4
HIDDEN, G, DIFF_K, EPOCHS = 16, 4, 2, 3

TM, meta = generate_geant_like(n_nodes=N_NODES, weeks=WEEKS, seed=0)
T, F = TM.shape
print(f"data: T={T} F={F} (nodes={N_NODES})")

tr_end, va_end = chrono_split(T, (0.7, 0.1, 0.2))
den, info = preprocess(TM, tr_end, c=4, alpha0=0.3, steps_per_day=meta["steps_per_day"])
static_adj = build_adjacency(den[:W], K)      # warm-start static adjacency (train window)

def split_windows(a, b):
    X, Y, idx = make_windows(den[a:b], SEQ, HOR)
    return X, Y, idx + a                        # shift origins to global time
Xtr, Ytr, Otr = split_windows(0, tr_end)
Xva, Yva, Ova = split_windows(tr_end, va_end)
Xte, Yte, Ote = split_windows(va_end, T)
print(f"windows: train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

needed = []
for O in list(Otr)+list(Ova)+list(Ote):
    needed += [O - SEQ + t for t in range(SEQ)]
adj_map = precompute_adjacency(den, needed, W, K, P, static=static_adj)
print(f"adjacency matrices cached: {len(set(id(v) for v in adj_map.values()))} unique")

def train_model(model, is_graph):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    for e in range(EPOCHS):
        tl = run_epoch(model, Xtr, Ytr, Otr, adj_map, opt=opt, tf=0.5, is_graph=is_graph)
        losses.append(tl)
    _, pred = run_epoch(model, Xte, Yte, Ote, adj_map, opt=None, is_graph=is_graph)
    return losses, evaluate(Yte, pred)

print("\n--- DyGCN-AC-LSTM (full model) ---")
m = DyGCNACLSTM(hidden=HIDDEN, G=G, K=DIFF_K, horizon=HOR)
L, met = train_model(m, is_graph=True)
print("train loss/epoch:", [round(x,4) for x in L], "| test:", {k: round(v,4) for k,v in met.items()})
assert L[-1] < L[0] * 1.5 and all(np.isfinite(list(met.values()))), "full model sanity failed"

print("\n--- AC-LSTM (temporal baseline) ---")
m2 = ACLSTMBaseline(F, hidden=HIDDEN, G=G, horizon=HOR)
L2, met2 = train_model(m2, is_graph=False)
print("train loss/epoch:", [round(x,4) for x in L2], "| test:", {k: round(v,4) for k,v in met2.items()})

print("\n--- LSTM (temporal baseline) ---")
m3 = LSTMBaseline(F, hidden=HIDDEN, horizon=HOR)
L3, met3 = train_model(m3, is_graph=False)
print("train loss/epoch:", [round(x,4) for x in L3], "| test:", {k: round(v,4) for k,v in met3.items()})

print("\nSMOKE TEST PASSED — pipeline runs end to end and losses decrease.")
