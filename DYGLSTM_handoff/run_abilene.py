"""Real-data benchmark on Abilene, through the same pipeline as the synthetic run.

USAGE (after downloading the Abilene TM data):
  1) Inspect the file so you know the detected shape BEFORE training:
       python -m src.abilene --path /path/to/abilene --inspect
     (add --values-per-flow 5 if the file stores 5 columns per flow;
      add --drop-node K to reproduce the 11-node/110-flow variant.)
  2) Run the benchmark:
       python run_abilene.py --path /path/to/abilene [--values-per-flow 5] [--drop-node K]

Abilene split follows the manuscript: 65/15/20 chronological, 5-min sampling
(steps_per_day=288), W=48 (~4h). Graph vertices = OD flows.
"""
import argparse, numpy as np, torch
from src.abilene import load_abilene
from src.preprocessing import preprocess
from src.datasets import make_windows
from src.adjacency import build_adjacency
from src.train import precompute_adjacency, run_epoch, evaluate
from src.dygcn_ac_lstm import DyGCNACLSTM
from src.dcrnn import DCRNN
from src.baselines import LSTMBaseline, ACLSTMBaseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = dict(SEQ=12, HOR=12, W=48, K=6, P=4, HIDDEN=64, G=4, DIFF_K=3,
           EPOCHS=30, SEEDS=[42, 123, 456, 789, 1024])


def prepare(TM, meta, seed):
    T, F = TM.shape
    tr = int(0.65 * T); va = int(0.80 * T)          # 65/15/20 for Abilene
    den, _ = preprocess(TM, tr, c=4, alpha0=0.3, steps_per_day=meta["steps_per_day"])
    static = build_adjacency(den[:CFG["W"]], CFG["K"])
    def win(a, b):
        X, Y, idx = make_windows(den[a:b], CFG["SEQ"], CFG["HOR"]); return X, Y, idx + a
    sets = dict(train=win(0, tr), val=win(tr, va), test=win(va, T))
    needed = []
    for _, _, O in sets.values():
        for o in O:
            needed += [o - CFG["SEQ"] + t for t in range(CFG["SEQ"])]
    amap = precompute_adjacency(den, needed, CFG["W"], CFG["K"], CFG["P"], static=static)
    import torch as _t
    static_adj = _t.tensor(static, dtype=_t.float32)   # one fixed graph for DCRNN
    return F, sets, amap, static_adj


def train_model(model, sets, amap, is_graph, static_adj=None):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)
    Xtr, Ytr, Otr = sets["train"]
    for _ in range(CFG["EPOCHS"]):
        run_epoch(model, Xtr, Ytr, Otr, amap, opt=opt, tf=0.5, is_graph=is_graph,
                  device=DEVICE, static_adj=static_adj); sch.step()
    return model

def eval_split(model, sets, split, amap, is_graph, static_adj=None):
    X, Y, O = sets[split]
    _, pred = run_epoch(model, X, Y, O, amap, opt=None, is_graph=is_graph,
                        device=DEVICE, static_adj=static_adj)
    return evaluate(Y, pred)

def fit_eval(model, sets, amap, is_graph, static_adj=None):
    m = train_model(model, sets, amap, is_graph, static_adj)
    return eval_split(m, sets, "test", amap, is_graph, static_adj)

def tune_dcrnn(sets, amap, static_adj, F):
    """Small validation grid search (addresses the reviewer: baselines must be
    tuned on validation, not run at defaults). Picks the config with best VAL RMSE."""
    grid = [(h, k) for h in (32, 64) for k in (2, 3)]
    best = None
    for (hidden, k) in grid:
        torch.manual_seed(0); np.random.seed(0)     # fixed seed for fair tuning
        m = train_model(DCRNN(hidden=hidden, K=k, layers=2, horizon=CFG["HOR"]),
                        sets, amap, True, static_adj)
        v = eval_split(m, sets, "val", amap, True, static_adj)["RMSE"]
        print(f"  [tune DCRNN] hidden={hidden} K={k} -> val RMSE {v:.4f}")
        if best is None or v < best[0]:
            best = (v, hidden, k)
    print(f"  [tune DCRNN] selected hidden={best[1]} K={best[2]} (val {best[0]:.4f})")
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--values-per-flow", type=int, default=1)
    ap.add_argument("--value-col", type=int, default=0)
    ap.add_argument("--drop-node", type=int, default=None)
    ap.add_argument("--sample-min", type=int, default=5, help="5 for Abilene, 15 for GEANT")
    ap.add_argument("--min-active-frac", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=None, help="use only first N timesteps (tractability)")
    ap.add_argument("--log-transform", action="store_true", help="log1p traffic (recommended for real data)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=CFG["SEEDS"])
    a = ap.parse_args()
    CFG["EPOCHS"] = a.epochs; CFG["SEEDS"] = a.seeds

    print(f"device: {DEVICE}")
    TM, meta = load_abilene(a.path, values_per_flow=a.values_per_flow,
                            value_col=a.value_col, drop_node=a.drop_node,
                            sample_min=a.sample_min, min_active_frac=a.min_active_frac,
                            log_transform=a.log_transform)
    if a.max_steps:
        TM = TM[:a.max_steps]; meta["T"] = TM.shape[0]
    print(f"Abilene: N={meta['N_nodes']} nodes, F={meta['F']} flows, T={meta['T']} steps "
          f"({meta['flows_dropped_unstable']} unstable flows dropped)\n")

    # --- tune DCRNN once on the first seed's validation split ---
    F0, sets0, amap0, sadj0 = prepare(TM, meta, CFG["SEEDS"][0])
    print("Tuning DCRNN on validation (equal-budget, frozen test):")
    dc_hidden, dc_K = tune_dcrnn(sets0, amap0, sadj0, F0)
    print()

    results = {k: [] for k in ["LSTM", "AC-LSTM", "DCRNN", "DyGCN-AC-LSTM"]}
    for seed in CFG["SEEDS"]:
        torch.manual_seed(seed); np.random.seed(seed)
        F, sets, amap, sadj = prepare(TM, meta, seed)
        # temporal baselines
        results["LSTM"].append(fit_eval(LSTMBaseline(F, CFG["HIDDEN"], CFG["HOR"]), sets, amap, False)["RMSE"])
        results["AC-LSTM"].append(fit_eval(ACLSTMBaseline(F, CFG["HIDDEN"], CFG["G"], CFG["HOR"]), sets, amap, False)["RMSE"])
        # tuned static-graph baseline
        results["DCRNN"].append(fit_eval(DCRNN(hidden=dc_hidden, K=dc_K, layers=2, horizon=CFG["HOR"]), sets, amap, True, static_adj=sadj)["RMSE"])
        # proposed dynamic-graph model
        results["DyGCN-AC-LSTM"].append(fit_eval(DyGCNACLSTM(CFG["HIDDEN"], CFG["G"], CFG["DIFF_K"], CFG["HOR"]), sets, amap, True)["RMSE"])
        for name in results:
            print(f"seed {seed} {name:16s} RMSE {results[name][-1]:.4f}")
    print("\n=== Abilene: RMSE mean +/- std over seeds ===")
    print(f"(DCRNN tuned on val: hidden={dc_hidden}, K={dc_K})")
    for name, vals in results.items():
        v = np.array(vals); print(f"{name:16s} {v.mean():.4f} +/- {v.std():.4f}")


if __name__ == "__main__":
    main()
