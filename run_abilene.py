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
from src.baselines import LSTMBaseline, ACLSTMBaseline

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
    return F, sets, amap


def fit_eval(model, sets, amap, is_graph):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)
    Xtr, Ytr, Otr = sets["train"]
    for _ in range(CFG["EPOCHS"]):
        run_epoch(model, Xtr, Ytr, Otr, amap, opt=opt, tf=0.5, is_graph=is_graph); sch.step()
    Xte, Yte, Ote = sets["test"]
    _, pred = run_epoch(model, Xte, Yte, Ote, amap, opt=None, is_graph=is_graph)
    return evaluate(Yte, pred)


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

    TM, meta = load_abilene(a.path, values_per_flow=a.values_per_flow,
                            value_col=a.value_col, drop_node=a.drop_node,
                            sample_min=a.sample_min, min_active_frac=a.min_active_frac,
                            log_transform=a.log_transform)
    if a.max_steps:
        TM = TM[:a.max_steps]; meta["T"] = TM.shape[0]
    print(f"Abilene: N={meta['N_nodes']} nodes, F={meta['F']} flows, T={meta['T']} steps "
          f"({meta['flows_dropped_unstable']} unstable flows dropped)\n")

    builders = {
        "LSTM":          (lambda F: LSTMBaseline(F, CFG["HIDDEN"], CFG["HOR"]), False),
        "AC-LSTM":       (lambda F: ACLSTMBaseline(F, CFG["HIDDEN"], CFG["G"], CFG["HOR"]), False),
        "DyGCN-AC-LSTM": (lambda F: DyGCNACLSTM(CFG["HIDDEN"], CFG["G"], CFG["DIFF_K"], CFG["HOR"]), True),
    }
    results = {k: [] for k in builders}
    for seed in CFG["SEEDS"]:
        torch.manual_seed(seed); np.random.seed(seed)
        F, sets, amap = prepare(TM, meta, seed)
        for name, (build, is_g) in builders.items():
            m = fit_eval(build(F), sets, amap, is_g)
            results[name].append(m["RMSE"])
            print(f"seed {seed} {name:16s} RMSE {m['RMSE']:.4f}")
    print("\n=== Abilene: RMSE mean +/- std over seeds ===")
    for name, vals in results.items():
        v = np.array(vals); print(f"{name:16s} {v.mean():.4f} +/- {v.std():.4f}")


if __name__ == "__main__":
    main()
