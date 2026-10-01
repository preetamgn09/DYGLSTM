"""Fuller synthetic-benchmark run: trains the full model + baselines over multiple
seeds and prints a results table with mean +/- std. Run AFTER smoke_test passes.
Still synthetic-only and small-ish; scale N_NODES/EPOCHS up as your compute allows.
For real datasets and graph baselines, see README 'Roadmap'.
"""
import numpy as np, torch
from src.synthetic import generate_geant_like
from src.preprocessing import preprocess
from src.datasets import make_windows, chrono_split
from src.adjacency import build_adjacency
from src.train import precompute_adjacency, run_epoch, evaluate
from src.dygcn_ac_lstm import DyGCNACLSTM
from src.baselines import LSTMBaseline, ACLSTMBaseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = dict(N_NODES=10, WEEKS=3, SEQ=12, HOR=12, W=48, K=6, P=4,
           HIDDEN=64, G=4, DIFF_K=3, EPOCHS=30, SEEDS=[42,123,456,789,1024])

def prepare(seed):
    TM, meta = generate_geant_like(n_nodes=CFG["N_NODES"], weeks=CFG["WEEKS"], seed=seed)
    T, F = TM.shape
    tr,va = chrono_split(T,(0.7,0.1,0.2))
    den,_ = preprocess(TM, tr, steps_per_day=meta["steps_per_day"])
    static = build_adjacency(den[:CFG["W"]], CFG["K"])
    def win(a,b):
        X,Y,idx = make_windows(den[a:b], CFG["SEQ"], CFG["HOR"]); return X,Y,idx+a
    sets = dict(train=win(0,tr), val=win(tr,va), test=win(va,T))
    needed=[]
    for _,_,O in sets.values():
        for o in O: needed += [o-CFG["SEQ"]+t for t in range(CFG["SEQ"])]
    amap = precompute_adjacency(den, needed, CFG["W"], CFG["K"], CFG["P"], static=static)
    return F, sets, amap

def fit_eval(model, sets, amap, is_graph):
    model=model.to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30,eta_min=1e-5)
    Xtr,Ytr,Otr=sets["train"]
    for _ in range(CFG["EPOCHS"]):
        run_epoch(model,Xtr,Ytr,Otr,amap,opt=opt,tf=0.5,is_graph=is_graph,device=DEVICE); sch.step()
    Xte,Yte,Ote=sets["test"]
    _,pred=run_epoch(model,Xte,Yte,Ote,amap,opt=None,is_graph=is_graph,device=DEVICE)
    return evaluate(Yte,pred)

def run_all():
    builders = {
        "LSTM":       (lambda F: LSTMBaseline(F,CFG["HIDDEN"],CFG["HOR"]), False),
        "AC-LSTM":    (lambda F: ACLSTMBaseline(F,CFG["HIDDEN"],CFG["G"],CFG["HOR"]), False),
        "DyGCN-AC-LSTM": (lambda F: DyGCNACLSTM(CFG["HIDDEN"],CFG["G"],CFG["DIFF_K"],CFG["HOR"]), True),
    }
    results={k:[] for k in builders}
    for seed in CFG["SEEDS"]:
        torch.manual_seed(seed); np.random.seed(seed)
        F,sets,amap=prepare(seed)
        for name,(build,is_g) in builders.items():
            m=fit_eval(build(F),sets,amap,is_g)
            results[name].append(m["RMSE"]); print(f"seed {seed} {name:16s} RMSE {m['RMSE']:.4f}")
    print("\n=== Synthetic benchmark: RMSE mean +/- std over seeds ===")
    for name,vals in results.items():
        v=np.array(vals); print(f"{name:16s} {v.mean():.4f} +/- {v.std():.4f}")

if __name__=="__main__":
    run_all()
