"""End-to-end training/eval for one model on one dataset split.
Adjacency is precomputed per cache-anchor (respecting warm-start + p-step caching)
so we never rebuild the same dCor matrix twice."""
import numpy as np
import torch
import torch.nn as nn
from .adjacency import build_adjacency
from .metrics import all_metrics, paired_wilcoxon


def _anchor(t, W, p):
    if t < W:
        return None                      # warm-start -> static adjacency
    return W + ((t - W) // p) * p        # most recent recompute time <= t


def precompute_adjacency(series, needed_times, W=48, k=6, p=4, static=None, device="cpu"):
    """Return dict: time -> torch adjacency [F,F]. Builds each anchor once."""
    cache = {}
    out = {}
    st = torch.tensor(static, dtype=torch.float32, device=device) if static is not None else None
    for t in sorted(set(int(x) for x in needed_times)):
        a = _anchor(t, W, p)
        if a is None:
            out[t] = st
        else:
            if a not in cache:
                A = build_adjacency(series[a - W:a], k)          # causal: window strictly < a
                cache[a] = torch.tensor(A, dtype=torch.float32, device=device)
            out[t] = cache[a]
    return out


def adj_list_for_window(origin, L, adj_map):
    """Encoder step times are [origin-L, origin); decoder reuses the last."""
    return [adj_map[origin - L + t] for t in range(L)]


def run_epoch(model, Xs, Ys, origins, adj_map, opt=None, tf=0.0, is_graph=True):
    train = opt is not None
    model.train(train)
    B = 32
    total, n = 0.0, 0
    loss_fn = nn.MSELoss()
    order = np.random.permutation(len(Xs)) if train else np.arange(len(Xs))
    preds = []
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        xb = torch.tensor(Xs[idx], dtype=torch.float32)
        yb = torch.tensor(Ys[idx], dtype=torch.float32)
        if is_graph:
            # per-sample adjacency lists differ; loop is simplest & correct for v1
            outs = []
            for bi, o in enumerate(origins[idx]):
                al = adj_list_for_window(int(o), xb.shape[1], adj_map)
                outs.append(model(xb[bi:bi+1], al, y_seq=yb[bi:bi+1], teacher_forcing=tf))
            out = torch.cat(outs, 0)
        else:
            out = model(xb, y_seq=yb, teacher_forcing=tf)
        loss = loss_fn(out, yb)
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        total += loss.item() * len(idx); n += len(idx)
        if not train:
            preds.append(out.detach().cpu().numpy())
    if train:
        return total / n
    return total / n, np.concatenate(preds, 0)


def evaluate(y_true, y_pred):
    return all_metrics(y_true, y_pred)
