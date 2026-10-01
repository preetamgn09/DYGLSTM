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


def run_epoch(model, Xs, Ys, origins, adj_map, opt=None, tf=0.0, is_graph=True,
              device="cpu", batch=64, static_adj=None):
    """Batched epoch. For graph models, the per-step adjacency for a whole
    minibatch is stacked into one [B,F,F] tensor so the batch runs in a single
    set of matmuls (and on GPU), instead of looping window-by-window."""
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    loss_fn = nn.MSELoss()
    order = np.random.permutation(len(Xs)) if train else np.arange(len(Xs))
    preds = []
    L = Xs.shape[1]
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        xb = torch.tensor(Xs[idx], dtype=torch.float32, device=device)
        yb = torch.tensor(Ys[idx], dtype=torch.float32, device=device)
        if is_graph:
            if static_adj is not None:
                # DCRNN-style: one fixed adjacency, same at every step
                Ab = static_adj.to(device).unsqueeze(0).expand(len(idx), -1, -1)
                adj_seq = [Ab] * L
            else:
                # dynamic: a different cached adjacency per encoder step
                adj_seq = []
                for t in range(L):
                    mats = [adj_map[int(origins[i]) - L + t] for i in idx]
                    adj_seq.append(torch.stack(mats, 0).to(device))
            out = model(xb, adj_seq, y_seq=yb, teacher_forcing=tf)
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
