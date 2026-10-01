"""DCRNN baseline (Li et al., ICLR 2018): diffusion-convolutional GRU, seq2seq.

Key difference from the proposed model: DCRNN uses a STATIC adjacency fixed at
training time (built once from the training-split distance correlation), whereas
DyGCN-AC-LSTM rebuilds the adjacency every step. This makes DCRNN the fair
"static graph" competitor. It reuses the same batched DiffConv operator.

Interface matches the graph models: forward(x_seq, adj_seq, ...) where adj_seq is
a list of [B,F,F]; DCRNN uses a single static adjacency (adj_seq[0]) at every step.
"""
import torch
import torch.nn as nn
from .diffconv import DiffConv


class DCGRUCell(nn.Module):
    """Diffusion-convolutional GRU cell. Gates are DiffConv over [x; h]."""
    def __init__(self, c_in, hidden, K=3):
        super().__init__()
        self.hidden = hidden
        self.gate = DiffConv(c_in + hidden, 2 * hidden, K)   # reset + update
        self.cand = DiffConv(c_in + hidden, hidden, K)       # candidate state

    def forward(self, x, h, A):
        # x [B,F,c_in], h [B,F,hidden], A [B,F,F]
        xh = torch.cat([x, h], dim=-1)
        ru = torch.sigmoid(self.gate(xh, A))
        r, u = ru.chunk(2, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], dim=-1), A))
        return u * h + (1.0 - u) * c


class DCRNN(nn.Module):
    def __init__(self, hidden=64, K=3, layers=2, horizon=12):
        super().__init__()
        self.hidden, self.layers, self.horizon = hidden, layers, horizon
        self.enc = nn.ModuleList(
            [DCGRUCell(1 if l == 0 else hidden, hidden, K) for l in range(layers)])
        self.dec = nn.ModuleList(
            [DCGRUCell(1 if l == 0 else hidden, hidden, K) for l in range(layers)])
        self.proj = nn.Linear(hidden, 1)

    def _zero(self, B, F, dev):
        return [torch.zeros(B, F, self.hidden, device=dev) for _ in range(self.layers)]

    def forward(self, x_seq, adj_seq, y_seq=None, teacher_forcing=0.0):
        B, L, F = x_seq.shape
        dev = x_seq.device
        A = adj_seq[0]                                   # static adjacency
        h = self._zero(B, F, dev)
        # encoder
        for t in range(L):
            inp = x_seq[:, t].unsqueeze(-1)              # [B,F,1]
            for l in range(self.layers):
                h[l] = self.enc[l](inp, h[l], A)
                inp = h[l]
        # decoder (autoregressive from a GO=zeros symbol)
        outs = []
        inp0 = torch.zeros(B, F, 1, device=dev)
        prev = inp0
        for k in range(self.horizon):
            inp = prev
            for l in range(self.layers):
                h[l] = self.dec[l](inp, h[l], A)
                inp = h[l]
            yhat = self.proj(h[-1]).squeeze(-1)          # [B,F]
            outs.append(yhat)
            if self.training and y_seq is not None and torch.rand(1).item() < teacher_forcing:
                prev = y_seq[:, k].unsqueeze(-1)
            else:
                prev = yhat.unsqueeze(-1).detach()
        return torch.stack(outs, dim=1)                  # [B,H,F]
