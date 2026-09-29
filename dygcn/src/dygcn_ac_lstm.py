"""DyGCN-AC-LSTM: diffusion graph convolution inside Adaptive Clockwork LSTM gates.

Design: the graph is over OD FLOWS. Each flow-node carries a per-node hidden vector
of size H. Hidden channels are split into G clockwork groups with periods 1,2,4,8;
only groups whose period divides the (1-indexed) step are updated, the rest hold
state. Every gate operator is a DiffConv over the per-step adjacency A_t, with
GROUP-SPECIFIC filters (each group has its own gate parameters).
"""
import torch
import torch.nn as nn
from .diffconv import DiffConv

PERIODS = [1, 2, 4, 8]


class GroupGates(nn.Module):
    """The four LSTM gates for ONE clockwork group, each a DiffConv over x and h."""
    def __init__(self, c_in, h_g, K):
        super().__init__()
        self.i_x = DiffConv(c_in, h_g, K); self.i_h = DiffConv(h_g, h_g, K, bias=False)
        self.f_x = DiffConv(c_in, h_g, K); self.f_h = DiffConv(h_g, h_g, K, bias=False)
        self.o_x = DiffConv(c_in, h_g, K); self.o_h = DiffConv(h_g, h_g, K, bias=False)
        self.g_x = DiffConv(c_in, h_g, K); self.g_h = DiffConv(h_g, h_g, K, bias=False)

    def forward(self, x, h_g, A):
        i = torch.sigmoid(self.i_x(x, A) + self.i_h(h_g, A))
        f = torch.sigmoid(self.f_x(x, A) + self.f_h(h_g, A))
        o = torch.sigmoid(self.o_x(x, A) + self.o_h(h_g, A))
        g = torch.tanh(self.g_x(x, A) + self.g_h(h_g, A))
        return i, f, o, g


class DyGCNACLSTMCell(nn.Module):
    def __init__(self, c_in, hidden=64, G=4, K=3):
        super().__init__()
        assert hidden % G == 0, "hidden must be divisible by G"
        self.G, self.h_g = G, hidden // G
        self.periods = PERIODS[:G]
        self.groups = nn.ModuleList([GroupGates(c_in, self.h_g, K) for _ in range(G)])

    def forward(self, x, h, c, A, step):
        """x [B,F,c_in]; h,c [B,F,H]; step is 1-indexed. Returns new h,c.
        Builds outputs by concatenating per-group results (no in-place slice
        assignment) so autograd is clean."""
        h_parts, c_parts = [], []
        for gi, gate in enumerate(self.groups):
            sl = slice(gi * self.h_g, (gi + 1) * self.h_g)
            hg, cg = h[..., sl], c[..., sl]
            if step % self.periods[gi] != 0:
                h_parts.append(hg); c_parts.append(cg)     # inactive: hold state
            else:
                i, f, o, g = gate(x, hg, A)
                cg_new = f * cg + i * g
                h_parts.append(o * torch.tanh(cg_new)); c_parts.append(cg_new)
        return torch.cat(h_parts, dim=-1), torch.cat(c_parts, dim=-1)


class DyGCNACLSTM(nn.Module):
    """Two-layer encoder + two-layer autoregressive decoder over dynamic adjacency."""
    def __init__(self, hidden=64, G=4, K=3, horizon=12):
        super().__init__()
        self.hidden, self.horizon = hidden, horizon
        self.enc1 = DyGCNACLSTMCell(1, hidden, G, K)
        self.enc2 = DyGCNACLSTMCell(hidden, hidden, G, K)
        self.dec1 = DyGCNACLSTMCell(1, hidden, G, K)
        self.dec2 = DyGCNACLSTMCell(hidden, hidden, G, K)
        self.proj = nn.Linear(hidden, 1)

    def _zero(self, B, F, device):
        z = torch.zeros(B, F, self.hidden, device=device)
        return z, z.clone()

    def forward(self, x_seq, adj_list, y_seq=None, teacher_forcing=0.0):
        """x_seq [B, L, F]; adj_list: list of [F,F] (one per encoder step, len L,
        plus decoder reuses the last). y_seq [B, H, F] for teacher forcing (train)."""
        B, L, F = x_seq.shape
        dev = x_seq.device
        h1, c1 = self._zero(B, F, dev); h2, c2 = self._zero(B, F, dev)
        for t in range(L):
            xt = x_seq[:, t].unsqueeze(-1)                 # [B,F,1]
            A = adj_list[t]
            h1, c1 = self.enc1(xt, h1, c1, A, t + 1)
            h2, c2 = self.enc2(h1, h2, c2, A, t + 1)
        # decoder: autoregressive
        A = adj_list[-1]
        outs = []
        prev = x_seq[:, -1].unsqueeze(-1)                  # last observed [B,F,1]
        for k in range(self.horizon):
            h1, c1 = self.dec1(prev, h1, c1, A, L + k + 1)
            h2, c2 = self.dec2(h1, h2, c2, A, L + k + 1)
            yhat = self.proj(h2).squeeze(-1)               # [B,F]
            outs.append(yhat)
            if self.training and y_seq is not None and torch.rand(1).item() < teacher_forcing:
                prev = y_seq[:, k].unsqueeze(-1)
            else:
                prev = yhat.unsqueeze(-1).detach()
        return torch.stack(outs, dim=1)                    # [B, H, F]
