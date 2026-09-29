"""Temporal-only baselines: LSTM (NeuTM-style) and AC-LSTM (clockwork, linear gates).
Both treat the F OD flows as a flat feature vector (no graph). AC-LSTM is exactly
the proposed model with DiffConv replaced by linear gates, so Variant-B ablation
== AC-LSTM (subject to the capacity note in the README)."""
import torch
import torch.nn as nn

PERIODS = [1, 2, 4, 8]


class LSTMBaseline(nn.Module):
    def __init__(self, F, hidden=64, horizon=12, layers=2):
        super().__init__()
        self.horizon, self.F = horizon, F
        self.lstm = nn.LSTM(F, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon * F)

    def forward(self, x_seq, *_, **__):
        out, _ = self.lstm(x_seq)                       # [B, L, H]
        y = self.head(out[:, -1])                       # [B, horizon*F]
        return y.view(-1, self.horizon, self.F)


class ClockworkLSTMCell(nn.Module):
    """Linear-gate clockwork LSTM on a flat feature vector [B, F]."""
    def __init__(self, in_dim, hidden=64, G=4):
        super().__init__()
        assert hidden % G == 0
        self.G, self.h_g, self.hidden = G, hidden // G, hidden
        self.periods = PERIODS[:G]
        # group-specific gate params: per group, W_x [in,4*h_g], W_h [h_g? -> use full h]
        self.x2h = nn.ModuleList([nn.Linear(in_dim, 4 * self.h_g) for _ in range(G)])
        self.h2h = nn.ModuleList([nn.Linear(hidden, 4 * self.h_g) for _ in range(G)])

    def forward(self, x, h, c, step):
        h_parts, c_parts = [], []
        for gi in range(self.G):
            sl = slice(gi * self.h_g, (gi + 1) * self.h_g)
            if step % self.periods[gi] != 0:
                h_parts.append(h[..., sl]); c_parts.append(c[..., sl]); continue
            z = self.x2h[gi](x) + self.h2h[gi](h)
            i, f, o, g = z.chunk(4, dim=-1)
            cg = torch.sigmoid(f) * c[..., sl] + torch.sigmoid(i) * torch.tanh(g)
            hg = torch.sigmoid(o) * torch.tanh(cg)
            h_parts.append(hg); c_parts.append(cg)
        return torch.cat(h_parts, dim=-1), torch.cat(c_parts, dim=-1)


class ACLSTMBaseline(nn.Module):
    def __init__(self, F, hidden=64, G=4, horizon=12):
        super().__init__()
        self.F, self.hidden, self.horizon = F, hidden, horizon
        self.enc = ClockworkLSTMCell(F, hidden, G)
        self.dec = ClockworkLSTMCell(F, hidden, G)
        self.proj = nn.Linear(hidden, F)

    def forward(self, x_seq, *_, y_seq=None, teacher_forcing=0.0, **__):
        B, L, F = x_seq.shape
        h = torch.zeros(B, self.hidden, device=x_seq.device); c = h.clone()
        for t in range(L):
            h, c = self.enc(x_seq[:, t], h, c, t + 1)
        outs, prev = [], x_seq[:, -1]
        for k in range(self.horizon):
            h, c = self.dec(prev, h, c, L + k + 1)
            yhat = self.proj(h)
            outs.append(yhat)
            if self.training and y_seq is not None and torch.rand(1).item() < teacher_forcing:
                prev = y_seq[:, k]
            else:
                prev = yhat.detach()
        return torch.stack(outs, dim=1)
