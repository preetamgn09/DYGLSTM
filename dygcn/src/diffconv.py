"""Bidirectional diffusion graph convolution (DCRNN-style).

DiffConv(X, A): X [B, F, C_in], A [F, F] (symmetric-normalised) -> [B, F, C_out].
Supports: identity + forward powers of A (out-neighbours) + backward powers of A^T
(in-neighbours), up to K-1 hops. A single linear layer mixes the stacked supports.
"""
import torch
import torch.nn as nn


class DiffConv(nn.Module):
    def __init__(self, c_in, c_out, K=3, bias=True):
        super().__init__()
        self.K = K
        n_support = 1 + 2 * (K - 1)          # identity + fwd(1..K-1) + bwd(1..K-1)
        self.lin = nn.Linear(n_support * c_in, c_out, bias=bias)

    def forward(self, X, A):
        # X: [B, F, C_in]; A: [F, F]
        supports = [X]
        xk = X
        for _ in range(1, self.K):                       # forward hops
            xk = torch.einsum("fg,bgc->bfc", A, xk)
            supports.append(xk)
        At = A.transpose(0, 1)
        xk = X
        for _ in range(1, self.K):                       # backward hops
            xk = torch.einsum("fg,bgc->bfc", At, xk)
            supports.append(xk)
        h = torch.cat(supports, dim=-1)                  # [B, F, n_support*C_in]
        return self.lin(h)
