"""Bidirectional diffusion graph convolution (DCRNN-style), BATCHED.

DiffConv(X, A): X [B, F, C_in], A [B, F, F] (per-sample adjacency) -> [B, F, C_out].
Batching the adjacency over the B dimension lets a whole minibatch run in one set
of matmuls (and on GPU), instead of looping window-by-window.
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
        # X: [B, F, C_in]; A: [B, F, F]
        supports = [X]
        xk = X
        for _ in range(1, self.K):                          # forward hops
            xk = torch.einsum("bfg,bgc->bfc", A, xk)
            supports.append(xk)
        At = A.transpose(1, 2)
        xk = X
        for _ in range(1, self.K):                          # backward hops
            xk = torch.einsum("bfg,bgc->bfc", At, xk)
            supports.append(xk)
        return self.lin(torch.cat(supports, dim=-1))
