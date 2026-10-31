"""Dense layers, optionally with random weight factorization (RWF).

RWF (Expert's Guide Sec. 4.3 and Algorithm 2): every weight matrix is written as
W = diag(exp(s)) V, one scale per output neuron. W is first initialized with the
Glorot scheme, s ~ N(mu, sigma^2), and V = W / exp(s), so the network function at
initialization is unchanged. With ``scale_param: log`` gradient descent then acts
on (s, V, b). With ``scale_param: direct`` (the default) it acts on (g, V, b),
where g = exp(s) is stored and trained as is (as in jaxpi; docs/decisions.md,
D21). The network function is the same, but the gradients and Adam's steps on
the scales differ.

Glorot initialization uses a normal distribution truncated at two standard
deviations and rescaled so that Var(W) = 2 / (fan_in + fan_out) still holds
(docs/decisions.md, D8). Biases start at zero.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

# Standard deviation of a unit normal truncated to [-2, 2].
_TRUNC_STD = 0.87962566103423978


def glorot_truncated_normal_(weight: torch.Tensor) -> torch.Tensor:
    fan_out, fan_in = weight.shape
    std = math.sqrt(2.0 / (fan_in + fan_out)) / _TRUNC_STD
    return nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


class Dense(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rwf=None):
        """rwf: an object with ``enabled``, ``mu``, ``sigma`` (RWFConfig) or None."""
        super().__init__()
        weight = torch.empty(out_dim, in_dim)
        glorot_truncated_normal_(weight)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.factorized = bool(rwf is not None and rwf.enabled)
        if self.factorized:
            if rwf.scale_param not in ("log", "direct"):
                raise ValueError(f"unknown rwf.scale_param {rwf.scale_param!r}; use 'log' or 'direct'")
            self.log_scale = rwf.scale_param == "log"
            s = rwf.mu + rwf.sigma * torch.randn(out_dim)
            if self.log_scale:
                self.s = nn.Parameter(s)
            else:
                self.g = nn.Parameter(torch.exp(s))
            self.v = nn.Parameter(weight / torch.exp(s)[:, None])
        else:
            self.weight = nn.Parameter(weight)

    def weight_matrix(self) -> torch.Tensor:
        if self.factorized:
            scale = torch.exp(self.s) if self.log_scale else self.g
            return scale[:, None] * self.v
        return self.weight

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.weight_matrix(), self.bias)
