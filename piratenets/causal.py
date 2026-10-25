"""Causal weighting of the PDE residual (Expert's Guide Sec. 5.1, eqs. 2.10-2.11, 5.2).

The time domain is split into M equal chunks with mean residual losses
L_1, ..., L_M. Then

    w_1 = 1,   w_i = exp(-eps * sum_{k<i} L_k),   L_r = (1/M) sum_i w_i L_i,

and the weights are excluded from back-propagation (stop-gradient).
"""
import torch


def causal_weights(chunk_losses: torch.Tensor, tol: float) -> torch.Tensor:
    losses = chunk_losses.detach()
    previous = torch.cumsum(losses, dim=0) - losses   # sum_{k<i} L_k
    return torch.exp(-tol * previous)


def residual_loss(chunk_losses: torch.Tensor, tol: float, enabled: bool):
    """Return (L_r, w). With causal weighting disabled, w = 1 and L_r is the plain mean."""
    if enabled:
        w = causal_weights(chunk_losses, tol)
    else:
        w = torch.ones_like(chunk_losses).detach()
    return (w * chunk_losses).mean(), w
