"""Global loss weights (Expert's Guide Algorithm 1, steps 4 and 5c-d; Sec. 5.2).

Grad-norm scheme, every f steps:

    lambda_hat_k = R_j ||grad L_j|| / ||grad L_k||                       (2.12-2.14)
    lambda_k <- alpha * lambda_k + (1 - alpha) * lambda_hat_k            (2.15)

where R is the mean over the loss terms (as in jaxpi, the default) or their sum
(as printed in eqs. 2.12-2.14); see docs/decisions.md, D20.

All weights start at 1 and are plain floats, so they never carry gradients.
"""
import torch


def grad_norm(loss: torch.Tensor, params) -> float:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = sum((g.detach() ** 2).sum() for g in grads if g is not None)
    return float(torch.sqrt(sq))


class LossWeighting:
    SCHEMES = ("none", "grad_norm")
    REDUCTIONS = ("mean", "sum")

    def __init__(self, names, scheme: str, update_every: int, momentum: float, reduction: str = "mean"):
        if scheme not in self.SCHEMES:
            raise ValueError(f"unknown weighting scheme {scheme!r}; choose from {self.SCHEMES}")
        if reduction not in self.REDUCTIONS:
            raise ValueError(f"unknown grad-norm reduction {reduction!r}; choose from {self.REDUCTIONS}")
        self.scheme = scheme
        self.reduction = reduction
        self.update_every = update_every
        self.momentum = momentum
        self.lambdas = {name: 1.0 for name in names}
        self.grad_norms = {name: float("nan") for name in names}

    def due(self, step: int) -> bool:
        return self.scheme != "none" and step % self.update_every == 0

    def update(self, losses: dict, params) -> None:
        """Recompute the weights from the gradient norms of the (unweighted) losses."""
        params = [p for p in params if p.requires_grad]
        norms = {name: grad_norm(losses[name], params) for name in self.lambdas}
        total = sum(norms.values())
        if self.reduction == "mean":
            total /= len(norms)
        for name, norm in norms.items():
            target = total / max(norm, 1e-30)
            self.lambdas[name] = self.momentum * self.lambdas[name] + (1.0 - self.momentum) * target
        self.grad_norms = norms

    def total_loss(self, losses: dict) -> torch.Tensor:
        return sum(self.lambdas[name] * losses[name] for name in self.lambdas)

    def state_dict(self) -> dict:
        return {"lambdas": dict(self.lambdas), "grad_norms": dict(self.grad_norms)}

    def load_state_dict(self, state: dict) -> None:
        self.lambdas = dict(state["lambdas"])
        self.grad_norms = dict(state["grad_norms"])
