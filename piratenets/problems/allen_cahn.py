"""1D Allen-Cahn benchmark (Expert's Guide Sec. 7.1, PirateNets Sec. 3 and 5.1).

    u_t - 1e-4 u_xx + 5 u^3 - 5 u = 0,   t in [0, 1],  x in [-1, 1] periodic,
    u(0, x) = x^2 cos(pi x).
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.func import grad_and_value, jvp, vmap

from ..config import resolve_path

DIFFUSIVITY = 1e-4
REACTION = 5.0
T_MAX = 1.0
X_MIN, X_MAX = -1.0, 1.0
X_PERIOD = X_MAX - X_MIN


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    return x**2 * torch.cos(torch.pi * x)


def residual(u, u_t, u_xx):
    return u_t - DIFFUSIVITY * u_xx + REACTION * u**3 - REACTION * u


def derivatives(u_fn, t: torch.Tensor, x: torch.Tensor, mode: str = "func"):
    """Return u, u_t, u_x, u_xx at the points (t_i, x_i).

    u_fn(t, x) maps tensors of shape (N,) to (N,) and must act on each point
    independently. The results stay differentiable with respect to the model
    parameters.

    mode="func": per-point torch.func under vmap. (u_t, u_x) by reverse mode
        (grad), u_xx by forward mode over that gradient (jvp along x).
    mode="autograd": torch.autograd.grad on the batch sum, create_graph=True.
        Same values; kept because it can be faster on some devices
        (scripts/benchmark_step.py).
    """
    if mode == "func":
        def point(ti, xi):
            def u_at(tj, xj):
                return u_fn(tj.reshape(1), xj.reshape(1)).reshape(())

            def first(xj):
                (u_t, u_x), u = grad_and_value(u_at, argnums=(0, 1))(ti, xj)
                return u, u_t, u_x

            (u, u_t, u_x), (_, _, u_xx) = jvp(first, (xi,), (torch.ones_like(xi),))
            return u, u_t, u_x, u_xx

        return vmap(point)(t, x)

    if mode == "autograd":
        t = t.detach().requires_grad_(True)
        x = x.detach().requires_grad_(True)
        u = u_fn(t, x)
        u_t, u_x = torch.autograd.grad(u.sum(), (t, x), create_graph=True)
        (u_xx,) = torch.autograd.grad(u_x.sum(), x, create_graph=True)
        return u, u_t, u_x, u_xx

    raise ValueError(f"unknown derivative mode {mode!r}")


def pde_residual(model, t, x, mode: str = "func"):
    u, u_t, _, u_xx = derivatives(model, t, x, mode)
    return residual(u, u_t, u_xx)


def sample_collocation(n: int, num_chunks: int, generator: torch.Generator, device):
    """n uniform samples in [0, T] x [-1, 1), grouped into causal chunks by time.

    Returns t, x of shape (num_chunks, n // num_chunks). The times are sorted, so
    row i holds the i-th group of n // num_chunks earliest points: the chunks
    have equal counts and their boundaries fall at the sample quantiles of t
    (docs/decisions.md, D10).
    """
    if n % num_chunks:
        raise ValueError(f"batch size {n} must be divisible by num_chunks {num_chunks}")
    t = T_MAX * torch.rand(n, generator=generator, device=device)
    x = X_MIN + X_PERIOD * torch.rand(n, generator=generator, device=device)
    t, order = torch.sort(t)
    return t.reshape(num_chunks, -1), x[order].reshape(num_chunks, -1)


def sample_initial(n: int, generator: torch.Generator, device):
    return X_MIN + X_PERIOD * torch.rand(n, generator=generator, device=device)


@dataclass
class Reference:
    t: np.ndarray      # (nt,)
    x: np.ndarray      # (nx,)
    usol: np.ndarray   # (nt, nx)


def load_reference(path="data/allen_cahn_ref.npz") -> Reference:
    with np.load(resolve_path(str(path))) as d:
        return Reference(t=d["t"], x=d["x"], usol=d["usol"])


@torch.no_grad()
def predict_grid(model, ref: Reference, device, chunk: int = 32768) -> np.ndarray:
    tt, xx = np.meshgrid(ref.t, ref.x, indexing="ij")
    t = torch.as_tensor(tt.ravel(), dtype=torch.float32, device=device)
    x = torch.as_tensor(xx.ravel(), dtype=torch.float32, device=device)
    out = torch.cat([model(t[i:i + chunk], x[i:i + chunk]) for i in range(0, t.numel(), chunk)])
    return out.reshape(tt.shape).double().cpu().numpy()


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
