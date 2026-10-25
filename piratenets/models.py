"""Network architectures. All models map (t, x), each of shape (N,), to u of shape (N,)."""
import torch
from torch import nn

from .embeddings import FourierFeatures, Identity2D, PeriodicEmbedding
from .layers import Dense

ACTIVATIONS = {
    "tanh": torch.tanh,
    "sin": torch.sin,
    "gelu": nn.functional.gelu,
    "swish": nn.functional.silu,
}


class MLP(nn.Module):
    """Plain MLP (Expert's Guide eqs. 4.1-4.2) behind optional periodic and Fourier embeddings."""

    def __init__(self, cfg, x_period: float):
        super().__init__()
        if cfg.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {cfg.activation!r}; choose from {sorted(ACTIVATIONS)}")
        self.act = ACTIVATIONS[cfg.activation]
        self.coords = PeriodicEmbedding(x_period) if cfg.periodic else Identity2D()
        dim = self.coords.out_dim
        self.fourier = None
        if cfg.fourier.enabled:
            self.fourier = FourierFeatures(
                dim, cfg.fourier.embed_dim, cfg.fourier.scale, cfg.fourier.trainable
            )
            dim = self.fourier.out_dim
        dims = [dim] + [cfg.hidden_dim] * cfg.num_layers
        self.hidden = nn.ModuleList(Dense(a, b, cfg.rwf) for a, b in zip(dims[:-1], dims[1:]))
        self.out = Dense(dims[-1], 1, cfg.rwf)

    def embed(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.coords(t, x)
        return self.fourier(h) if self.fourier is not None else h

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(t, x)
        for layer in self.hidden:
            h = self.act(layer(h))
        return self.out(h).squeeze(-1)


def build_model(cfg, x_period: float) -> nn.Module:
    if cfg.arch == "mlp":
        return MLP(cfg, x_period)
    raise ValueError(f"unknown architecture {cfg.arch!r}")
