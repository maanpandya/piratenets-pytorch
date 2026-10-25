import pytest
import torch

from piratenets.config import ModelConfig
from piratenets.models import build_model
from piratenets.problems import allen_cahn as ac

MODES = ["func", "autograd"]


def smooth(t, x):
    return torch.sin(torch.pi * x) * torch.exp(-t) + t**2 * x**3


def smooth_exact(t, x):
    u = smooth(t, x)
    u_t = -torch.sin(torch.pi * x) * torch.exp(-t) + 2 * t * x**3
    u_x = torch.pi * torch.cos(torch.pi * x) * torch.exp(-t) + 3 * t**2 * x**2
    u_xx = -torch.pi**2 * torch.sin(torch.pi * x) * torch.exp(-t) + 6 * t**2 * x
    return u, u_t, u_x, u_xx


def finite_differences(fn, t, x, h_t=1e-5, h_x=1e-4):
    u = fn(t, x)
    u_t = (fn(t + h_t, x) - fn(t - h_t, x)) / (2 * h_t)
    u_x = (fn(t, x + h_x) - fn(t, x - h_x)) / (2 * h_x)
    u_xx = (fn(t, x + h_x) - 2 * u + fn(t, x - h_x)) / h_x**2
    return u, u_t, u_x, u_xx


def points(n=64, dtype=torch.float64):
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, generator=g, dtype=dtype), 2 * torch.rand(n, generator=g, dtype=dtype) - 1


@pytest.mark.parametrize("mode", MODES)
def test_smooth_function_exact(mode):
    t, x = points()
    for got, want in zip(ac.derivatives(smooth, t, x, mode), smooth_exact(t, x)):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_smooth_function_vs_finite_differences(mode):
    t, x = points()
    for got, want in zip(ac.derivatives(smooth, t, x, mode), finite_differences(smooth, t, x)):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rwf", [False, True])
def test_network_vs_finite_differences(mode, rwf):
    torch.manual_seed(0)
    cfg = ModelConfig(hidden_dim=32, num_layers=3)
    cfg.rwf.enabled = rwf
    model = build_model(cfg, x_period=2.0).double()
    t, x = points()
    got = ac.derivatives(model, t, x, mode)
    want = finite_differences(model, t, x)
    for name, g, w in zip(["u", "u_t", "u_x", "u_xx"], got, want):
        scale = w.abs().max()
        assert (g - w).abs().max() <= 1e-5 * scale, name


def test_modes_agree_in_float32_including_parameter_gradients():
    torch.manual_seed(0)
    model = build_model(ModelConfig(hidden_dim=32, num_layers=3), x_period=2.0)
    t, x = points(dtype=torch.float32)
    grads = []
    for mode in MODES:
        model.zero_grad()
        r = ac.pde_residual(model, t, x, mode)
        (r**2).mean().backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-6)


def test_residual_vanishes_on_spatially_constant_exact_solution():
    """u' = 5u - 5u^3 has u(t)^2 = 1 / (1 + (1/u0^2 - 1) exp(-10 t)); checks the sign convention."""
    u0 = 0.3

    def exact(t, x):
        return (1.0 / (1.0 + (1.0 / u0**2 - 1.0) * torch.exp(-10.0 * t))) ** 0.5 + 0.0 * x

    t, x = points()
    u, u_t, _, u_xx = ac.derivatives(exact, t, x)
    torch.testing.assert_close(ac.residual(u, u_t, u_xx), torch.zeros_like(t), rtol=0, atol=1e-12)
