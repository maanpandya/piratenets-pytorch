import pytest
import torch

from piratenets.weighting import LossWeighting


def _losses():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(10))
    return p, {"ic": (3.0 * p).pow(2).sum(), "r": p.sum()}


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_grad_norm_formula_and_moving_average(reduction):
    p, losses = _losses()
    g_ic = torch.autograd.grad(losses["ic"], p, retain_graph=True)[0].norm().item()
    g_r = torch.autograd.grad(losses["r"], p, retain_graph=True)[0].norm().item()
    w = LossWeighting(["ic", "r"], "grad_norm", update_every=10, momentum=0.9, reduction=reduction)
    w.update(losses, [p])
    total = (g_ic + g_r) / (2 if reduction == "mean" else 1)
    assert abs(w.lambdas["ic"] - (0.9 + 0.1 * total / g_ic)) < 1e-9
    assert abs(w.lambdas["r"] - (0.9 + 0.1 * total / g_r)) < 1e-9
    assert isinstance(w.lambdas["ic"], float)
    assert p.grad is None  # the update does not touch .grad


def test_default_reduction_is_mean_and_unknown_is_rejected():
    assert LossWeighting(["ic", "r"], "grad_norm", 1000, 0.9).reduction == "mean"
    with pytest.raises(ValueError):
        LossWeighting(["ic", "r"], "grad_norm", 1000, 0.9, reduction="median")


def test_schedule_and_none_scheme():
    w = LossWeighting(["ic", "r"], "grad_norm", update_every=1000, momentum=0.9)
    assert [s for s in range(3001) if w.due(s)] == [0, 1000, 2000, 3000]
    none = LossWeighting(["ic", "r"], "none", update_every=1000, momentum=0.9)
    assert not any(none.due(s) for s in range(3001))
    p, losses = _losses()
    torch.testing.assert_close(none.total_loss(losses), losses["ic"] + losses["r"])


def test_state_dict_roundtrip():
    p, losses = _losses()
    w = LossWeighting(["ic", "r"], "grad_norm", update_every=1, momentum=0.5)
    w.update(losses, [p])
    w2 = LossWeighting(["ic", "r"], "grad_norm", update_every=1, momentum=0.5)
    w2.load_state_dict(w.state_dict())
    assert w2.lambdas == w.lambdas
