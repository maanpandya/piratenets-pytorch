import math

import torch

from piratenets.causal import causal_weights, residual_loss


def test_causal_weights_properties():
    losses = torch.tensor([0.5, 0.1, 2.0, 0.0, 0.3], requires_grad=True)
    w = causal_weights(losses, tol=1.0)
    assert not w.requires_grad and w.grad_fn is None
    assert torch.all((w >= 0) & (w <= 1))
    assert torch.all(w[1:] <= w[:-1])
    assert w[0] == 1.0
    expected = [1.0, math.exp(-0.5), math.exp(-0.6), math.exp(-2.6), math.exp(-2.6)]
    torch.testing.assert_close(w, torch.tensor(expected))


def test_causal_weights_non_increasing_for_random_losses():
    g = torch.Generator().manual_seed(0)
    for tol in (0.1, 1.0, 100.0):
        w = causal_weights(torch.rand(32, generator=g) * 10, tol)
        assert torch.all(w[1:] <= w[:-1]) and torch.all((w >= 0) & (w <= 1))


def test_residual_loss_gradient_treats_weights_as_constants():
    losses = torch.tensor([0.5, 0.1, 2.0], requires_grad=True)
    loss, w = residual_loss(losses, tol=1.0, enabled=True)
    loss.backward()
    torch.testing.assert_close(losses.grad, w / 3)


def test_disabled_gives_plain_mean():
    losses = torch.tensor([0.5, 0.1, 2.0])
    loss, w = residual_loss(losses, tol=1.0, enabled=False)
    assert torch.all(w == 1)
    torch.testing.assert_close(loss, losses.mean())
