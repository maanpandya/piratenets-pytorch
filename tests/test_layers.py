import pytest
import torch
from torch import nn

from piratenets.config import RWFConfig
from piratenets.layers import Dense


def test_rwf_dense_equals_linear_with_recombined_weight():
    torch.manual_seed(0)
    layer = Dense(7, 5, RWFConfig(enabled=True, mu=1.0, sigma=0.1, scale_param="log"))
    with torch.no_grad():
        layer.bias.normal_()
    ref = nn.Linear(7, 5)
    with torch.no_grad():
        ref.weight.copy_(torch.exp(layer.s)[:, None] * layer.v)
        ref.bias.copy_(layer.bias)
    h = torch.randn(11, 7)
    torch.testing.assert_close(layer(h), ref(h), rtol=0, atol=0)


def test_rwf_parameters_and_initialization():
    torch.manual_seed(0)
    plain = Dense(64, 32, None)
    torch.manual_seed(0)
    fact = Dense(64, 32, RWFConfig(enabled=True, mu=1.0, sigma=0.1, scale_param="log"))
    names = {n for n, _ in fact.named_parameters()}
    assert names == {"s", "v", "bias"}
    assert {n for n, _ in plain.named_parameters()} == {"weight", "bias"}
    # The factorization does not change the network at initialization: W = diag(exp(s)) V
    torch.testing.assert_close(fact.weight_matrix(), plain.weight_matrix(), rtol=1e-6, atol=1e-7)
    assert fact.s.shape == (32,)  # one scale per output neuron
    assert abs(fact.s.mean().item() - 1.0) < 0.1
    assert torch.all(fact.bias == 0)


def test_rwf_direct_scale_same_function_different_gradients():
    torch.manual_seed(0)
    log = Dense(64, 32, RWFConfig(enabled=True, mu=0.5, sigma=0.1, scale_param="log"))
    torch.manual_seed(0)
    direct = Dense(64, 32, RWFConfig(enabled=True, mu=0.5, sigma=0.1, scale_param="direct"))
    assert {n for n, _ in direct.named_parameters()} == {"g", "v", "bias"}
    torch.testing.assert_close(direct.g, torch.exp(log.s), rtol=0, atol=0)
    torch.testing.assert_close(direct.weight_matrix(), log.weight_matrix(), rtol=0, atol=0)
    # Same function, but dL/dg = (dL/ds) / g.
    h = torch.randn(11, 64)
    log(h).pow(2).sum().backward()
    direct(h).pow(2).sum().backward()
    torch.testing.assert_close(direct.g.grad, log.s.grad / direct.g.detach(), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(direct.v.grad, log.v.grad, rtol=0, atol=0)


def test_rwf_rejects_an_unknown_scale_param():
    with pytest.raises(ValueError, match="scale_param"):
        Dense(4, 4, RWFConfig(enabled=True, scale_param="exp"))


def test_glorot_truncated_normal():
    torch.manual_seed(0)
    w = Dense(256, 256, None).weight_matrix()
    target = 2.0 / 512
    assert abs(w.var().item() - target) < 0.05 * target
    # truncated at two standard deviations of the underlying (rescaled) normal
    assert w.abs().max().item() <= 2.0 * (target**0.5 / 0.87962566103423978) + 1e-7
