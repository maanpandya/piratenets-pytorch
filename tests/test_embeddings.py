import torch

from piratenets.config import ModelConfig
from piratenets.embeddings import FourierFeatures
from piratenets.models import build_model
from piratenets.problems import allen_cahn as ac


def _model(fourier=True, rwf=True):
    torch.manual_seed(0)
    cfg = ModelConfig(hidden_dim=64, num_layers=3)
    cfg.fourier.enabled = fourier
    cfg.rwf.enabled = rwf
    return build_model(cfg, x_period=ac.X_PERIOD)


def test_periodic_embedding_value_and_slope_match_at_boundaries():
    for fourier in (False, True):
        model = _model(fourier=fourier)
        t = torch.linspace(0, 1, 101)
        left, right = -torch.ones_like(t), torch.ones_like(t)
        u_l, _, ux_l, uxx_l = ac.derivatives(model, t, left)
        u_r, _, ux_r, uxx_r = ac.derivatives(model, t, right)
        # float32 pi is inexact, so sin(-pi) != -sin(pi) at the 1e-7 level
        torch.testing.assert_close(u_l, u_r, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ux_l, ux_r, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(uxx_l, uxx_r, rtol=1e-4, atol=1e-4)


def test_periodic_embedding_exact_in_float64():
    model = _model().double()
    t = torch.linspace(0, 1, 11, dtype=torch.float64)
    x = torch.linspace(-1, 1, 11, dtype=torch.float64)
    torch.testing.assert_close(model(t, x), model(t, x + 2.0), rtol=0, atol=1e-12)


def test_fourier_features_output():
    torch.manual_seed(0)
    ff = FourierFeatures(in_dim=3, embed_dim=8, scale=2.0)
    assert ff.B.shape == (4, 3)
    v = torch.randn(5, 3)
    out = ff(v)
    assert out.shape == (5, 8)
    torch.testing.assert_close(out[:, :4] ** 2 + out[:, 4:] ** 2, torch.ones(5, 4))
    torch.testing.assert_close(out[:, :4], torch.cos(v @ ff.B.T))


def test_fourier_matrix_trainable_or_fixed():
    trained = FourierFeatures(in_dim=3, embed_dim=8, scale=2.0, trainable=True)
    assert [n for n, _ in trained.named_parameters()] == ["B"]
    fixed = FourierFeatures(in_dim=3, embed_dim=8, scale=2.0, trainable=False)
    assert list(fixed.parameters()) == [] and "B" in fixed.state_dict()


def test_fourier_scale_controls_frequency_spread():
    torch.manual_seed(0)
    ff = FourierFeatures(in_dim=3, embed_dim=20000, scale=2.0)
    assert abs(ff.B.std().item() - 2.0) < 0.05
