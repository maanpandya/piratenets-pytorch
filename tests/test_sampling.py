import torch

from piratenets.problems import allen_cahn as ac
from piratenets.trainer import Trainer


def test_collocation_points_are_uniform_and_chunked_by_sorted_time():
    g = torch.Generator().manual_seed(0)
    t, x = ac.sample_collocation(4096, 32, g, "cpu")
    assert t.shape == x.shape == (32, 128)
    flat = t.reshape(-1)
    assert torch.all(flat[1:] >= flat[:-1])                 # chunk i holds the i-th group of earliest times
    assert t.min() >= 0 and t.max() <= ac.T_MAX
    assert x.min() >= ac.X_MIN and x.max() < ac.X_MAX
    # still uniform: chunk boundaries sit near the quantiles i / M
    assert abs(t[:, -1].mean() - t[:, -1].new_tensor([(i + 1) / 32 for i in range(32)]).mean()) < 0.02


def test_x_is_paired_with_its_own_time_after_sorting():
    g = torch.Generator().manual_seed(1)
    t, x = ac.sample_collocation(64, 4, g, "cpu")
    g = torch.Generator().manual_seed(1)
    t_raw = ac.T_MAX * torch.rand(64, generator=g)
    x_raw = ac.X_MIN + ac.X_PERIOD * torch.rand(64, generator=g)
    pairs = {(round(a, 6), round(b, 6)) for a, b in zip(t_raw.tolist(), x_raw.tolist())}
    assert {(round(a, 6), round(b, 6)) for a, b in zip(t.reshape(-1).tolist(), x.reshape(-1).tolist())} == pairs


def test_ic_points_grid_or_random(tiny_cfg):
    trainer = Trainer(tiny_cfg)
    _, _, x_ic = trainer.sample_batch()
    torch.testing.assert_close(x_ic.double(), torch.as_tensor(trainer.reference.x), rtol=0, atol=1e-7)
    tiny_cfg.training.ic_sampling = "random"
    trainer = Trainer(tiny_cfg)
    _, _, x_a = trainer.sample_batch()
    _, _, x_b = trainer.sample_batch()
    assert x_a.shape == (tiny_cfg.training.n_ic,) and not torch.equal(x_a, x_b)


def test_loss_weights_update_after_the_optimizer_step(tiny_cfg):
    """Step 0 uses lambda = 1; the weights computed at step 0 apply from step 1 on."""
    trainer = Trainer(tiny_cfg)
    _, _, _, updated = trainer.train_step()
    assert updated and trainer.weighting.lambdas["ic"] != 1.0
    # recomputing at the post-step parameters on the same batch must reproduce the update
    ref = Trainer(tiny_cfg)
    batch = ref.sample_batch()
    losses, _ = ref.compute_losses(batch)
    ref.weighting.total_loss(losses).backward()
    ref.optimizer.step()
    new_losses, _ = ref.compute_losses(batch)
    ref.weighting.update(new_losses, ref.model.parameters())
    assert ref.weighting.lambdas == trainer.weighting.lambdas
