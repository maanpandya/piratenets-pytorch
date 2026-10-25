import pytest
import torch

from piratenets.config import Config

torch.set_float32_matmul_precision("highest")


@pytest.fixture
def tiny_cfg(tmp_path):
    """A small, fast configuration with every pipeline component switched on."""
    cfg = Config(name="tiny", device="cpu")
    cfg.model.hidden_dim = 16
    cfg.model.num_layers = 2
    cfg.model.fourier.embed_dim = 16
    cfg.training.steps = 10
    cfg.training.batch_size = 64
    cfg.training.n_ic = 32
    cfg.weighting.update_every = 3
    cfg.causal.num_chunks = 4
    cfg.logging.out_dir = str(tmp_path / "run")
    cfg.logging.log_every = 1
    cfg.logging.eval_every = 5
    cfg.logging.checkpoint_every = 5
    return cfg
