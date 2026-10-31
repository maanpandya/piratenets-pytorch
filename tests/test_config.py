from pathlib import Path

import pytest

from piratenets.config import (
    REPO_ROOT, Config, config_from_dict, config_to_dict, load_config, saved_config_from_dict,
)

ABLATION = REPO_ROOT / "configs" / "allen_cahn" / "ablation"

# Table 2 row -> the only settings allowed to differ from the full pipeline.
EXPECTED_DIFFS = {
    "default.yaml": {},
    "no_fourier.yaml": {"model.fourier.enabled": False},
    "no_rwf.yaml": {"model.rwf.enabled": False},
    "no_grad_norm.yaml": {"weighting.scheme": "none"},
    "no_causal.yaml": {"causal.enabled": False},
    "plain.yaml": {
        "model.fourier.enabled": False,
        "model.rwf.enabled": False,
        "weighting.scheme": "none",
        "causal.enabled": False,
    },
}


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_ablation_rows_differ_only_in_their_switches():
    assert sorted(p.name for p in ABLATION.glob("*.yaml")) == sorted(EXPECTED_DIFFS)
    base = _flatten(config_to_dict(load_config(ABLATION / "default.yaml")))
    for fname, expected in EXPECTED_DIFFS.items():
        cfg = _flatten(config_to_dict(load_config(ABLATION / fname)))
        diff = {k: v for k, v in cfg.items() if base[k] != v and k != "name"}
        assert diff == expected, fname


def test_default_row_matches_paper_and_jaxpi_config():
    cfg = load_config(ABLATION / "default.yaml")
    # EG Sec. 7 and Algorithm 1
    assert (cfg.model.arch, cfg.model.num_layers, cfg.model.hidden_dim, cfg.model.activation) == ("mlp", 4, 256, "tanh")
    assert cfg.optim.lr == 1e-3 and cfg.optim.decay_rate == 0.9 and cfg.optim.decay_steps == 2000
    assert cfg.training.batch_size == 4096
    assert cfg.weighting.update_every == 1000 and cfg.weighting.momentum == 0.9
    assert cfg.causal.tol == 1.0
    # jaxpi examples/allen_cahn/configs/default.py (docs/decisions.md D2, D3, D4, D6, D12)
    assert cfg.training.steps == 200_000
    assert cfg.model.fourier.scale == 1.0 and cfg.model.fourier.embed_dim == 256
    assert cfg.model.rwf.mu == 0.5 and cfg.model.rwf.sigma == 0.1
    assert cfg.causal.num_chunks == 32
    # design decisions taken from jaxpi's code (D5, D9, D11, D20, D21)
    assert cfg.model.fourier.trainable is True
    assert cfg.weighting.grad_norm_reduction == "mean"
    assert cfg.model.rwf.scale_param == "direct"
    assert cfg.optim.staircase is False and cfg.optim.warmup_steps == 0
    assert cfg.training.ic_sampling == "grid"


def test_overrides_and_strict_keys(tmp_path):
    cfg = load_config(ABLATION / "default.yaml", ["training.steps=500", "optim.lr=1e-4", "causal.enabled=false"])
    assert cfg.training.steps == 500 and cfg.optim.lr == 1e-4 and cfg.causal.enabled is False
    with pytest.raises(KeyError):
        load_config(ABLATION / "default.yaml", ["model.fourier.sclae=3.0"])
    with pytest.raises(TypeError):
        load_config(ABLATION / "default.yaml", ["causal.enabled=maybe"])


def test_saved_config_without_scale_param_reads_as_log():
    # Runs saved before rwf.scale_param existed trained s (D21).
    data = config_to_dict(Config())
    del data["model"]["rwf"]["scale_param"]
    assert config_from_dict(data).model.rwf.scale_param == "direct"
    assert saved_config_from_dict(data).model.rwf.scale_param == "log"
    assert "scale_param" not in data["model"]["rwf"]  # the input is left alone


def test_paths_resolve_relative_to_repo(tmp_path):
    cfg = load_config(ABLATION / "default.yaml")
    assert cfg.out_dir == REPO_ROOT / "runs/allen_cahn/ablation/ablation_default"
    assert cfg.checkpoint_dir == cfg.out_dir / "checkpoints"
    cfg.logging.checkpoint_dir = str(tmp_path / "ck" / "{name}")
    assert cfg.checkpoint_dir == Path(tmp_path / "ck" / "ablation_default")
