import csv
import json
import math
import sys

import pytest
import torch

from piratenets.config import OptimConfig, config_to_dict
from piratenets.trainer import Trainer, config_mismatch, disable_power_throttling, lr_factor


def _rows(path):
    with open(path, newline="") as f:
        return [{k: v for k, v in r.items() if k != "wall_time"} for r in csv.DictReader(f)]


def _params(trainer):
    return [p.detach().clone() for p in trainer.model.parameters()]


def test_lr_schedule():
    f = lr_factor(OptimConfig(decay_rate=0.9, decay_steps=2000))
    assert f(0) == 1.0 and math.isclose(f(2000), 0.9) and math.isclose(f(1000), 0.9**0.5)
    f = lr_factor(OptimConfig(decay_rate=0.9, decay_steps=2000, staircase=True))
    assert f(1999) == 1.0 and math.isclose(f(2000), 0.9)
    f = lr_factor(OptimConfig(decay_rate=0.9, decay_steps=5000, warmup_steps=5000))
    assert f(0) == 0.0 and f(2500) == 0.5 and f(5000) == 1.0 and math.isclose(f(10000), 0.9)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
])
def test_resume_reproduces_uninterrupted_run_exactly(tiny_cfg, tmp_path, device):
    tiny_cfg.device = device
    # uninterrupted run
    ref = Trainer(tiny_cfg)
    ref.train(resume=False)
    ref_params, ref_rows = _params(ref), _rows(tiny_cfg.out_dir / "metrics.csv")
    ref_lambdas = dict(ref.weighting.lambdas)

    # stopped after 5 steps, then continued by a fresh trainer (as after a Colab disconnect)
    tiny_cfg.logging.out_dir = str(tmp_path / "run_resumed")
    Trainer(tiny_cfg).train(resume=False, max_steps=5)
    resumed = Trainer(tiny_cfg)
    resumed.train(resume=True)
    assert resumed.step == tiny_cfg.training.steps
    for a, b in zip(ref_params, _params(resumed)):
        assert torch.equal(a, b)
    assert resumed.weighting.lambdas == ref_lambdas
    assert _rows(tiny_cfg.out_dir / "metrics.csv") == ref_rows


def test_resume_drops_rows_logged_after_the_checkpoint(tiny_cfg):
    t1 = Trainer(tiny_cfg)
    t1.train(resume=False, max_steps=5)          # checkpoint at step 5
    for _ in range(3):                             # 3 more steps logged, then "crash"
        t1.train_step()
    t2 = Trainer(tiny_cfg)
    t2.train(resume=True)
    steps = [int(r["step"]) for r in _rows(tiny_cfg.out_dir / "metrics.csv")]
    assert steps == list(range(10)) + [10]


def test_finished_run_writes_outputs(tiny_cfg):
    summary = Trainer(tiny_cfg).train(resume=False)
    out = tiny_cfg.out_dir
    assert summary["finished"] and summary["step"] == 10
    assert (out / "prediction.npz").exists() and (out / "config.yaml").exists()
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["rel_l2"] == pytest.approx(summary["rel_l2"])
    rows = _rows(out / "metrics.csv")
    assert rows[0]["rel_l2"] and rows[-1]["step"] == "10" and rows[-1]["rel_l2"]
    assert rows[0]["grad_norm_ic"] and rows[3]["grad_norm_ic"] and not rows[1]["grad_norm_ic"]


def test_power_throttling_opt_out_applies_only_on_windows():
    assert disable_power_throttling() == (sys.platform == "win32")


def test_resume_rejects_a_different_config(tiny_cfg):
    Trainer(tiny_cfg).train(resume=False, max_steps=5)
    tiny_cfg.causal.tol = 2.0
    with pytest.raises(ValueError, match="causal"):
        Trainer(tiny_cfg).train(resume=True)


def test_resume_reads_a_checkpoint_without_scale_param_as_log(tiny_cfg):
    # Checkpoints written before rwf.scale_param existed trained s (D21).
    tiny_cfg.model.rwf.scale_param = "log"
    Trainer(tiny_cfg).train(resume=False, max_steps=5)
    path = tiny_cfg.checkpoint_dir / "last.pt"
    state = torch.load(path, weights_only=False)
    del state["config"]["model"]["rwf"]["scale_param"]  # as saved before the key was added
    torch.save(state, path)
    tiny_cfg.model.rwf.scale_param = "direct"
    with pytest.raises(ValueError, match="model"):
        Trainer(tiny_cfg).train(resume=True)
    tiny_cfg.model.rwf.scale_param = "log"
    resumed = Trainer(tiny_cfg)
    resumed.train(resume=True)
    assert resumed.step == tiny_cfg.training.steps


def test_config_mismatch_ignores_rwf_settings_when_rwf_is_off(tiny_cfg):
    saved = config_to_dict(tiny_cfg)
    tiny_cfg.model.rwf.scale_param = "log"
    assert config_mismatch(saved, tiny_cfg) == ["model"]
    saved["model"]["rwf"]["enabled"] = tiny_cfg.model.rwf.enabled = False
    assert config_mismatch(saved, tiny_cfg) == []
    tiny_cfg.training.steps += 10  # extending a run is allowed
    assert config_mismatch(saved, tiny_cfg) == []
