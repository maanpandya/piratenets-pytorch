"""Experiment configuration: nested dataclasses loaded from YAML.

Every pipeline component is a switch here, so each row of an ablation table is
one YAML file. Unknown keys are an error (a typo must not silently fall back to
a default). Values can be overridden from the command line with dotted keys,
e.g. ``--set training.steps=500 causal.enabled=false``.
"""
import copy
import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FourierConfig:
    enabled: bool = True
    scale: float = 1.0       # std of the Gaussian frequency matrix B
    embed_dim: int = 256     # output size: embed_dim/2 cosines + embed_dim/2 sines
    trainable: bool = True   # train B along with the network (False: fixed buffer)


@dataclass
class RWFConfig:
    enabled: bool = True
    mu: float = 0.5          # scale factors s ~ N(mu, sigma^2), W = diag(exp(s)) V
    sigma: float = 0.1
    scale_param: str = "direct"  # "direct": train exp(s) itself (jaxpi, D21) | "log": train s (EG Algorithm 2)


@dataclass
class ModelConfig:
    arch: str = "mlp"        # "mlp"
    hidden_dim: int = 256
    num_layers: int = 4      # number of hidden layers
    activation: str = "tanh"
    periodic: bool = True    # exact periodic embedding x -> (cos(pi x), sin(pi x))
    fourier: FourierConfig = field(default_factory=FourierConfig)
    rwf: RWFConfig = field(default_factory=RWFConfig)


@dataclass
class OptimConfig:
    lr: float = 1e-3
    decay_rate: float = 0.9
    decay_steps: int = 2000
    staircase: bool = False  # False: lr * rate**(step/decay_steps); True: floor the exponent
    warmup_steps: int = 0    # linear warmup from 0 (PirateNets uses 5000; the Expert's Guide ablation does not)


@dataclass
class TrainingConfig:
    steps: int = 200_000
    batch_size: int = 4096   # residual collocation points per step
    ic_sampling: str = "grid"  # "grid": the reference x-grid (fixed) | "random": n_ic fresh points per step
    n_ic: int = 4096         # initial-condition points per step when ic_sampling = "random"
    seed: int = 0
    derivatives: str = "func"  # "func" (torch.func vmap/grad/jvp) | "autograd"


@dataclass
class WeightingConfig:
    scheme: str = "grad_norm"  # "none" | "grad_norm"
    update_every: int = 1000   # f in Algorithm 1
    momentum: float = 0.9      # alpha in eq. (2.15)
    grad_norm_reduction: str = "mean"  # "mean" (jaxpi) | "sum" (eqs. 2.12-2.14 as printed)


@dataclass
class CausalConfig:
    enabled: bool = True
    tol: float = 1.0         # epsilon in eq. (2.11)
    num_chunks: int = 32     # M; also used for stratified time sampling when disabled


@dataclass
class ProblemConfig:
    name: str = "allen_cahn"
    reference: str = "data/allen_cahn_ref.npz"


@dataclass
class LoggingConfig:
    out_dir: str = "runs/{name}"       # CSV log, resolved config, final prediction
    checkpoint_dir: str = ""           # empty: <out_dir>/checkpoints
    log_every: int = 100
    eval_every: int = 1000
    checkpoint_every: int = 5000


@dataclass
class Config:
    name: str = "default"
    device: str = "auto"     # "auto" | "cpu" | "cuda" | "cuda:0" ...
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def out_dir(self) -> Path:
        return resolve_path(self.logging.out_dir.format(name=self.name))

    @property
    def checkpoint_dir(self) -> Path:
        if self.logging.checkpoint_dir:
            return resolve_path(self.logging.checkpoint_dir.format(name=self.name))
        return self.out_dir / "checkpoints"


def resolve_path(p: str) -> Path:
    """Relative paths are taken relative to the repository root, not the CWD."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _coerce(value, tp, key):
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"{key}: expected bool, got {value!r}")
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError(f"{key}: expected int, got {value!r}")
        if isinstance(value, float) and not value.is_integer():
            raise TypeError(f"{key}: expected int, got {value!r}")
        return int(float(value)) if isinstance(value, str) else int(value)
    if tp is float:
        # YAML 1.1 reads "1e-3" as a string; accept it.
        if isinstance(value, bool):
            raise TypeError(f"{key}: expected float, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise TypeError(f"{key}: expected str, got {value!r}")
        return value
    raise TypeError(f"{key}: unsupported field type {tp}")


def _from_dict(cls, data: dict, prefix: str = ""):
    if not isinstance(data, dict):
        raise TypeError(f"{prefix or 'config'}: expected a mapping, got {data!r}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown config key(s) under '{prefix or '<root>'}': {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        tp, key = hints[name], f"{prefix}{name}"
        if dataclasses.is_dataclass(tp):
            kwargs[name] = _from_dict(tp, value, key + ".")
        else:
            kwargs[name] = _coerce(value, tp, key)
    return cls(**kwargs)


def config_from_dict(data: dict) -> Config:
    return _from_dict(Config, data)


# Keys added after runs had been saved, each with the value that reproduces what those
# runs did. A saved config that lacks one is read with this value, not the current default.
_LEGACY_VALUES = {("model", "rwf", "scale_param"): "log"}


def saved_config_from_dict(data: dict) -> Config:
    """Read the config of a saved run (checkpoint or config.yaml), possibly an older one."""
    data = copy.deepcopy(data)
    for keys, value in _LEGACY_VALUES.items():
        node = data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node.setdefault(keys[-1], value)
    return config_from_dict(data)


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def apply_overrides(data: dict, overrides) -> dict:
    """Apply ``a.b.c=value`` strings to a nested dict (values parsed as YAML)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def load_config(path, overrides=None) -> Config:
    with open(resolve_path(str(path)), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return config_from_dict(apply_overrides(data, overrides))


def save_config(cfg: Config, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)
