# piratenets-pytorch

An independent PyTorch reimplementation of the methods in two papers:

- **PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks.**
  S. Wang, B. Li, Y. Chen, P. Perdikaris. *JMLR* 25 (2024). [arXiv:2402.00326](https://arxiv.org/abs/2402.00326)
- **An Expert's Guide to Training Physics-informed Neural Networks.**
  S. Wang, S. Sankaran, H. Wang, P. Perdikaris. 2023. [arXiv:2308.08468](https://arxiv.org/abs/2308.08468)

All credit for the methods belongs to the authors. This code is not affiliated
with them and contains no code from their JAX implementation
([jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi)). The method is
implemented from the papers. Where the papers leave a hyperparameter or design
detail open, jaxpi's example configs and code were consulted for that detail
only. Every such choice is recorded, with its source, in
[`docs/decisions.md`](docs/decisions.md).

It is a course project for CEE 292 at Stanford, and is a work in progress.

## Status

| Component | Status |
|---|---|
| Allen–Cahn problem, reference data, exact periodic embedding | done |
| Random Fourier features, random weight factorization | done |
| Grad-norm loss weighting, causal weighting | done |
| Expert's Guide Table 2 ablation configs and runner | done, results pending |
| Modified MLP, PirateNet + physics-informed init, NTK weighting | not started |

## Install

```bash
pip install -e ".[dev]"
pytest -q
```

The distribution and import name are both `piratenets`.

## Usage

```bash
# one configuration (resumes automatically from <out_dir>/checkpoints/last.pt)
python scripts/train.py --config configs/allen_cahn/ablation/default.yaml

# the whole ablation (Expert's Guide Table 2), skipping finished rows
python scripts/run_ablation.py --out-root runs/allen_cahn/ablation

# figures and the comparison table, from the CSV logs
python scripts/plot_ablation.py --out-root runs/allen_cahn/ablation
python scripts/plot_run.py runs/allen_cahn/ablation/ablation_default
```

Any config value can be overridden from the command line, for example
`--set training.steps=2000 causal.num_chunks=16`.

To run on a Colab GPU, with checkpoints and logs saved to Google Drive, use
[`notebooks/colab_ablation.ipynb`](notebooks/colab_ablation.ipynb)
([open in Colab](https://colab.research.google.com/github/maanpandya/piratenets-pytorch/blob/main/notebooks/colab_ablation.ipynb)).

## Layout

```
piratenets/            config, embeddings, layers (RWF), models, weighting, causal, trainer
piratenets/problems/   allen_cahn.py: PDE residual, sampling, reference data
configs/               one YAML per experiment (one per Table 2 row)
scripts/               train, run_ablation, benchmark_step, plot_*
data/                  reference solution and its generator
docs/                  spec.md, decisions.md
tests/                 pytest suite
```

## License

MIT, see [LICENSE](LICENSE). The papers themselves are not included in this repository.
