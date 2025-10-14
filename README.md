# Gradient-Sparse LoRA

## Overview
- Adapts Residual Networks with LoRA adapters for continual learning on Split CIFAR-100.
- Introduces gradient sparsification masks to keep LoRA updates compact.
- Provides ready-to-run executables for continual-training and unlearning workflows.

## Layout
- `src/gs_lora/` – installable package with datasets, models, LoRA utilities, and training loops.
  - `training/continual.py` – Gradient-sparse continual learning routine.
  - `training/unlearning.py` – Experience-replay based unlearning demo.
- `notebooks/` – exploratory Jupyter notebooks (`continual_learning_sanity_check.ipynb`).
- `data/` – optional local copy of CIFAR-100 (downloaded automatically if absent).
- `results/` – saved accuracy traces from previous experiments.
- `tests/` – smoke tests that ensure the package stays importable.
- `requirements.txt` / `pyproject.toml` – dependency declarations for both pip and Poetry.

## Quickstart
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `export PYTHONPATH=src`
4. Continual run: `python -m gs_lora.run_gslora_cl`

### Unlearning demo
```
export PYTHONPATH=src
python -m gs_lora.unlearning_demo
```

## Data
- Scripts default to `./data/cifar-100-python`; keep existing assets here or update the `root` argument in the loaders.
- To conserve space you can point multiple repositories to a shared dataset by symlinking `data/`.

## Next Steps
- Tweak LoRA rank, sparsity, and scheduling via `gs_lora.training.continual`.
- Extend LoRA adapters or masking strategies inside `gs_lora.lora`.
