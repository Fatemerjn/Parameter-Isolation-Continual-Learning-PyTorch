# Gradient-Sparse LoRA

## Overview
- Adapts Residual Networks with LoRA adapters for continual learning on Split CIFAR-100.
- Introduces gradient sparsification masks to keep LoRA updates compact.
- Bundles continual training (`run_gslora_cl.py`) and unlearning-focused experimentation (`gs-lora-unlearning.py`).

## Layout
- `src/gs_lora/` – package with LoRA utilities, data helpers, and training scripts.
- `data/` – optional local copy of CIFAR-100 used by default when running scripts.
- `results/` – saved accuracy traces from previous experiments.
- `requirements.txt` – dependencies for LoRA experiments.

## Quickstart
1. `cd repositories/ParameterEfficient-Continual-Learning-PyTorch`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `export PYTHONPATH=src`
5. Continual run: `python -m gs_lora.run_gslora_cl`

### Unlearning demo
```
export PYTHONPATH=src
python -m gs_lora.gs-lora-unlearning
```

## Data
- Scripts default to `./data/cifar-100-python`; keep existing assets here or update the `root` argument in the loaders.
- To conserve space you can point multiple repositories to a shared dataset by symlinking `data/`.

## Next Steps
- Tweak LoRA rank, sparsity, and scheduling in `run_gslora_cl.py`.
- Extend `lora_utils.py` with advanced masking or quantisation strategies.
