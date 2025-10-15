"""
Gradient-Sparse LoRA toolkit.
"""

from .datasets import SplitCIFAR100, SplitTask, get_cifar100_dataloaders
from .lora import (
    LoRALinear,
    apply_gradient_mask,
    get_gradient_sparse_mask,
    inject_lora_to_model,
    setup_lora_training,
)
from .models import MultiHeadLoRAModel, MultiHeadResNet
from .training.continual import ContinualLearningConfig, evaluate as evaluate_continual, run_experiment
from .training.unlearning import (
    ExperienceReplay,
    UnlearningConfig,
    evaluate as evaluate_unlearning,
    perform_unlearning,
    run_demo,
)

__all__ = [
    "SplitCIFAR100",
    "SplitTask",
    "get_cifar100_dataloaders",
    "LoRALinear",
    "apply_gradient_mask",
    "get_gradient_sparse_mask",
    "inject_lora_to_model",
    "setup_lora_training",
    "MultiHeadLoRAModel",
    "MultiHeadResNet",
    "ContinualLearningConfig",
    "evaluate_continual",
    "run_experiment",
    "ExperienceReplay",
    "UnlearningConfig",
    "evaluate_unlearning",
    "perform_unlearning",
    "run_demo",
]
