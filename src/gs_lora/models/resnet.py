"""
Model definitions for the Gradient-Sparse LoRA experiments.
"""

from __future__ import annotations

from typing import List

import torch.nn as nn
import torchvision.models as models

from ..lora import setup_lora_training


class MultiHeadLoRAModel(nn.Module):
    """
    ResNet-18 backbone with per-task LoRA adapters and classification heads.
    """

    def __init__(self, num_tasks: int, num_classes_per_task: int = 10, rank: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.tasks_models = nn.ModuleList()

        for _ in range(num_tasks):
            task_model = models.resnet18(weights="IMAGENET1K_V1")
            task_model.fc = nn.Linear(task_model.fc.in_features, num_classes_per_task)
            task_model = setup_lora_training(task_model, rank=rank, alpha=alpha)
            self.tasks_models.append(task_model)

    def forward(self, x, task_id: int):
        return self.tasks_models[task_id](x)


class MultiHeadResNet(nn.Module):
    """
    Standard multi-head ResNet-18 used for unlearning experiments.
    """

    def __init__(self, num_tasks: int, num_classes_per_task: int) -> None:
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        in_features = backbone.fc.in_features

        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.heads: List[nn.Module] = nn.ModuleList(
            [nn.Linear(in_features, num_classes_per_task) for _ in range(num_tasks)]
        )

    def forward(self, x, task_id: int):
        features = self.backbone(x)
        return self.heads[task_id](features)


__all__ = ["MultiHeadLoRAModel", "MultiHeadResNet"]
