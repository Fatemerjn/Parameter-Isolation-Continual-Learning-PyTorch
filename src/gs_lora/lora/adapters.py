"""
LoRA adapter utilities.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer augmented with low-rank adapters."""

    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha

        # The original, frozen linear layer
        self.linear = linear_layer
        
        # Create the low-rank trainable matrices
        self.lora_A = nn.Parameter(torch.randn(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        
        # Scaling factor
        self.scaling = self.alpha / self.rank

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def forward(self, x):
        # Original path (frozen)
        frozen_output = self.linear(x)
        
        if self.rank > 0:
            lora_output = (x @ self.lora_A.T) @ self.lora_B.T
            return frozen_output + lora_output * self.scaling
            
        return frozen_output


def inject_lora_to_model(model: nn.Module, rank: int = 8, alpha: int = 16) -> None:
    """Recursively replace linear layers with LoRA-augmented ones."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, rank, alpha))
        elif len(list(module.children())) > 0:
            inject_lora_to_model(module, rank, alpha)


def setup_lora_training(model: nn.Module, rank: int = 8, alpha: int = 16) -> nn.Module:
    """Freeze backbone weights and inject trainable LoRA adapters."""
    for param in model.parameters():
        param.requires_grad = False
        
    if rank > 0:
        inject_lora_to_model(model, rank, alpha)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    return model


def get_gradient_sparse_mask(
    model: nn.Module,
    dataloader,
    device: torch.device,
    sparsity: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Probes the model to create a sparsity mask for LoRA weights."""
    print("Probing model for Gradient-Sparsity...")
    model.train()
    abs_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters() if "lora_" in name}

    criterion = nn.CrossEntropyLoss()
    for i, (inputs, targets, _) in enumerate(dataloader):
        if i >= 3:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if "lora_" in name and param.grad is not None:
                abs_grads[name] += param.grad.abs()
    
    all_grads = torch.cat([grad.view(-1) for grad in abs_grads.values()])
    threshold = torch.quantile(all_grads, sparsity)
    
    masks = {}
    for name, grad in abs_grads.items():
        masks[name] = (grad >= threshold).float()

    kept_params = int(sum(m.sum() for m in masks.values()))
    total_params = sum(m.numel() for m in masks.values())
    print(f"Created mask with {sparsity*100}% sparsity. Training {kept_params}/{total_params} LoRA params.")
    return masks


def apply_gradient_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Applies the sparsity masks to the model's LoRA parameter gradients."""
    for name, param in model.named_parameters():
        if name in masks and param.grad is not None:
            param.grad.mul_(masks[name])


__all__ = [
    "LoRALinear",
    "inject_lora_to_model",
    "setup_lora_training",
    "get_gradient_sparse_mask",
    "apply_gradient_mask",
]
