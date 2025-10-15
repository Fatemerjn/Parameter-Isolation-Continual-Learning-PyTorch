"""
Continual learning training loop for Gradient-Sparse LoRA experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import get_cifar100_dataloaders
from ..lora import apply_gradient_mask, get_gradient_sparse_mask
from ..models import MultiHeadLoRAModel


@dataclass
class ContinualLearningConfig:
    device: torch.device
    num_tasks: int = 5
    epochs_per_task: int = 5
    batch_size: int = 64
    learning_rate: float = 5e-3
    lora_rank: int = 4
    lora_alpha: int = 16
    sparsity: float = 0.8


def evaluate(model: MultiHeadLoRAModel, test_loaders: Sequence[DataLoader], device: torch.device) -> List[float]:
    model.eval()
    task_accuracies: List[float] = []
    for task_id, loader in enumerate(test_loaders):
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task_id=task_id)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        task_accuracies.append(100 * correct / total)
    return task_accuracies


def run_experiment(config: ContinualLearningConfig) -> Tuple[List[List[float]], float, float]:
    """
    Execute the full continual learning schedule and return per-task results.

    Returns
    -------
    accuracies_history:
        Accuracy after each task; index 0 corresponds to results after task 1, etc.
    final_average_accuracy:
        Mean accuracy over tasks at the end of training.
    average_forgetting:
        Empirical forgetting metric across tasks.
    """

    print(f"Using device: {config.device}")

    task_dataloaders = get_cifar100_dataloaders(num_tasks=config.num_tasks, batch_size=config.batch_size)
    model = MultiHeadLoRAModel(num_tasks=config.num_tasks, rank=config.lora_rank, alpha=config.lora_alpha).to(config.device)
    criterion = nn.CrossEntropyLoss()

    test_loaders = [loader for _, loader in task_dataloaders]
    results: List[List[float]] = []

    for task_id in range(config.num_tasks):
        print(f"\n--- Training on Task {task_id + 1}/{config.num_tasks} ---")
        train_loader, _ = task_dataloaders[task_id]

        task_model = model.tasks_models[task_id]
        masks = get_gradient_sparse_mask(task_model, train_loader, config.device, sparsity=config.sparsity)

        optimizer = optim.Adam([p for p in task_model.parameters() if p.requires_grad], lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(config.epochs_per_task):
            task_model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs_per_task}")
            for images, labels, _ in pbar:
                images, labels = images.to(config.device), labels.to(config.device)

                optimizer.zero_grad()
                outputs = task_model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                apply_gradient_mask(task_model, masks)
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
            scheduler.step()

        accuracies = evaluate(model, test_loaders[: task_id + 1], config.device)
        results.append(accuracies)
        print(f"Accuracies after Task {task_id + 1}: {['{:.2f}'.format(acc) for acc in accuracies]}")

    final_accuracies = results[-1]
    avg_acc = float(np.mean(final_accuracies))
    forgetting = 0.0
    for i in range(len(final_accuracies) - 1):
        forgetting += max(res[i] for res in results if len(res) > i) - final_accuracies[i]
    avg_forgetting = forgetting / (len(final_accuracies) - 1) if len(final_accuracies) > 1 else 0.0

    print("\n--- Final Report ---")
    print(f"Final Average Accuracy: {avg_acc:.2f}%")
    print(f"Average Forgetting: {avg_forgetting:.2f}%")

    return results, avg_acc, avg_forgetting


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = ContinualLearningConfig(device=device)
    run_experiment(config)


__all__ = ["ContinualLearningConfig", "evaluate", "run_experiment", "main"]
