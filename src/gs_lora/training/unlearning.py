"""
Unlearning demonstration using an experience replay buffer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets import get_cifar100_dataloaders
from ..models import MultiHeadResNet


class ExperienceReplay:
    """Simple fixed-size buffer storing balanced samples per task."""

    def __init__(self, buffer_size_per_task: int = 200) -> None:
        self.buffer_size_per_task = buffer_size_per_task
        self.buffer: List[Tuple[torch.Tensor, int, int]] = []

    def get_full_buffer(self) -> List[Tuple[torch.Tensor, int, int]]:
        return self.buffer

    def on_task_end(self, task_id: int, train_loader: DataLoader) -> None:
        num_samples_to_add = self.buffer_size_per_task
        samples_added = 0

        for images, labels, _ in train_loader:
            for image, label in zip(images, labels):
                if samples_added < num_samples_to_add:
                    self.buffer.append((image, int(label.item()), task_id))
                    samples_added += 1
                else:
                    break
            if samples_added >= num_samples_to_add:
                break


def evaluate(model: MultiHeadResNet, test_loaders: Sequence[DataLoader], device: torch.device) -> List[float]:
    model.eval()
    accuracies: List[float] = []
    for task_id, loader in enumerate(test_loaders):
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task_id=task_id)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracies.append(100 * correct / total)
    return accuracies


@dataclass
class UnlearningConfig:
    device: torch.device
    num_tasks: int = 5
    epochs_per_task: int = 5
    training_lr: float = 1e-3
    unlearning_lr: float = 1e-4
    unlearning_epochs: int = 5
    buffer_size_per_task: int = 200
    task_to_forget: int = 2


def perform_unlearning(model: MultiHeadResNet, buffer: ExperienceReplay, config: UnlearningConfig) -> None:
    print(f"\n--- Starting Unlearning Process for Task {config.task_to_forget + 1} ---")

    model.eval()
    full_buffer = buffer.get_full_buffer()
    forget_data = [sample for sample in full_buffer if sample[2] == config.task_to_forget]
    retain_data = [sample for sample in full_buffer if sample[2] != config.task_to_forget]

    if not forget_data or not retain_data:
        print("Not enough data in buffer to perform unlearning.")
        return

    optimizer = optim.Adam(model.parameters(), lr=config.unlearning_lr)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    num_classes_per_task = model.heads[0].out_features

    for epoch in range(config.unlearning_epochs):
        random.shuffle(retain_data)
        random.shuffle(forget_data)

        num_batches = min(len(retain_data), len(forget_data))

        for batch_idx in range(num_batches):
            optimizer.zero_grad()

            retain_image, retain_label, retain_task_id = retain_data[batch_idx]
            retain_image = retain_image.unsqueeze(0).to(config.device)
            retain_label_tensor = torch.tensor([retain_label], device=config.device)

            retain_outputs = model(retain_image, task_id=retain_task_id)
            loss_retain = criterion_ce(retain_outputs, retain_label_tensor)

            forget_image, _, forget_task_id = forget_data[batch_idx]
            forget_image = forget_image.unsqueeze(0).to(config.device)
            forget_outputs = model(forget_image, task_id=forget_task_id)

            uniform_dist = torch.full_like(forget_outputs, 1.0 / num_classes_per_task)
            loss_forget = criterion_kl(torch.log_softmax(forget_outputs, dim=1), uniform_dist)

            total_loss = loss_retain + loss_forget
            total_loss.backward()
            optimizer.step()

        print(f"Unlearning Epoch {epoch + 1}/{config.unlearning_epochs}, Last Total Loss: {total_loss.item():.4f}")


def run_demo(config: UnlearningConfig) -> Tuple[List[float], List[float]]:
    """Train tasks sequentially, then run the unlearning procedure."""

    task_dataloaders = get_cifar100_dataloaders(num_tasks=config.num_tasks)
    model = MultiHeadResNet(num_tasks=config.num_tasks, num_classes_per_task=(100 // config.num_tasks)).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.training_lr)
    criterion = nn.CrossEntropyLoss()
    replay_buffer = ExperienceReplay(buffer_size_per_task=config.buffer_size_per_task)

    for task_id in range(config.num_tasks):
        print(f"\n--- Training on Task {task_id + 1}/{config.num_tasks} ---")
        train_loader, _ = task_dataloaders[task_id]
        for epoch in range(config.epochs_per_task):
            model.train()
            for images, labels, _ in tqdm(train_loader):
                images, labels = images.to(config.device), labels.to(config.device)
                optimizer.zero_grad()
                outputs = model(images, task_id=task_id)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        replay_buffer.on_task_end(task_id, train_loader)

    test_loaders = [loader for _, loader in task_dataloaders]
    print("\n--- Accuracies BEFORE Unlearning ---")
    before_accuracies = evaluate(model, test_loaders, config.device)
    for i, acc in enumerate(before_accuracies):
        print(f"Task {i + 1} Accuracy: {acc:.2f}%")
    print(f"Average Accuracy: {np.mean(before_accuracies):.2f}%")

    perform_unlearning(model, replay_buffer, config)

    print(f"\n--- Accuracies AFTER Unlearning Task {config.task_to_forget + 1} ---")
    after_accuracies = evaluate(model, test_loaders, config.device)
    for i, acc in enumerate(after_accuracies):
        status = " (FORGOTTEN)" if i == config.task_to_forget else " (RETAINED)"
        change = after_accuracies[i] - before_accuracies[i]
        print(f"Task {i + 1} Accuracy: {acc:.2f}%{status} | Change: {change:+.2f}%")
    print(f"Average Accuracy: {np.mean(after_accuracies):.2f}%")

    return before_accuracies, after_accuracies


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = UnlearningConfig(device=device)
    run_demo(config)


__all__ = ["ExperienceReplay", "UnlearningConfig", "evaluate", "perform_unlearning", "run_demo", "main"]
