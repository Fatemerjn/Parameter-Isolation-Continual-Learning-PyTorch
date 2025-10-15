"""
Dataset utilities for continual learning experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


@dataclass(frozen=True)
class SplitTask:
    """Metadata describing a single continual learning task partition."""

    task_id: int
    class_indices: Sequence[int]


class SplitCIFAR100(Dataset):
    """
    Thin wrapper that relabels a CIFAR-100 dataset for a given task split.

    Each sample returns (image, remapped_label, task_id).
    """

    def __init__(
        self,
        cifar100_dataset: Dataset,
        task: SplitTask,
    ) -> None:
        self.cifar100_dataset = cifar100_dataset
        self.task = task
        self.class_map = {old_idx: new_idx for new_idx, old_idx in enumerate(task.class_indices)}
        self.indices: List[int] = [
            idx
            for idx, (_, label) in enumerate(self.cifar100_dataset)
            if label in self.class_map
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        original_idx = self.indices[idx]
        image, label = self.cifar100_dataset[original_idx]
        new_label = self.class_map[label]
        return image, new_label, self.task.task_id


def _build_tasks(num_tasks: int, num_classes: int = 100) -> Iterable[SplitTask]:
    classes_per_task = num_classes // num_tasks
    all_classes = list(range(num_classes))

    for task_id in range(num_tasks):
        start = task_id * classes_per_task
        end = (task_id + 1) * classes_per_task
        yield SplitTask(task_id=task_id, class_indices=all_classes[start:end])


def get_cifar100_dataloaders(
    *,
    num_tasks: int = 10,
    batch_size: int = 64,
    root: str | Path = "./data",
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Build train/test dataloaders for a Split-CIFAR100 continual learning setup.
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    root = Path(root)
    train_dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)

    task_dataloaders: List[Tuple[DataLoader, DataLoader]] = []
    for task in _build_tasks(num_tasks):
        train_task_dataset = SplitCIFAR100(train_dataset, task)
        test_task_dataset = SplitCIFAR100(test_dataset, task)

        train_loader = DataLoader(train_task_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_task_dataset, batch_size=batch_size, shuffle=False)
        task_dataloaders.append((train_loader, test_loader))

    return task_dataloaders


__all__ = ["SplitCIFAR100", "SplitTask", "get_cifar100_dataloaders"]
