"""
Compatibility facade re-exporting dataset helpers from the data subpackage.
"""

from __future__ import annotations

from .data.cifar100 import SplitCIFAR100, SplitTask, get_cifar100_dataloaders

__all__ = ["SplitCIFAR100", "SplitTask", "get_cifar100_dataloaders"]
