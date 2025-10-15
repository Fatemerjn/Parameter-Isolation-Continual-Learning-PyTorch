"""
Command-line entry points for experiments and demos.
"""

from .continual import main as continual_main
from .unlearning import main as unlearning_main

__all__ = ["continual_main", "unlearning_main"]
