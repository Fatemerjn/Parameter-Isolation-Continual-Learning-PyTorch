"""
Compatibility wrapper for the continual learning CLI entry-point.
"""

from __future__ import annotations

from .training.continual import ContinualLearningConfig, run_experiment
from .cli.continual import main

__all__ = ["ContinualLearningConfig", "run_experiment", "main"]


if __name__ == "__main__":
    main()
