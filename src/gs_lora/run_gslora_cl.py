"""
Compatibility wrapper for the legacy continual learning entry-point.
"""

from __future__ import annotations

from .training.continual import main as continual_main
from .training.continual import run_experiment, ContinualLearningConfig


def main() -> None:
    continual_main()


__all__ = ["ContinualLearningConfig", "run_experiment", "main"]


if __name__ == "__main__":
    main()
