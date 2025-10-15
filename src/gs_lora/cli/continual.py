"""
CLI entry-point for running the continual learning experiment.
"""

from __future__ import annotations

from ..training.continual import ContinualLearningConfig, run_experiment, main as continual_main


def main() -> None:
    continual_main()


__all__ = ["ContinualLearningConfig", "run_experiment", "main"]


if __name__ == "__main__":
    main()
