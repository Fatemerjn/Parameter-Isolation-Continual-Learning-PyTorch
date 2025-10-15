"""
Compatibility wrapper for the legacy unlearning demonstration entry-point.
"""

from __future__ import annotations

from .training.unlearning import (
    ExperienceReplay,
    UnlearningConfig,
    evaluate,
    perform_unlearning,
    run_demo,
)
from .cli.unlearning import main

__all__ = [
    "ExperienceReplay",
    "UnlearningConfig",
    "evaluate",
    "perform_unlearning",
    "run_demo",
    "main",
]


if __name__ == "__main__":
    main()
