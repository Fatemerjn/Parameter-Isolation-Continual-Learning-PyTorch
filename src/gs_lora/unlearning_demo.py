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
    main as unlearning_main,
)


def main() -> None:
    unlearning_main()


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
