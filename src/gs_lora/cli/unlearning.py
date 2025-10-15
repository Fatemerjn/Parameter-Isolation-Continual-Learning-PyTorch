"""
CLI entry-point for running the unlearning demonstration.
"""

from __future__ import annotations

from ..training.unlearning import main as unlearning_main


def main() -> None:
    unlearning_main()


__all__ = ["main"]


if __name__ == "__main__":
    main()
