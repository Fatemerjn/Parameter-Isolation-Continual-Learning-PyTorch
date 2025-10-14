import sys
from pathlib import Path


def pytest_configure(config):
    # Ensure the src/ (preferred) or top-level package is importable during tests
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    elif str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
