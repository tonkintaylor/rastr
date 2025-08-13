from pathlib import Path

import pytest

collect_ignore_glob = ["assets/**"]
pytest_plugins = []


@pytest.fixture(scope="session")
def assets_dir() -> Path:
    """Return a path to the test assets directory."""
    return Path(__file__).parent / "assets"
