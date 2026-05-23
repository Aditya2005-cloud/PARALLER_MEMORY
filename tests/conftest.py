import pytest
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Use a writable tmp dir for torch/HF caches (cross-platform)
_tmp_cache = Path(tempfile.gettempdir())
os.environ.setdefault("TORCH_HOME", str(_tmp_cache / "torch_home"))
os.environ.setdefault("HF_HOME", str(_tmp_cache / "hf_home"))


@pytest.fixture(scope="session")
def test_data_dir():
    return project_root / "tests" / "data"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


@pytest.fixture(autouse=True)
def isolated_db_dirs(tmp_path, monkeypatch):
    """Redirect all DB paths to a per-test temp directory so tests never share state."""
    import parallel_memory.config as cfg
    import parallel_memory.database as db_mod

    users_dir = tmp_path / "users"
    global_dir = tmp_path / "global"
    users_dir.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "USERS_DIR", users_dir)
    monkeypatch.setattr(cfg, "GLOBAL_DIR", global_dir)
    monkeypatch.setattr(db_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db_mod, "USERS_DIR", users_dir)


@pytest.fixture
def capture_logs(caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog
