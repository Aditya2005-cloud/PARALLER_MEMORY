import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["TORCH_HOME"] = "/tmp/torch_home"
os.environ["HF_HOME"] = "/tmp/hf_home"


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


@pytest.fixture
def capture_logs(caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog
