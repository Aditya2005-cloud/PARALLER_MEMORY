from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
USERS_DIR = DATA_DIR / "users"
GLOBAL_DIR = DATA_DIR / "global"


def ensure_base_dirs() -> None:
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)

