import os
import sys
import logging
import subprocess
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger("parallel_memory.colab_utils")


class ColabEnvironment:
    def __init__(self):
        self.in_colab = self._check_colab()
        self.gpu_available = self._check_gpu()
        self.repo_root = Path.cwd()

    def _check_colab(self) -> bool:
        try:
            import google.colab
            return True
        except ImportError:
            return False

    def _check_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def setup_colab(self, github_token: Optional[str] = None):
        if not self.in_colab:
            logger.warning("Not running in Colab")
            return

        logger.info("Setting up Colab environment")

        self._install_dependencies()
        self._setup_gpu()
        self._clone_repo_if_needed()
        if github_token:
            self._setup_github_auth(github_token)

    def _install_dependencies(self):
        logger.info("Installing dependencies...")
        req_file = self.repo_root / "requirements-colab.txt"
        try:
            if req_file.exists():
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
                    check=True,
                    capture_output=True
                )
                logger.info(f"Installed dependencies from {req_file}")
            else:
                # Minimal fallback if file is missing.
                packages = ["fastapi", "uvicorn", "pydantic", "numpy", "scipy", "scikit-learn"]
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", *packages],
                    check=True,
                    capture_output=True
                )
                logger.info("Installed fallback minimal dependencies")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed dependency installation: {e}")

    def _setup_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA version: {torch.version.cuda}")
                device_props = torch.cuda.get_device_properties(0)
                logger.info(f"GPU memory: {device_props.total_memory / 1e9:.2f} GB")
            else:
                logger.warning("No GPU available, using CPU (will be slow)")
        except Exception as e:
            logger.error(f"GPU setup error: {e}")

    def _clone_repo_if_needed(self):
        git_dir = Path(".git")
        if not git_dir.exists():
            logger.info("Cloning repository...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/Aditya2005-cloud/PARALLER_MEMORY.git", "."],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone repo: {e}")

    def _setup_github_auth(self, github_token: str):
        logger.info("Setting up GitHub authentication...")
        os.environ["GITHUB_TOKEN"] = github_token
        subprocess.run(
            ["git", "config", "--global", "user.email", "colab@parallel-memory.local"],
            check=False
        )
        subprocess.run(
            ["git", "config", "--global", "user.name", "Colab Trainer"],
            check=False
        )


class GitHubSync:
    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()
        self.token = os.getenv("GITHUB_TOKEN")

    def commit_and_push(self, files: list[str], message: str, branch: str = "main") -> bool:
        try:
            os.chdir(self.repo_path)

            subprocess.run(["git", "add"] + files, check=True, capture_output=True)

            subprocess.run(
                ["git", "commit", "-m", message],
                check=False,
                capture_output=True
            )

            subprocess.run(
                ["git", "push", "-u", "origin", branch],
                check=False,
                capture_output=True
            )

            logger.info(f"Pushed {len(files)} files to {branch}: {message}")
            return True

        except Exception as e:
            logger.error(f"Git push failed: {e}")
            return False

    def pull_latest(self, branch: str = "main") -> bool:
        try:
            os.chdir(self.repo_path)
            subprocess.run(
                ["git", "pull", "origin", branch],
                check=True,
                capture_output=True
            )
            logger.info(f"Pulled latest from {branch}")
            return True
        except Exception as e:
            logger.error(f"Git pull failed: {e}")
            return False

    def create_branch(self, branch_name: str) -> bool:
        try:
            os.chdir(self.repo_path)
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                check=True,
                capture_output=True
            )
            logger.info(f"Created and checked out branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e}")
            return False


class ModelCheckpointer:
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model_dict: dict, name: str, metadata: Optional[dict] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / f"{name}_{timestamp}.json"

        checkpoint = {
            "timestamp": timestamp,
            "name": name,
            "metadata": metadata or {},
            "models": {k: str(v) for k, v in model_dict.items()},
        }

        checkpoint_file.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        logger.info(f"Saved checkpoint: {checkpoint_file}")
        return checkpoint_file

    def list_checkpoints(self) -> list[dict]:
        checkpoints = []
        for cp_file in sorted(self.checkpoint_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(cp_file.read_text(encoding="utf-8"))
                checkpoints.append(data)
            except Exception as e:
                logger.error(f"Failed to load checkpoint {cp_file}: {e}")
        return checkpoints

    def load_latest_checkpoint(self) -> Optional[dict]:
        checkpoints = self.list_checkpoints()
        if checkpoints:
            logger.info(f"Loading latest checkpoint: {checkpoints[0]['timestamp']}")
            return checkpoints[0]
        return None


class MemoryOptimizer:
    @staticmethod
    def enable_reduced_precision():
        try:
            import torch
            torch.set_float32_matmul_precision("medium")
            logger.info("Enabled reduced precision (bfloat16)")
        except Exception as e:
            logger.error(f"Failed to enable reduced precision: {e}")

    @staticmethod
    def clear_gpu_cache():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared GPU cache")
        except Exception as e:
            logger.error(f"Failed to clear GPU cache: {e}")

    @staticmethod
    def get_memory_stats() -> dict:
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
            return {"status": "GPU not available"}
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}


class SessionManager:
    def __init__(self, session_dir: Optional[Path] = None):
        self.session_dir = session_dir or Path("/tmp/parallel_memory_session")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.session_dir / "session_state.json"

    def save_session(self, data: dict) -> Path:
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        self.session_file.write_text(json.dumps(session_data, indent=2), encoding="utf-8")
        logger.info(f"Saved session to {self.session_file}")
        return self.session_file

    def load_session(self) -> Optional[dict]:
        if self.session_file.exists():
            try:
                data = json.loads(self.session_file.read_text(encoding="utf-8"))
                logger.info(f"Loaded session from {self.session_file}")
                return data.get("data")
            except Exception as e:
                logger.error(f"Failed to load session: {e}")
        return None

    def backup_to_github(self, git_sync: GitHubSync, message: str) -> bool:
        try:
            files = [str(f.relative_to(git_sync.repo_path)) for f in self.session_dir.rglob("*") if f.is_file()]
            return git_sync.commit_and_push(files, message)
        except Exception as e:
            logger.error(f"Failed to backup to GitHub: {e}")
            return False


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"parallel_memory_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger("parallel_memory")
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger.info(f"Logging initialized to {log_file}")
    return root_logger
