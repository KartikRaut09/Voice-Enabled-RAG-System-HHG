"""
Shared utility functions for the Colab research pipeline.
Handles path resolution, configuration loading, JSON/YAML I/O,
reproducibility seeding, and environment metadata logging.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ── Repository Root Discovery ──


def find_repo_root(marker: str = "colab") -> Path:
    """Walk up from CWD to find the repository root.

    Looks for a directory containing the ``marker`` folder.
    Works after ``git clone`` into Colab, local dev, or CI.

    Args:
        marker: Name of a directory that must exist at the repo root.

    Returns:
        Absolute path to the repository root.

    Raises:
        FileNotFoundError: If the marker directory is not found.
    """
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(
        f"Cannot find repository root (looked for '{marker}/' directory "
        f"starting from {Path.cwd()}). Make sure you cloned the repo and "
        f"are running from within it."
    )


def get_colab_root() -> Path:
    """Return the ``colab/`` directory path."""
    return find_repo_root("colab") / "colab"


def get_reports_dir() -> Path:
    """Return the ``colab/reports/`` directory, creating it if needed."""
    d = get_colab_root() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_artifacts_dir() -> Path:
    """Return the ``colab/artifacts/`` directory, creating it if needed."""
    d = get_colab_root() / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_data_dir() -> Path:
    """Return the ``colab/data/`` directory, creating it if needed."""
    d = get_colab_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Configuration ──


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the experiment configuration YAML file.

    Args:
        path: Explicit path to the config file.  If ``None``, uses the
              default ``colab/configs/experiment_config.yaml``.

    Returns:
        Parsed configuration dictionary.
    """
    if path is None:
        path = get_colab_root() / "configs" / "experiment_config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Reproducibility ──


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ── Environment Metadata ──


def get_environment_info() -> dict[str, Any]:
    """Collect environment metadata for experiment logging."""
    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # GPU info
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 2
            )
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False

    # Key package versions
    for pkg_name in [
        "sentence_transformers",
        "transformers",
        "datasets",
        "faiss",
        "numpy",
        "scipy",
    ]:
        try:
            mod = __import__(pkg_name)
            info[f"{pkg_name}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg_name}_version"] = "not installed"

    return info


# ── JSON / YAML I/O ──


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Save data as a JSON file with proper encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    return path


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data: Any, path: str | Path) -> Path:
    """Save data as a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return path


# ── Logging Helpers ──


def print_header(title: str, char: str = "═", width: int = 60) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_table(headers: list[str], rows: list[list[Any]], col_width: int = 15) -> None:
    """Print a simple aligned text table."""
    fmt = " | ".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*[str(h) for h in headers]))
    print("-" * (col_width * len(headers) + 3 * (len(headers) - 1)))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


# ── Secret Access ──


def get_secret(name: str, required: bool = False) -> str | None:
    """Retrieve a secret from Colab Secrets or environment variables.

    Args:
        name: Secret name (e.g. ``SARVAM_API_KEY``).
        required: If True, raise an error when the secret is missing.

    Returns:
        The secret value, or None if not found and not required.
    """
    # Try Colab secrets first
    try:
        from google.colab import userdata

        val = userdata.get(name)
        if val:
            return val
    except (ImportError, Exception):
        pass

    # Fall back to environment variable
    val = os.environ.get(name)
    if val:
        return val

    if required:
        raise EnvironmentError(
            f"Required secret '{name}' not found.\n"
            f"In Google Colab: Add it via 🔑 Secrets in the left sidebar.\n"
            f"Locally: Set the environment variable: export {name}=<your-key>"
        )
    return None
