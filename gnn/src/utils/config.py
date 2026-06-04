from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Repository root (gnnvstree/)."""
    return Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = repo_root()
    for key in ("data_dir", "artifacts_dir", "results_dir", "sqlite_path"):
        if key in cfg:
            p = Path(cfg[key])
            if not p.is_absolute():
                cfg[key] = str(root / p)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("data_dir", "artifacts_dir", "results_dir"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)
