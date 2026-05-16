"""Small shared utilities."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def numeric_stats(values: Iterable[float]) -> dict[str, float | None]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None, "p25": None, "p75": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def rolling_mean(values: list[float], window: int) -> float | None:
    if not values:
        return None
    sample = values[-window:]
    return float(np.mean(sample))


def rolling_std(values: list[float], window: int) -> float | None:
    if not values:
        return None
    sample = values[-window:]
    return float(np.std(sample))


def action_entropy(action_counts: np.ndarray) -> float:
    total = int(np.sum(action_counts))
    if total == 0:
        return 0.0
    probs = action_counts.astype(np.float64) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def git_info(repo_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = run_git(["rev-parse", "HEAD"])
    dirty = run_git(["status", "--porcelain"])
    return {
        "git_commit": commit,
        "git_branch": branch,
        "dirty_working_tree": bool(dirty),
    }


def software_info() -> dict[str, Any]:
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    try:
        import gymnasium

        info["gymnasium_version"] = gymnasium.__version__
    except Exception:
        info["gymnasium_version"] = None
    try:
        import ale_py

        info["ale_py_version"] = ale_py.__version__
    except Exception:
        info["ale_py_version"] = None
    return info


def hardware_info(device: torch.device) -> dict[str, Any]:
    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
    cpu_name = platform.processor() or os.uname().machine
    return {
        "device": str(device),
        "gpu_name": gpu_name,
        "cpu_name": cpu_name,
        "ram_gb": None,
    }
