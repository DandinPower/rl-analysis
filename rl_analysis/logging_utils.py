"""JSON and JSONL logging helpers for experiment artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from rl_analysis.utils import to_jsonable, write_json


class JSONLLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        import json

        self._handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class RunLoggers:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.train_episode = JSONLLogger(run_dir / "train_episode_metrics.jsonl")
        self.train_update = JSONLLogger(run_dir / "train_update_metrics.jsonl")
        self.eval = JSONLLogger(run_dir / "eval_metrics.jsonl")
        self.checkpoint = JSONLLogger(run_dir / "checkpoint_metrics.jsonl")
        self.system = JSONLLogger(run_dir / "system_metrics.jsonl")

    def close(self) -> None:
        self.train_episode.close()
        self.train_update.close()
        self.eval.close()
        self.checkpoint.close()
        self.system.close()

    def __enter__(self) -> "RunLoggers":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def write_run_config(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "run_config.json", payload)


def _mps_memory_gb(name: str) -> float | None:
    if not hasattr(torch, "mps"):
        return None
    reader = getattr(torch.mps, name, None)
    if reader is None:
        return None
    try:
        return float(reader() / 1024**3)
    except Exception:
        return None


def collect_system_metrics(
    *,
    run_id: str,
    global_env_step: int,
    device: torch.device,
    wall_time_start: float,
    env_steps_per_second: float | None,
    updates_per_second: float | None,
    replay_buffer_memory_gb: float | None,
) -> dict[str, Any]:
    cpu_percent = None
    ram_used_gb = None
    try:
        import psutil

        cpu_percent = float(psutil.cpu_percent(interval=None))
        ram_used_gb = float(psutil.virtual_memory().used / 1024**3)
    except Exception:
        pass

    gpu_util_percent = None
    gpu_memory_used_gb = None
    mps_memory_allocated_gb = None
    mps_driver_allocated_gb = None
    mps_recommended_max_memory_gb = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_memory_used_gb = float(torch.cuda.memory_allocated() / 1024**3)
    elif device.type == "mps":
        mps_memory_allocated_gb = _mps_memory_gb("current_allocated_memory")
        mps_driver_allocated_gb = _mps_memory_gb("driver_allocated_memory")
        mps_recommended_max_memory_gb = _mps_memory_gb("recommended_max_memory")
        gpu_memory_used_gb = mps_memory_allocated_gb

    return {
        "run_id": run_id,
        "global_env_step": global_env_step,
        "device": str(device),
        "wall_time_elapsed_sec": time.perf_counter() - wall_time_start,
        "env_steps_per_second": env_steps_per_second,
        "updates_per_second": updates_per_second,
        "cpu_percent": cpu_percent,
        "ram_used_gb": ram_used_gb,
        "gpu_util_percent": gpu_util_percent,
        "gpu_memory_used_gb": gpu_memory_used_gb,
        "mps_memory_allocated_gb": mps_memory_allocated_gb,
        "mps_driver_allocated_gb": mps_driver_allocated_gb,
        "mps_recommended_max_memory_gb": mps_recommended_max_memory_gb,
        "replay_buffer_memory_gb": replay_buffer_memory_gb,
    }
