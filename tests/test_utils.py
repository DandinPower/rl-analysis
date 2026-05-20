import os
import random

import numpy as np
import pytest
import torch

from rl_analysis.utils import hardware_info, resolve_device, set_global_seeds


class FakeMPSBackend:
    def __init__(self, *, available: bool, built: bool = True):
        self.available = available
        self.built = built

    def is_available(self):
        return self.available

    def is_built(self):
        return self.built


def test_set_global_seeds_repeats_python_numpy_and_torch_rngs():
    set_global_seeds(123)
    first = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

    set_global_seeds(123)
    second = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

    assert first == second
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", FakeMPSBackend(available=True), raising=False)

    assert resolve_device("auto") == torch.device("cuda")


def test_resolve_device_auto_uses_mps_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", FakeMPSBackend(available=True), raising=False)

    assert resolve_device("auto") == torch.device("mps")


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", FakeMPSBackend(available=False), raising=False)

    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_rejects_unavailable_mps(monkeypatch):
    monkeypatch.setattr(torch.backends, "mps", FakeMPSBackend(available=False, built=False), raising=False)

    with pytest.raises(RuntimeError, match="MPS device requested"):
        resolve_device("mps")


def test_set_global_seeds_seeds_mps_when_available(monkeypatch):
    class FakeTorchMPS:
        seeded_with = None

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def _is_in_bad_fork():
            return False

        @classmethod
        def manual_seed(cls, seed):
            cls.seeded_with = seed

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "mps", FakeTorchMPS, raising=False)

    set_global_seeds(456)

    assert FakeTorchMPS.seeded_with == 456


def test_hardware_info_reports_mps(monkeypatch):
    monkeypatch.setattr(torch.backends, "mps", FakeMPSBackend(available=True, built=True), raising=False)

    info = hardware_info(torch.device("mps"))

    assert info["device"] == "mps"
    assert info["gpu_name"] == "Apple Metal GPU"
    assert info["mps_available"] is True
    assert info["mps_built"] is True
