import os
import random

import numpy as np
import torch

from rl_analysis.utils import set_global_seeds


def test_set_global_seeds_repeats_python_numpy_and_torch_rngs():
    set_global_seeds(123)
    first = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

    set_global_seeds(123)
    second = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

    assert first == second
    assert os.environ["PYTHONHASHSEED"] == "123"
