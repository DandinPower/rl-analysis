import numpy as np
import torch
from torch import nn

from rl_analysis.agent import DQNAgent, calculate_dqn_targets
from rl_analysis.config import DQNHyperparameters, NetworkConfig, OptimizerConfig, get_variant_spec
from rl_analysis.networks import build_q_network


def test_mlp_q_network_output_shape():
    network = build_q_network((8,), 4, NetworkConfig(hidden_layers=[16], dueling_aggregation=None))
    output = network(torch.zeros(3, 8))
    assert output.shape == (3, 4)


def test_dueling_q_network_outputs_stream_metrics():
    network = build_q_network((8,), 4, NetworkConfig(hidden_layers=[16], dueling_aggregation="mean_subtraction"))
    q_values, streams = network(torch.zeros(3, 8), return_streams=True)
    assert q_values.shape == (3, 4)
    assert streams["value"].shape == (3, 1)
    assert streams["advantage"].shape == (3, 4)


def test_cnn_q_network_output_shape():
    network = build_q_network((4, 84, 84), 3, NetworkConfig(architecture="cnn", dueling_aggregation=None))
    output = network(torch.zeros(2, 4, 84, 84, dtype=torch.uint8))
    assert output.shape == (2, 3)


def test_dueling_cnn_q_network_outputs_stream_metrics():
    network = build_q_network((4, 84, 84), 3, NetworkConfig(architecture="cnn", dueling_aggregation="mean_subtraction"))
    q_values, streams = network(torch.zeros(2, 4, 84, 84, dtype=torch.uint8), return_streams=True)
    assert q_values.shape == (2, 3)
    assert streams["value"].shape == (2, 1)
    assert streams["advantage"].shape == (2, 3)


class FixedQ(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.register_buffer("rows", torch.tensor(rows, dtype=torch.float32))

    def forward(self, x):
        return self.rows[: x.shape[0]]


class RandomBranchQ(nn.Module):
    action_dim = 3

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros(x.shape[0], self.action_dim, dtype=torch.float32, device=x.device)


def test_select_action_uses_random_action_probabilities_when_exploring():
    agent = DQNAgent(
        online_network=RandomBranchQ(),
        target_network=None,
        variant=get_variant_spec("dqn_no_target"),
        dqn_config=DQNHyperparameters(),
        optimizer_config=OptimizerConfig(),
        device=torch.device("cpu"),
    )
    probabilities = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    actions = [
        agent.select_action(
            np.zeros(1, dtype=np.float32),
            epsilon=1.0,
            rng=np.random.default_rng(seed),
            random_action_probabilities=probabilities,
        )
        for seed in range(10)
    ]

    assert actions == [1] * 10


def test_double_dqn_target_uses_online_selection_and_target_evaluation():
    online = FixedQ([[10.0, 0.0], [0.0, 10.0]])
    target = FixedQ([[1.0, 5.0], [7.0, 3.0]])
    rewards = torch.tensor([1.0, 1.0])
    next_states = torch.zeros(2, 1)
    terminated = torch.tensor([0.0, 0.0])

    double_targets, double_diag = calculate_dqn_targets(
        online_network=online,
        target_network=target,
        rewards=rewards,
        next_states=next_states,
        terminated=terminated,
        gamma=0.5,
        use_double_dqn=True,
    )
    vanilla_targets, _ = calculate_dqn_targets(
        online_network=online,
        target_network=target,
        rewards=rewards,
        next_states=next_states,
        terminated=terminated,
        gamma=0.5,
        use_double_dqn=False,
    )

    np.testing.assert_allclose(double_targets.numpy(), np.array([1.5, 2.5], dtype=np.float32))
    np.testing.assert_allclose(vanilla_targets.numpy(), np.array([3.5, 4.5], dtype=np.float32))
    assert double_diag["estimated_overestimation_reduction"] == 4.0
