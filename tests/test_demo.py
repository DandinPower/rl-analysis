import pytest
import torch
from torch import nn

from rl_analysis import demo
from rl_analysis.config import NetworkConfig
from rl_analysis.networks import build_q_network


def _checkpoint_payload():
    network_config = NetworkConfig(hidden_layers=[4], dueling_aggregation=None)
    network = build_q_network((2,), 3, network_config)
    with torch.no_grad():
        for index, parameter in enumerate(network.parameters()):
            parameter.fill_(0.1 * (index + 1))
    return {
        "run_id": "demo_test",
        "global_env_step": 123,
        "variant": "dqn",
        "seed": 7,
        "online_network_state_dict": network.state_dict(),
        "config": {
            "env": {"env_id": "LunarLander-v3"},
            "network": {
                "architecture": "mlp",
                "hidden_layers": [4],
                "activation": "relu",
                "dueling_aggregation": None,
                "num_parameters": sum(parameter.numel() for parameter in network.parameters()),
            },
        },
    }, network


def test_parse_args_requires_checkpoint():
    with pytest.raises(SystemExit):
        demo.parse_args([])


def test_parse_args_accepts_demo_options(tmp_path):
    args = demo.parse_args(
        [
            "--checkpoint",
            "model_step_100.pt",
            "--episodes",
            "2",
            "--video-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )

    assert args.checkpoint.name == "model_step_100.pt"
    assert args.episodes == 2
    assert args.video_dir == tmp_path
    assert args.device == "cpu"


def test_validate_checkpoint_payload_reports_missing_keys():
    with pytest.raises(ValueError, match="missing required key"):
        demo.validate_checkpoint_payload({"run_id": "incomplete"})


def test_validate_checkpoint_payload_requires_env_id():
    payload, _network = _checkpoint_payload()
    payload["config"]["env"] = {}

    with pytest.raises(ValueError, match="config.env.env_id"):
        demo.validate_checkpoint_payload(payload)


def test_build_network_from_checkpoint_restores_state_dict():
    payload, expected_network = _checkpoint_payload()

    restored_network = demo.build_network_from_checkpoint(
        payload,
        observation_shape=(2,),
        action_dim=3,
        device=torch.device("cpu"),
    )

    observation = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    torch.testing.assert_close(restored_network(observation), expected_network(observation))
    assert not restored_network.training


class FixedQNetwork(nn.Module):
    def forward(self, observations):
        return torch.tensor([[1.0, 4.0, 2.0]], device=observations.device).repeat(observations.shape[0], 1)


def test_select_greedy_action_uses_highest_q_value():
    action = demo.select_greedy_action(
        FixedQNetwork(),
        observation=torch.zeros(2).numpy(),
        device=torch.device("cpu"),
    )

    assert action == 1
