"""Q-network definitions."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rl_analysis.config import NetworkConfig


def _activation(name: str) -> type[nn.Module]:
    key = name.lower()
    if key == "relu":
        return nn.ReLU
    if key == "tanh":
        return nn.Tanh
    if key == "gelu":
        return nn.GELU
    raise ValueError(f"Unsupported activation {name!r}.")


def _orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear | nn.Conv2d):
        nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_layers: list[int],
        activation: str = "relu",
        dueling: bool = False,
    ):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.dueling = bool(dueling)

        act = _activation(activation)
        layers: list[nn.Module] = []
        in_dim = self.observation_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act())
            in_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)

        if self.dueling:
            self.value_head = nn.Linear(in_dim, 1)
            self.advantage_head = nn.Linear(in_dim, self.action_dim)
        else:
            self.q_head = nn.Linear(in_dim, self.action_dim)

        self.apply(_orthogonal_init)

    def forward(self, observations: torch.Tensor, return_streams: bool = False):
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        features = self.feature_extractor(observations.float())
        if self.dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            if return_streams:
                return q_values, {"value": value, "advantage": advantage}
            return q_values
        q_values = self.q_head(features)
        if return_streams:
            return q_values, {"value": None, "advantage": None}
        return q_values


class CnnQNetwork(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_dim: int,
        conv_layers: list[dict[str, int]],
        fully_connected_layers: list[int],
        activation: str = "relu",
        dueling: bool = False,
    ):
        super().__init__()
        if len(observation_shape) != 3:
            raise ValueError(f"CNN QNetwork expects stacked image observations, got shape {observation_shape}.")
        self.observation_shape = tuple(int(x) for x in observation_shape)
        self.action_dim = int(action_dim)
        self.dueling = bool(dueling)

        act = _activation(activation)
        channels = self.observation_shape[0]
        conv_modules: list[nn.Module] = []
        for layer in conv_layers:
            conv_modules.append(
                nn.Conv2d(
                    channels,
                    int(layer["out_channels"]),
                    kernel_size=int(layer["kernel_size"]),
                    stride=int(layer["stride"]),
                )
            )
            conv_modules.append(act())
            channels = int(layer["out_channels"])
        self.conv = nn.Sequential(*conv_modules)

        with torch.no_grad():
            dummy = torch.zeros(1, *self.observation_shape)
            conv_output_dim = int(self.conv(dummy).flatten(start_dim=1).shape[1])

        fc_modules: list[nn.Module] = []
        in_dim = conv_output_dim
        for hidden_dim in fully_connected_layers:
            fc_modules.append(nn.Linear(in_dim, hidden_dim))
            fc_modules.append(act())
            in_dim = hidden_dim
        self.feature_head = nn.Sequential(*fc_modules)

        if self.dueling:
            self.value_head = nn.Linear(in_dim, 1)
            self.advantage_head = nn.Linear(in_dim, self.action_dim)
        else:
            self.q_head = nn.Linear(in_dim, self.action_dim)

        self.apply(_orthogonal_init)

    def forward(self, observations: torch.Tensor, return_streams: bool = False):
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)
        if observations.ndim != 4:
            raise ValueError(f"CNN QNetwork expects observations with 3 or 4 dims, got {observations.ndim}.")
        features = self.conv(observations.float() / 255.0).flatten(start_dim=1)
        features = self.feature_head(features)
        if self.dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            if return_streams:
                return q_values, {"value": value, "advantage": advantage}
            return q_values
        q_values = self.q_head(features)
        if return_streams:
            return q_values, {"value": None, "advantage": None}
        return q_values


def build_q_network(observation_shape: tuple[int, ...], action_dim: int, config: NetworkConfig) -> nn.Module:
    if config.architecture == "cnn":
        return CnnQNetwork(
            observation_shape=observation_shape,
            action_dim=action_dim,
            conv_layers=config.conv_layers,
            fully_connected_layers=config.fully_connected_layers,
            activation=config.activation,
            dueling=config.dueling_aggregation is not None,
        )
    if config.architecture != "mlp":
        raise ValueError(f"Unsupported network architecture {config.architecture!r}.")
    if len(observation_shape) != 1:
        raise ValueError(f"MLP QNetwork expects a flat vector observation, got shape {observation_shape}.")
    return QNetwork(
        observation_dim=observation_shape[0],
        action_dim=action_dim,
        hidden_layers=config.hidden_layers,
        activation=config.activation,
        dueling=config.dueling_aggregation is not None,
    )


def network_architecture_dict(config: NetworkConfig, num_parameters: int) -> dict[str, Any]:
    if config.architecture == "cnn":
        return {
            "architecture": config.architecture,
            "conv_layers": config.conv_layers,
            "fully_connected_layers": config.fully_connected_layers,
            "activation": config.activation,
            "dueling_aggregation": config.dueling_aggregation,
            "num_parameters": num_parameters,
        }
    return {
        "architecture": config.architecture,
        "hidden_layers": config.hidden_layers,
        "activation": config.activation,
        "dueling_aggregation": config.dueling_aggregation,
        "num_parameters": num_parameters,
        "initializer": config.initializer,
    }
