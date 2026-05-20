"""Dataclass configuration for DQN experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    use_target_network: bool
    use_replay_buffer: bool
    use_double_dqn: bool
    use_dueling_network: bool
    use_prioritized_replay: bool = False
    use_noisy_network: bool = False
    use_reward_clipping: bool = False


VARIANTS: dict[str, VariantSpec] = {
    "dqn": VariantSpec(
        name="dqn",
        label="DQN",
        use_target_network=True,
        use_replay_buffer=True,
        use_double_dqn=False,
        use_dueling_network=False,
    ),
    "dqn_no_target": VariantSpec(
        name="dqn_no_target",
        label="DQN without target network",
        use_target_network=False,
        use_replay_buffer=True,
        use_double_dqn=False,
        use_dueling_network=False,
    ),
    "dqn_no_replay": VariantSpec(
        name="dqn_no_replay",
        label="DQN without replay buffer",
        use_target_network=True,
        use_replay_buffer=False,
        use_double_dqn=False,
        use_dueling_network=False,
    ),
    "double_dqn": VariantSpec(
        name="double_dqn",
        label="Double DQN",
        use_target_network=True,
        use_replay_buffer=True,
        use_double_dqn=True,
        use_dueling_network=False,
    ),
    "dueling_dqn": VariantSpec(
        name="dueling_dqn",
        label="Dueling DQN",
        use_target_network=True,
        use_replay_buffer=True,
        use_double_dqn=False,
        use_dueling_network=True,
    ),
    "double_dueling_dqn": VariantSpec(
        name="double_dueling_dqn",
        label="Double DQN + Dueling DQN",
        use_target_network=True,
        use_replay_buffer=True,
        use_double_dqn=True,
        use_dueling_network=True,
    ),
}

VARIANT_NAMES = tuple(VARIANTS.keys())


@dataclass(frozen=True)
class EnvironmentProfile:
    env_id: str
    slug: str
    observation_type: str
    action_space_type: str
    network_architecture: str
    success_threshold: float | None
    reward_clipping: bool = False
    raw_frame_skip: int | None = None
    repeat_action_probability: float | None = None
    full_action_space: bool | None = None
    frame_skip: int | None = None
    frame_stack: int | None = None
    grayscale: bool | None = None
    resize_shape: tuple[int, int] | None = None
    noop_max: int | None = None
    terminal_on_life_loss: bool | None = None
    max_episode_steps: int | None = None
    score_thresholds: tuple[float, ...] = ()
    reference_random_score: float | None = None
    reference_human_score: float | None = None
    reference_dqn_score: float | None = None


ENV_PROFILES: dict[str, EnvironmentProfile] = {
    "LunarLander-v3": EnvironmentProfile(
        env_id="LunarLander-v3",
        slug="lunarlander_v3",
        observation_type="vector",
        action_space_type="discrete",
        network_architecture="mlp",
        success_threshold=200.0,
        reward_clipping=False,
    ),
    "ALE/Freeway-v5": EnvironmentProfile(
        env_id="ALE/Freeway-v5",
        slug="freeway",
        observation_type="image",
        action_space_type="discrete",
        network_architecture="cnn",
        success_threshold=15.0,
        reward_clipping=True,
        raw_frame_skip=1,
        repeat_action_probability=0.25,
        full_action_space=False,
        frame_skip=4,
        frame_stack=4,
        grayscale=True,
        resize_shape=(84, 84),
        noop_max=30,
        terminal_on_life_loss=False,
        score_thresholds=(5.0, 10.0, 15.0, 22.5),
        reference_random_score=0.0,
        reference_human_score=29.6,
        reference_dqn_score=30.3,
    ),
}

ENV_ALIASES = {
    "lunarlander": "LunarLander-v3",
    "lunarlander-v3": "LunarLander-v3",
    "lunarlander_v3": "LunarLander-v3",
    "freeway": "ALE/Freeway-v5",
    "atari_freeway": "ALE/Freeway-v5",
    "ale/freeway-v5": "ALE/Freeway-v5",
}


@dataclass
class TrainingConfig:
    total_env_steps: int = 500_000
    max_episodes: int | None = None
    num_seeds: int = 5
    eval_frequency_env_steps: int = 10_000
    checkpoint_frequency_env_steps: int = 50_000
    num_eval_episodes: int = 20
    update_log_frequency: int = 100
    system_metrics_frequency_env_steps: int = 10_000
    sustained_threshold_window: int = 5
    device: str = "auto"


@dataclass
class OptimizerConfig:
    name: str = "Adam"
    learning_rate: float = 5e-4
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    gradient_clip_norm: float = 10.0


@dataclass
class DQNHyperparameters:
    gamma: float = 0.99
    batch_size: int = 64
    replay_buffer_size: int = 100_000
    learning_starts: int = 5_000
    train_frequency_env_steps: int = 4
    gradient_steps_per_train: int = 1
    target_update_frequency_env_steps: int = 1_000
    target_update_tau: float = 1.0
    max_grad_norm: float = 10.0


@dataclass
class ExplorationConfig:
    type: str = "epsilon_greedy"
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_env_steps: int = 100_000
    eval_epsilon: float = 0.001
    freeway_up_bias: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.freeway_up_bias <= 1.0:
            raise ValueError("freeway_up_bias must be between 0.0 and 1.0.")

    def epsilon_at(self, env_step: int) -> float:
        if self.epsilon_decay_env_steps <= 0:
            return self.epsilon_final
        fraction = min(1.0, max(0.0, env_step / self.epsilon_decay_env_steps))
        return self.epsilon_start + fraction * (self.epsilon_final - self.epsilon_start)


@dataclass
class NetworkConfig:
    architecture: str = "mlp"
    hidden_layers: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    dueling_aggregation: str | None = None
    conv_layers: list[dict[str, int]] = field(
        default_factory=lambda: [
            {"out_channels": 32, "kernel_size": 8, "stride": 4},
            {"out_channels": 64, "kernel_size": 4, "stride": 2},
            {"out_channels": 64, "kernel_size": 3, "stride": 1},
        ]
    )
    fully_connected_layers: list[int] = field(default_factory=lambda: [512])
    num_parameters: int | None = None
    initializer: str = "orthogonal"


@dataclass
class RunConfig:
    run_id: str
    experiment_group: str
    env: EnvironmentProfile
    variant: VariantSpec
    seed: int
    output_root: Path
    run_dir: Path
    datetime_start: str
    datetime_end: str | None
    status: str
    training: TrainingConfig
    optimizer: OptimizerConfig
    dqn: DQNHyperparameters
    exploration: ExplorationConfig
    network: NetworkConfig


def get_variant_spec(name: str) -> VariantSpec:
    key = name.strip().lower()
    if key not in VARIANTS:
        allowed = ", ".join(VARIANT_NAMES)
        raise ValueError(f"Unknown variant {name!r}. Expected one of: {allowed}.")
    return VARIANTS[key]


def get_env_profile(env_id: str) -> EnvironmentProfile:
    normalized = env_id.strip()
    canonical = ENV_ALIASES.get(normalized.lower(), normalized)
    if canonical not in ENV_PROFILES:
        allowed = ", ".join(ENV_PROFILES)
        raise ValueError(f"Unknown environment {env_id!r}. Expected one of: {allowed}.")
    return ENV_PROFILES[canonical]


def default_training_for_env(env: EnvironmentProfile) -> TrainingConfig:
    if env.env_id == "ALE/Freeway-v5":
        return TrainingConfig(total_env_steps=1_000_000)
    return TrainingConfig()


def default_dqn_for_variant(variant: VariantSpec, env: EnvironmentProfile | None = None) -> DQNHyperparameters:
    dqn = DQNHyperparameters()
    if env is not None and env.env_id == "ALE/Freeway-v5":
        dqn = replace(
            dqn,
            batch_size=32,
            replay_buffer_size=25_000,
            learning_starts=20_000,
            train_frequency_env_steps=4,
            target_update_frequency_env_steps=8_000,
        )
    if not variant.use_replay_buffer:
        dqn = replace(
            dqn,
            batch_size=1,
            replay_buffer_size=0,
            learning_starts=0,
            train_frequency_env_steps=1,
            gradient_steps_per_train=1,
        )
    return dqn


def default_exploration_for_env(env: EnvironmentProfile) -> ExplorationConfig:
    if env.env_id == "ALE/Freeway-v5":
        return ExplorationConfig(epsilon_final=0.01, epsilon_decay_env_steps=250_000)
    return ExplorationConfig()


def default_optimizer_for_env(env: EnvironmentProfile) -> OptimizerConfig:
    if env.env_id == "ALE/Freeway-v5":
        return OptimizerConfig(learning_rate=6.25e-5, epsilon=1.5e-4)
    return OptimizerConfig()


def default_network_for_env(env: EnvironmentProfile, variant: VariantSpec) -> NetworkConfig:
    if env.network_architecture == "cnn":
        return NetworkConfig(
            architecture="cnn",
            hidden_layers=[],
            dueling_aggregation="mean_subtraction" if variant.use_dueling_network else None,
        )
    return NetworkConfig(
        architecture="mlp",
        hidden_layers=[128, 128],
        dueling_aggregation="mean_subtraction" if variant.use_dueling_network else None,
    )


def build_run_config(
    *,
    env_id: str,
    variant_name: str,
    seed: int,
    output_root: str | Path = "output",
    run_id: str | None = None,
    training_overrides: dict[str, Any] | None = None,
    dqn_overrides: dict[str, Any] | None = None,
    exploration_overrides: dict[str, Any] | None = None,
    network_overrides: dict[str, Any] | None = None,
    optimizer_overrides: dict[str, Any] | None = None,
) -> RunConfig:
    env = get_env_profile(env_id)
    variant = get_variant_spec(variant_name)
    output_root_path = Path(output_root)
    resolved_run_id = run_id or f"{env.slug}_{variant.name}_seed{seed}_{_utc_timestamp()}"
    run_dir = output_root_path / env.slug / variant.name / f"seed_{seed}" / resolved_run_id

    training = default_training_for_env(env)
    dqn = default_dqn_for_variant(variant, env)
    exploration = default_exploration_for_env(env)
    network = default_network_for_env(env, variant)
    optimizer = default_optimizer_for_env(env)

    if training_overrides:
        training = replace(training, **training_overrides)
    if dqn_overrides:
        dqn = replace(dqn, **dqn_overrides)
    if exploration_overrides:
        exploration = replace(exploration, **exploration_overrides)
    if env.env_id != "ALE/Freeway-v5" and exploration.freeway_up_bias != 0.0:
        raise ValueError("freeway_up_bias can only be used with ALE/Freeway-v5.")
    if network_overrides:
        network = replace(network, **network_overrides)
    if optimizer_overrides:
        optimizer = replace(optimizer, **optimizer_overrides)

    now = datetime.now(timezone.utc).isoformat()
    return RunConfig(
        run_id=resolved_run_id,
        experiment_group="dqn_ablation_lunarlander_freeway",
        env=env,
        variant=variant,
        seed=seed,
        output_root=output_root_path,
        run_dir=run_dir,
        datetime_start=now,
        datetime_end=None,
        status="created",
        training=training,
        optimizer=optimizer,
        dqn=dqn,
        exploration=exploration,
        network=network,
    )
