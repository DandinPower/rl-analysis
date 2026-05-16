Below is a full Markdown report section you can paste into your project notes or adapt into the final report. It uses the **maximum logging plan**, not the minimum.

````markdown
# Metrics Logging Plan for DQN Ablation Experiments

## 1. Motivation

This project compares several DQN-based training configurations on two reinforcement learning environments: `LunarLander-v3` and Atari `Freeway`. The goal is not only to determine which agent achieves the highest return, but also to understand *why* different configurations behave differently during training.

The assignment requires meaningful experiments and discussion, not only raw result tables. Therefore, the logging system must preserve enough information to analyze performance, sample efficiency, stability, value estimation behavior, optimization dynamics, replay-buffer effects, and environment-specific behavior.

The tested variants are:

| Variant | What it tests |
| --- | --- |
| DQN | Baseline configuration |
| DQN without target network | Contribution of the target network to training stability |
| DQN without replay buffer | Contribution of experience replay to decorrelation and sample reuse |
| Double DQN | Effect of reducing Q-value overestimation |
| Dueling DQN | Effect of separating state value and action advantage |
| Double DQN + Dueling DQN | Whether reduced overestimation and value-advantage decomposition provide complementary benefits |

The environments are:

| Environment | Purpose |
| --- | --- |
| LunarLander-v3 | Low-dimensional control task with dense reward and discrete actions |
| Atari Freeway | Visual Atari task with sparse, score-based reward and image observations |

Because these environments differ substantially, the logs should support both within-environment comparison and cross-environment discussion.

---

## 2. Logging Design Overview

Each training run should produce multiple JSON or JSONL files. JSONL is preferred for time-series logs because it is appendable and easy to load into pandas.

Recommended file structure:

```text
experiments/
  lunarlander_v3/
    dqn/
      seed_0/
        run_config.json
        train_episode_metrics.jsonl
        train_update_metrics.jsonl
        eval_metrics.jsonl
        checkpoint_metrics.jsonl
        system_metrics.jsonl
        summary.json
    dqn_no_target/
    dqn_no_replay/
    double_dqn/
    dueling_dqn/
    double_dueling_dqn/

  freeway/
    dqn/
    dqn_no_target/
    dqn_no_replay/
    double_dqn/
    dueling_dqn/
    double_dueling_dqn/
````

Each run should have a unique `run_id`.

Example:

```json
{
  "run_id": "lunarlander_v3_dqn_seed0_20260515_001"
}
```

A good `run_id` should encode:

| Field             | Example                              |
| ----------------- | ------------------------------------ |
| Environment       | `lunarlander_v3`                     |
| Variant           | `dqn`, `double_dqn`, `dqn_no_replay` |
| Seed              | `seed0`                              |
| Date or run index | `20260515_001`                       |

---

# 3. Run Configuration Log

File:

```text
run_config.json
```

This file records all immutable configuration information for one run. It is essential for reproducibility and fair comparison.

## 3.1 Required Run Metadata

```json
{
  "run_id": "lunarlander_v3_dqn_seed0_20260515_001",
  "experiment_group": "dqn_ablation_lunarlander_freeway",
  "env_id": "LunarLander-v3",
  "variant": "DQN",
  "seed": 0,
  "datetime_start": "2026-05-15T10:30:00+08:00",
  "datetime_end": null,
  "status": "running"
}
```

### Why this matters

These fields allow aggregation across variants, seeds, and environments. They also help detect incomplete or failed runs.

---

## 3.2 Algorithm Configuration

```json
{
  "algorithm": {
    "base": "DQN",
    "use_target_network": true,
    "use_replay_buffer": true,
    "use_double_dqn": false,
    "use_dueling_network": false,
    "use_prioritized_replay": false,
    "use_noisy_network": false,
    "use_reward_clipping": false
  }
}
```

### Why this matters

This makes the ablation condition explicit. It avoids ambiguity when analyzing results later.

---

## 3.3 Training Budget

```json
{
  "training_budget": {
    "total_env_steps": 500000,
    "max_episodes": null,
    "max_episode_steps": null,
    "num_seeds": 5,
    "eval_frequency_env_steps": 10000,
    "checkpoint_frequency_env_steps": 50000
  }
}
```

### Why this matters

All variants must use the same training budget. Otherwise, the comparison is not fair.

---

## 3.4 Environment Configuration

For LunarLander:

```json
{
  "environment": {
    "env_id": "LunarLander-v3",
    "observation_type": "vector",
    "action_space_type": "discrete",
    "reward_clipping": false,
    "observation_normalization": false,
    "max_episode_steps": null
  }
}
```

For Freeway:

```json
{
  "environment": {
    "env_id": "ALE/Freeway-v5",
    "observation_type": "image",
    "action_space_type": "discrete",
    "frame_skip": 4,
    "frame_stack": 4,
    "grayscale": true,
    "resize_shape": [84, 84],
    "reward_clipping": true,
    "noop_max": 30,
    "terminal_on_life_loss": false,
    "max_episode_steps": null
  }
}
```

### Why this matters

Atari preprocessing can strongly affect performance. Frame stacking, frame skip, reward clipping, no-op starts, and image resizing must be logged because they are part of the experimental condition.

---

## 3.5 Network Architecture

```json
{
  "network": {
    "architecture": "mlp",
    "hidden_layers": [128, 128],
    "activation": "relu",
    "dueling_aggregation": null,
    "num_parameters": 34820,
    "initializer": "orthogonal"
  }
}
```

For Atari:

```json
{
  "network": {
    "architecture": "cnn",
    "conv_layers": [
      {
        "out_channels": 32,
        "kernel_size": 8,
        "stride": 4
      },
      {
        "out_channels": 64,
        "kernel_size": 4,
        "stride": 2
      },
      {
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1
      }
    ],
    "fully_connected_layers": [512],
    "activation": "relu",
    "dueling_aggregation": "mean_subtraction",
    "num_parameters": 1685420
  }
}
```

### Why this matters

Architecture differences can confound algorithmic comparisons. If Dueling DQN uses a different head structure, this should be explicit.

---

## 3.6 Optimizer Configuration

```json
{
  "optimizer": {
    "name": "Adam",
    "learning_rate": 0.0005,
    "epsilon": 1e-8,
    "weight_decay": 0.0,
    "gradient_clip_norm": 10.0
  }
}
```

### Why this matters

Gradient clipping and learning rate can heavily affect DQN stability.

---

## 3.7 DQN Hyperparameters

```json
{
  "dqn_hyperparameters": {
    "gamma": 0.99,
    "batch_size": 64,
    "replay_buffer_size": 100000,
    "learning_starts": 5000,
    "train_frequency_env_steps": 4,
    "gradient_steps_per_train": 1,
    "target_update_frequency_env_steps": 1000,
    "target_update_tau": 1.0,
    "max_grad_norm": 10.0
  }
}
```

For no-replay-buffer variant:

```json
{
  "dqn_hyperparameters": {
    "gamma": 0.99,
    "batch_size": 1,
    "replay_buffer_size": 0,
    "learning_starts": 0,
    "train_frequency_env_steps": 1,
    "gradient_steps_per_train": 1,
    "target_update_frequency_env_steps": 1000,
    "target_update_tau": 1.0
  }
}
```

### Why this matters

The ablation must be clear. For example, “without replay buffer” could mean online updates with batch size 1, or a small recent batch. These are not the same experiment.

---

## 3.8 Exploration Configuration

```json
{
  "exploration": {
    "type": "epsilon_greedy",
    "epsilon_start": 1.0,
    "epsilon_final": 0.05,
    "epsilon_decay_env_steps": 100000,
    "eval_epsilon": 0.001
  }
}
```

### Why this matters

Exploration strongly affects sample efficiency and final behavior. Evaluation should use a fixed low epsilon across all variants.

---

## 3.9 Software and Hardware

```json
{
  "software": {
    "python_version": "3.11.0",
    "gymnasium_version": "PUT_VERSION_HERE",
    "ale_py_version": "PUT_VERSION_HERE",
    "torch_version": "PUT_VERSION_HERE",
    "numpy_version": "PUT_VERSION_HERE"
  },
  "hardware": {
    "device": "cuda",
    "gpu_name": "PUT_GPU_HERE",
    "cpu_name": "PUT_CPU_HERE",
    "ram_gb": 32
  },
  "code": {
    "git_commit": "PUT_COMMIT_HASH_HERE",
    "git_branch": "main",
    "dirty_working_tree": false
  }
}
```

### Why this matters

This helps reproduce the experiment and diagnose unexpected performance differences caused by implementation or system changes.

---

# 4. Training Episode Metrics

File:

```text
train_episode_metrics.jsonl
```

Log one record per training episode.

## 4.1 General Episode Metrics

```json
{
  "run_id": "lunarlander_v3_dqn_seed0_20260515_001",
  "env_id": "LunarLander-v3",
  "variant": "DQN",
  "seed": 0,

  "episode_index": 42,
  "global_env_step": 18432,
  "episode_return": -87.4,
  "episode_length": 712,
  "episode_wall_time_sec": 1.84,

  "epsilon": 0.842,
  "learning_rate": 0.0005,
  "replay_buffer_size": 18432,

  "rolling_return_10": -112.8,
  "rolling_return_50": -96.3,
  "rolling_return_100": -82.7,
  "rolling_length_100": 684.1,

  "terminated": true,
  "truncated": false,
  "termination_reason": "environment"
}
```

## 4.2 Why These Metrics Matter

| Metric                     | Why it matters                                                    |
| -------------------------- | ----------------------------------------------------------------- |
| `episode_return`           | Main training reward signal                                       |
| `episode_length`           | Indicates survival, early failure, or time-limit behavior         |
| `epsilon`                  | Connects behavior to exploration schedule                         |
| `replay_buffer_size`       | Useful for analyzing early training and no-replay variants        |
| `rolling_return_10`        | Short-term learning trend                                         |
| `rolling_return_50`        | Medium-term learning trend                                        |
| `rolling_return_100`       | Stable training curve for reporting                               |
| `terminated` / `truncated` | Distinguishes real environment termination from time-limit cutoff |

---

## 4.3 LunarLander-Specific Episode Metrics

```json
{
  "game_specific": {
    "landing_success": false,
    "crash": true,
    "fuel_proxy_main_engine_count": 84,
    "fuel_proxy_side_engine_count": 37,
    "main_engine_action_fraction": 0.118,
    "side_engine_action_fraction": 0.052,
    "final_x_position": -0.21,
    "final_y_position": 0.03,
    "final_x_velocity": 0.14,
    "final_y_velocity": -0.28,
    "final_angle": 0.07,
    "final_angular_velocity": -0.02,
    "final_left_leg_contact": true,
    "final_right_leg_contact": false
  }
}
```

### Why this matters

LunarLander return alone does not explain behavior. Engine-use counts help analyze whether a policy is inefficient, unstable, or overly aggressive. Final state variables help explain crashes and failed landings.

---

## 4.4 Freeway-Specific Episode Metrics

```json
{
  "game_specific": {
    "score": 12,
    "zero_score_episode": false,
    "max_score_so_far": 18,
    "action_up_fraction": 0.63,
    "action_down_fraction": 0.04,
    "action_noop_fraction": 0.33
  }
}
```

### Why this matters

Freeway is sparse and score-driven. A policy may achieve low return because it fails to move upward, overuses no-op, or oscillates. Action fractions help explain these failure modes.

---

# 5. Training Update Metrics

File:

```text
train_update_metrics.jsonl
```

Log one record every fixed number of gradient updates, for example every 100 updates.

## 5.1 General Update Metrics

```json
{
  "run_id": "lunarlander_v3_dqn_seed0_20260515_001",
  "env_id": "LunarLander-v3",
  "variant": "DQN",
  "seed": 0,

  "global_env_step": 20000,
  "update_index": 3750,

  "loss": {
    "td_loss_mean": 0.421,
    "td_loss_std": 0.198,
    "td_error_mean": -0.031,
    "td_error_abs_mean": 0.514,
    "td_error_abs_median": 0.332,
    "td_error_abs_p90": 1.021,
    "td_error_abs_p95": 1.382,
    "td_error_abs_p99": 2.714
  },

  "q_values": {
    "online_q_mean": 4.21,
    "online_q_std": 2.17,
    "online_q_min": -3.92,
    "online_q_max": 15.83,
    "online_q_max_mean": 7.84,
    "target_q_mean": 3.88,
    "target_q_std": 1.93,
    "target_q_min": -2.77,
    "target_q_max": 13.41,
    "q_overestimation_proxy": 0.33,
    "action_gap_mean": 1.27,
    "action_gap_std": 0.64
  },

  "targets": {
    "bellman_target_mean": 3.91,
    "bellman_target_std": 2.44,
    "bellman_target_min": -5.0,
    "bellman_target_max": 18.2,
    "reward_mean": -0.12,
    "reward_std": 0.83,
    "done_fraction": 0.07
  },

  "optimization": {
    "learning_rate": 0.0005,
    "grad_norm": 3.42,
    "grad_norm_before_clip": 12.7,
    "grad_norm_after_clip": 10.0,
    "grad_norm_clipped": true,
    "param_norm": 18.7,
    "update_wall_time_sec": 0.006
  }
}
```

## 5.2 Why These Metrics Matter

| Metric                   | Why it matters                                                              |
| ------------------------ | --------------------------------------------------------------------------- |
| `td_loss_mean`           | Shows whether Bellman regression is stable                                  |
| `td_error_abs_p95`       | Detects rare but large TD errors that may destabilize learning              |
| `online_q_max_mean`      | Helps detect Q-value overestimation                                         |
| `target_q_mean`          | Useful for comparing vanilla DQN and Double DQN                             |
| `q_overestimation_proxy` | Direct diagnostic for overestimation behavior                               |
| `action_gap_mean`        | Measures how clearly the policy separates the best action from alternatives |
| `grad_norm`              | Detects optimization instability                                            |
| `grad_norm_clipped`      | Shows how often clipping is preventing divergence                           |
| `reward_mean`            | Shows the reward distribution in sampled transitions                        |
| `done_fraction`          | Helps detect whether the sampled batch is dominated by terminal states      |

---

# 6. Replay Buffer Metrics

Replay metrics can be included inside `train_update_metrics.jsonl`.

```json
{
  "replay": {
    "enabled": true,
    "buffer_size": 20000,
    "buffer_capacity": 100000,
    "buffer_fill_fraction": 0.20,

    "sample_age_mean": 7421.5,
    "sample_age_std": 3910.2,
    "sample_age_min": 4,
    "sample_age_max": 18329,
    "sample_age_p50": 6931,
    "sample_age_p90": 12942,

    "sample_reward_mean": -0.12,
    "sample_reward_std": 0.83,
    "sample_done_fraction": 0.07,

    "sample_unique_episode_count": 41,
    "sample_consecutive_transition_fraction": 0.03
  }
}
```

For no-replay-buffer variant:

```json
{
  "replay": {
    "enabled": false,
    "buffer_size": null,
    "buffer_capacity": 0,
    "buffer_fill_fraction": null,
    "sample_age_mean": 0,
    "sample_age_std": 0,
    "sample_unique_episode_count": 1,
    "sample_consecutive_transition_fraction": 1.0
  }
}
```

## Why this matters

Replay buffer ablation cannot be fully explained by return curves alone. These logs show whether replay provides diverse, decorrelated samples. For the no-replay variant, the agent updates from highly recent and correlated transitions, which can increase variance and instability.

---

# 7. Target Network Metrics

Target-network metrics should also be included in `train_update_metrics.jsonl`.

```json
{
  "target_network": {
    "enabled": true,
    "target_update_count": 20,
    "steps_since_target_update": 0,
    "target_update_frequency_env_steps": 1000,
    "target_update_tau": 1.0,
    "online_target_param_l2": 0.0,
    "online_target_param_l2_mean_per_param": 0.0
  }
}
```

For no-target-network variant:

```json
{
  "target_network": {
    "enabled": false,
    "target_update_count": null,
    "steps_since_target_update": null,
    "target_update_frequency_env_steps": null,
    "target_update_tau": null,
    "online_target_param_l2": null,
    "online_target_param_l2_mean_per_param": null
  }
}
```

## Why this matters

The target network stabilizes bootstrapping by preventing the target from changing at every update. Without it, TD errors, Q-values, and gradients may become unstable. These logs allow the discussion to connect instability with the absence of a delayed target.

---

# 8. Double DQN Metrics

For Double DQN variants, log the action-selection and action-evaluation behavior separately.

```json
{
  "double_dqn": {
    "enabled": true,
    "online_selected_action_q_mean": 7.84,
    "target_evaluated_selected_action_q_mean": 6.91,
    "vanilla_max_target_q_mean_proxy": 7.46,
    "double_dqn_target_q_mean": 6.91,
    "estimated_overestimation_reduction": 0.55
  }
}
```

For vanilla DQN:

```json
{
  "double_dqn": {
    "enabled": false,
    "online_selected_action_q_mean": null,
    "target_evaluated_selected_action_q_mean": null,
    "vanilla_max_target_q_mean_proxy": 7.46,
    "double_dqn_target_q_mean": null,
    "estimated_overestimation_reduction": null
  }
}
```

## Why this matters

Double DQN is specifically designed to reduce overestimation caused by using the same value function for action selection and evaluation. These metrics allow the report to show whether Double DQN actually reduced Q-value inflation.

---

# 9. Dueling Network Metrics

For Dueling DQN variants, log value-stream and advantage-stream statistics.

```json
{
  "dueling": {
    "enabled": true,
    "value_stream_mean": 4.82,
    "value_stream_std": 1.41,
    "advantage_stream_mean": 0.02,
    "advantage_stream_std": 0.94,
    "advantage_abs_mean": 0.71,
    "advantage_max_mean": 1.38,
    "advantage_min_mean": -1.24,
    "action_gap_mean": 1.27,
    "action_gap_std": 0.64
  }
}
```

For non-dueling variants:

```json
{
  "dueling": {
    "enabled": false,
    "value_stream_mean": null,
    "value_stream_std": null,
    "advantage_stream_mean": null,
    "advantage_stream_std": null,
    "advantage_abs_mean": null,
    "action_gap_mean": null,
    "action_gap_std": null
  }
}
```

## Why this matters

Dueling DQN is useful when many actions have similar value but the state value itself is important. Logging value and advantage statistics allows analysis of whether the architecture learns a meaningful decomposition.

---

# 10. Evaluation Metrics

File:

```text
eval_metrics.jsonl
```

Log one record per evaluation phase. Evaluation should be separate from training because training returns include exploration noise.

## 10.1 General Evaluation Metrics

```json
{
  "run_id": "lunarlander_v3_dqn_seed0_20260515_001",
  "env_id": "LunarLander-v3",
  "variant": "DQN",
  "seed": 0,

  "global_env_step": 100000,
  "eval_index": 10,
  "num_eval_episodes": 20,
  "eval_epsilon": 0.001,

  "return_mean": 184.2,
  "return_std": 42.7,
  "return_median": 198.4,
  "return_min": 82.1,
  "return_max": 254.8,
  "return_p25": 152.3,
  "return_p75": 221.9,

  "episode_length_mean": 731.5,
  "episode_length_std": 103.2,

  "best_eval_return_mean_so_far": 184.2,
  "best_eval_step_so_far": 100000,

  "action_entropy_mean": 1.24,
  "action_entropy_std": 0.11,
  "action_counts_mean": [120, 340, 95, 210]
}
```

## 10.2 LunarLander Evaluation Metrics

```json
{
  "lunarlander_eval": {
    "success_rate": 0.65,
    "success_definition": "return >= 200",
    "landing_success_rate": 0.60,
    "crash_rate": 0.25,
    "timeout_rate": 0.15,
    "main_engine_action_fraction_mean": 0.12,
    "side_engine_action_fraction_mean": 0.05,
    "both_legs_contact_rate": 0.58,
    "one_leg_contact_rate": 0.17,
    "no_leg_contact_rate": 0.25
  }
}
```

## 10.3 Freeway Evaluation Metrics

```json
{
  "freeway_eval": {
    "score_mean": 18.6,
    "score_std": 4.1,
    "score_median": 19.0,
    "score_min": 10,
    "score_max": 25,
    "zero_score_rate": 0.0,
    "action_up_fraction_mean": 0.63,
    "action_down_fraction_mean": 0.04,
    "action_noop_fraction_mean": 0.33
  }
}
```

## Why this matters

Evaluation logs are the main source for fair performance comparison. They should be used for final tables, learning curves, sample-efficiency analysis, and stability analysis.

---

# 11. Checkpoint Metrics

File:

```text
checkpoint_metrics.jsonl
```

Log when saving a model checkpoint.

```json
{
  "run_id": "lunarlander_v3_dqn_seed0_20260515_001",
  "global_env_step": 100000,
  "checkpoint_index": 2,
  "checkpoint_path": "checkpoints/model_step_100000.pt",

  "eval_return_mean_at_checkpoint": 184.2,
  "eval_return_std_at_checkpoint": 42.7,
  "is_best_checkpoint_so_far": true,

  "model_size_mb": 6.4,
  "optimizer_state_size_mb": 12.8
}
```

## Why this matters

Checkpoint logs make it possible to recover the best model, not just the final model. This matters because unstable variants may learn temporarily and later collapse.

---

# 12. System and Runtime Metrics

File:

```text
system_metrics.jsonl
```

Log periodically, for example every 10,000 environment steps.

```json
{
  "run_id": "lunarlander_v3_dqn_seed0_20260515_001",
  "global_env_step": 100000,
  "wall_time_elapsed_sec": 842.5,
  "env_steps_per_second": 118.7,
  "updates_per_second": 213.4,

  "cpu_percent": 62.1,
  "ram_used_gb": 9.8,
  "gpu_util_percent": 71.4,
  "gpu_memory_used_gb": 3.2,

  "replay_buffer_memory_gb": 1.1
}
```

## Why this matters

Some variants may improve performance but cost more compute. System metrics help discuss practical tradeoffs.

---

# 13. Failure and Safety Diagnostics

Failure flags should be logged in both update metrics and final summary.

```json
{
  "failure_diagnostics": {
    "nan_detected": false,
    "inf_detected": false,
    "q_value_explosion": false,
    "q_value_explosion_threshold": 1000.0,
    "loss_explosion": false,
    "loss_explosion_threshold": 1000.0,
    "gradient_explosion": false,
    "gradient_explosion_threshold": 100.0,
    "early_terminated": false,
    "early_terminated_reason": null
  }
}
```

## Why this matters

No-target-network and no-replay-buffer variants may fail catastrophically. Logging failure conditions prevents vague discussion such as “training became unstable.” Instead, the report can say exactly what failed.

---

# 14. Final Summary Metrics

File:

```text
summary.json
```

This file is generated after training finishes. It contains derived metrics used for final tables.

```json
{
  "run_id": "lunarlander_v3_dqn_seed0_20260515_001",
  "env_id": "LunarLander-v3",
  "variant": "DQN",
  "seed": 0,

  "training_completed": true,
  "total_env_steps": 500000,
  "total_episodes": 812,
  "total_updates": 123750,

  "performance": {
    "final_train_return_mean_last_100": 221.4,
    "final_train_return_std_last_100": 38.2,
    "final_eval_return_mean": 232.7,
    "final_eval_return_std": 21.5,
    "final_eval_return_median": 235.9,
    "best_eval_return_mean": 241.8,
    "best_eval_step": 430000,
    "area_under_eval_curve": 74200000.0,
    "normalized_area_under_eval_curve": 0.71
  },

  "sample_efficiency": {
    "threshold": 200,
    "first_step_reaching_threshold": 310000,
    "sustained_threshold_window": 5,
    "first_step_sustained_threshold": 350000,
    "never_reached_threshold": false
  },

  "stability": {
    "rolling_return_std_mean": 46.2,
    "rolling_return_std_max": 118.7,
    "eval_return_std_across_time": 57.4,
    "catastrophic_collapse_count": 1,
    "largest_eval_drop": 83.4,
    "largest_eval_drop_fraction": 0.35,
    "collapse_definition": "eval_return_mean drops by >= 30% from previous best after crossing threshold"
  },

  "optimization": {
    "td_loss_mean_last_10pct": 0.312,
    "td_error_abs_mean_last_10pct": 0.441,
    "td_error_abs_p95_last_10pct": 1.72,
    "grad_norm_mean_last_10pct": 2.81,
    "grad_norm_max": 28.4,
    "gradient_clip_fraction": 0.08
  },

  "q_diagnostics": {
    "online_q_mean_last_10pct": 6.21,
    "online_q_max_mean_last_10pct": 9.43,
    "target_q_mean_last_10pct": 8.88,
    "mean_q_overestimation_proxy_last_10pct": 0.55,
    "max_q_value_seen": 37.9
  },

  "replay_diagnostics": {
    "replay_enabled": true,
    "final_buffer_size": 100000,
    "sample_age_mean_last_10pct": 42133.2,
    "sample_consecutive_transition_fraction_last_10pct": 0.02
  },

  "target_network_diagnostics": {
    "target_network_enabled": true,
    "target_update_count": 500,
    "online_target_param_l2_mean": 0.84,
    "online_target_param_l2_max": 2.91
  },

  "compute": {
    "wall_time_total_sec": 3842.1,
    "env_steps_per_second": 130.1,
    "updates_per_second": 241.8
  },

  "failure_diagnostics": {
    "nan_detected": false,
    "inf_detected": false,
    "q_value_explosion": false,
    "loss_explosion": false,
    "gradient_explosion": false,
    "early_terminated": false
  }
}
```

---

# 15. Cross-Seed Aggregated Metrics

After all seeds finish, generate one aggregate file per variant and environment.

File:

```text
aggregate_summary.json
```

Example:

```json
{
  "env_id": "LunarLander-v3",
  "variant": "DQN",
  "num_seeds": 5,
  "seeds": [0, 1, 2, 3, 4],

  "final_eval_return": {
    "mean": 228.4,
    "std": 24.7,
    "min": 197.2,
    "max": 254.8,
    "median": 232.1
  },

  "best_eval_return": {
    "mean": 246.9,
    "std": 18.3
  },

  "area_under_eval_curve": {
    "mean": 73500000.0,
    "std": 6200000.0
  },

  "steps_to_threshold": {
    "threshold": 200,
    "mean": 326000,
    "std": 48000,
    "success_rate": 0.8,
    "num_seeds_reached": 4
  },

  "stability": {
    "catastrophic_collapse_count_mean": 0.6,
    "largest_eval_drop_mean": 72.4,
    "eval_return_across_seed_std_mean": 24.7
  },

  "failure_rate": {
    "nan_rate": 0.0,
    "q_value_explosion_rate": 0.0,
    "early_termination_rate": 0.0
  }
}
```

## Why this matters

RL results from one seed are weak evidence. Cross-seed aggregation is necessary to discuss reliability.

---

# 16. Derived Metrics Definitions

## 16.1 Final Performance

```text
Final performance = mean evaluation return at the final evaluation point.
```

Use:

```text
mean ± standard deviation across seeds
```

## 16.2 Best Performance

```text
Best performance = maximum evaluation return mean achieved during training.
```

This is useful because unstable agents may reach high performance temporarily and later collapse.

## 16.3 Area Under Evaluation Curve

```text
AUC = area under the evaluation return curve over environment steps.
```

This measures both learning speed and performance.

A normalized version can be computed as:

```text
normalized_AUC = AUC / total_env_steps
```

## 16.4 Steps to Threshold

For LunarLander:

```text
Threshold = evaluation return mean >= 200
```

For Freeway:

```text
Threshold = a task-specific score threshold based on the observed DQN baseline.
```

For example:

```text
Freeway threshold = 50% or 75% of the best DQN baseline score.
```

## 16.5 Sustained Steps to Threshold

```text
Sustained threshold is reached if the evaluation return stays above the threshold for K consecutive evaluation points.
```

Recommended:

```text
K = 3 or 5
```

This avoids overcounting lucky temporary spikes.

## 16.6 Catastrophic Collapse

Recommended relative definition:

```text
A catastrophic collapse occurs if evaluation return drops by at least 30% from the best previous evaluation return after the agent has crossed the success threshold.
```

For LunarLander, an absolute definition may also be used:

```text
A collapse occurs if evaluation return mean previously exceeded 200 and later drops below 140.
```

## 16.7 Q-Value Overestimation Proxy

```text
q_overestimation_proxy = online_q_max_mean - target_q_mean
```

This is not a perfect measurement of true overestimation, but it is useful for comparing vanilla DQN and Double DQN.

## 16.8 Action Gap

```text
action_gap = max_a Q(s, a) - second_max_a Q(s, a)
```

This measures how clearly the agent distinguishes the best action from the next-best action.

---

# 17. Mapping Logs to Research Questions

| Research question                                     | Logs needed                                                            |
| ----------------------------------------------------- | ---------------------------------------------------------------------- |
| Which variant performs best?                          | Final eval return, best eval return, aggregate summary                 |
| Which variant learns fastest?                         | AUC, steps to threshold, sustained threshold                           |
| Which variant is most stable?                         | Rolling return std, across-seed std, collapse count, largest eval drop |
| Does the target network stabilize training?           | TD error, Q-value scale, gradient norm, target-network diagnostics     |
| Does replay buffer improve learning?                  | Replay sample age, sample diversity, rolling variance, TD error        |
| Does Double DQN reduce overestimation?                | Q-max, target Q, overestimation proxy                                  |
| Does Dueling DQN improve action-value representation? | Value stream, advantage stream, action gap                             |
| Are results environment-dependent?                    | Same metrics compared across LunarLander and Freeway                   |
| Are gains worth the compute cost?                     | Wall-clock time, steps per second, updates per second                  |
| Did any method fail catastrophically?                 | Failure flags, collapse count, Q-value explosion, loss explosion       |

---