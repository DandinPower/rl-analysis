import json

from rl_analysis.config import build_run_config
from rl_analysis.logging_utils import JSONLLogger
from rl_analysis.trainer import DQNTrainer


def test_jsonl_logger_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JSONLLogger(path)
    logger.write({"b": 2, "a": 1})
    logger.close()
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}


def test_trainer_creates_expected_output_files_with_tiny_run(tmp_path):
    config = build_run_config(
        env_id="LunarLander-v3",
        variant_name="dqn",
        seed=0,
        output_root=tmp_path,
        run_id="tiny_run",
        training_overrides={
            "total_env_steps": 20,
            "eval_frequency_env_steps": 10,
            "checkpoint_frequency_env_steps": 20,
            "num_eval_episodes": 1,
            "device": "cpu",
            "update_log_frequency": 1,
            "system_metrics_frequency_env_steps": 20,
        },
        dqn_overrides={
            "learning_starts": 2,
            "batch_size": 2,
            "replay_buffer_size": 32,
            "train_frequency_env_steps": 1,
            "target_update_frequency_env_steps": 10,
        },
        network_overrides={"hidden_layers": [16]},
    )
    summary = DQNTrainer(config, repo_root=tmp_path).run()
    assert summary["training_completed"]
    assert (config.run_dir / "run_config.json").exists()
    assert (config.run_dir / "train_episode_metrics.jsonl").exists()
    assert (config.run_dir / "train_update_metrics.jsonl").exists()
    assert (config.run_dir / "eval_metrics.jsonl").exists()
    assert (config.run_dir / "checkpoint_metrics.jsonl").exists()
    assert (config.run_dir / "system_metrics.jsonl").exists()
    assert (config.run_dir / "summary.json").exists()
    assert list((config.run_dir / "checkpoints").glob("model_step_*.pt"))
