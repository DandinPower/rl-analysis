import numpy as np

from rl_analysis.envs import freeway_episode_metrics


def test_freeway_episode_metrics_score_and_action_fractions():
    metrics = freeway_episode_metrics(
        episode_return=12.0,
        episode_length=10,
        action_counts=np.array([3, 6, 1], dtype=np.int64),
        max_score_so_far=18.0,
    )

    assert metrics["score"] == 12
    assert not metrics["zero_score_episode"]
    assert metrics["max_score_so_far"] == 18.0
    assert metrics["action_noop_fraction"] == 0.3
    assert metrics["action_up_fraction"] == 0.6
    assert metrics["action_down_fraction"] == 0.1
