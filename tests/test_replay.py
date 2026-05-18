import numpy as np

from rl_analysis.replay import ReplayBuffer, Transition, replay_diagnostics, transition_to_batch


def make_transition(step: int) -> Transition:
    return Transition(
        state=np.array([step, step + 1], dtype=np.float32),
        action=step % 2,
        reward=float(step),
        next_state=np.array([step + 1, step + 2], dtype=np.float32),
        terminated=False,
        truncated=False,
        global_step=step,
        episode_index=step // 3,
    )


def test_replay_buffer_samples_expected_shapes():
    buffer = ReplayBuffer(capacity=10, observation_shape=(2,))
    for step in range(6):
        buffer.add(make_transition(step))
    batch = buffer.sample(4, np.random.default_rng(0))
    assert batch.states.shape == (4, 2)
    assert batch.next_states.shape == (4, 2)
    assert batch.actions.shape == (4,)
    assert len(buffer) == 6


def test_replay_buffer_preserves_uint8_image_storage():
    buffer = ReplayBuffer(capacity=3, observation_shape=(4, 84, 84), observation_dtype=np.uint8)
    transition = Transition(
        state=np.ones((4, 84, 84), dtype=np.uint8) * 255,
        action=1,
        reward=1.0,
        next_state=np.zeros((4, 84, 84), dtype=np.uint8),
        terminated=False,
        truncated=False,
        global_step=1,
        episode_index=0,
    )
    buffer.add(transition)
    batch = buffer.sample(1, np.random.default_rng(0))
    assert buffer.states.dtype == np.uint8
    assert batch.states.dtype == np.uint8
    assert batch.states.shape == (1, 4, 84, 84)


def test_no_replay_transition_batch_is_single_online_sample():
    batch = transition_to_batch(make_transition(7))
    assert batch.states.shape == (1, 2)
    assert batch.actions.tolist() == [1]
    assert batch.steps.tolist() == [7]


def test_replay_diagnostics_include_sample_age_and_diversity():
    buffer = ReplayBuffer(capacity=10, observation_shape=(2,))
    for step in range(6):
        buffer.add(make_transition(step))
    batch = buffer.sample(4, np.random.default_rng(1))
    diagnostics = replay_diagnostics(batch=batch, replay_buffer=buffer, current_step=10, replay_enabled=True)
    assert diagnostics["enabled"]
    assert diagnostics["buffer_size"] == 6
    assert diagnostics["sample_age_mean"] > 0
    assert diagnostics["sample_unique_episode_count"] >= 1
