# Metrics Tutorial for LunarLander DQN Experiments

This guide explains how to read the result logs in this repository and what to compare when analyzing DQN-style reinforcement learning runs. It assumes you already understand normal ML training ideas such as optimization loss, validation curves, overfitting, randomness, and ablation studies. The main difference is that in RL the model changes the data it will see next. A better policy visits different states, gets different rewards, and creates a different training distribution.

## 1. Start With the Question, Not the Loss

For supervised learning, a lower validation loss is often close to the final goal. For DQN, the TD loss is only the Bellman regression loss on sampled transitions. A low TD loss can mean the value function is accurate, but it can also mean the replay data is easy, narrow, or unhelpful. The first question should always be: does the policy actually get better return in evaluation?

In these logs, the main policy-quality fields are in `eval_metrics.jsonl` and `summary.json`.

`return_mean` is the average evaluation return over a fixed number of evaluation episodes. This is the closest metric to final task performance. For LunarLander, a return above 200 is usually treated as solved.

`return_std`, `return_p25`, `return_p75`, `return_min`, and `return_max` show how much one evaluation batch varies. High within-evaluation variance means the same checkpoint can sometimes land well and sometimes fail.

`final_eval_return_mean` in `summary.json` is the last evaluation return. It answers: how good was the policy at the end of training?

`best_eval_return_mean` answers: did the run ever find a good policy? This matters because DQN can learn and later regress.

`best_final_gap`, computed as best return minus final return, answers: did the run keep the good policy or lose it later?

## 2. Use Learning Curves to Separate Speed From Final Quality

A single final score hides the training path. In DQN, two runs can have the same final return but very different stories. One may learn early and stay stable. Another may fail for most of training and recover near the end. The evaluation learning curve shows this.

Check `global_env_step` against `return_mean`. Then compare variants at the same environment step, not at the same wall-clock time. Environment steps measure sample budget and make the ablation fair.

Useful questions:

Does the curve cross 200 early, late, or never?

Does it stay above 200 after crossing, or does it collapse?

Does it improve smoothly, or jump sharply after a long flat period?

Does the mean curve hide large seed variance?

`area_under_eval_curve` and `normalized_area_under_eval_curve` summarize the whole learning curve. In this report, normalized AUC is roughly the average evaluation return over the training budget. A high normalized AUC means the agent was useful for more of training, not only at the end.

## 3. Sample Efficiency: How Much Experience Was Needed?

Sample efficiency is about how many environment interactions were needed before the agent became good. In these logs, the main fields are:

`first_step_reaching_threshold`: the first evaluation step where mean return reached the success threshold.

`first_step_sustained_threshold`: the first step where the run stayed above threshold for a window of evaluations.

`never_reached_threshold`: whether the run failed to reach the threshold at all.

For a fair comparison, use the same threshold for every variant. For LunarLander, this project uses return >= 200. Compare the median first reach step across seeds, but also report how many seeds reached it. A variant with a fast median over only two successful seeds is not better than a variant that reaches the threshold in all seeds.

## 4. Stability: RL Can Learn and Then Forget

DQN is bootstrapped: it learns Q-values from targets that depend on other Q-values. This can create feedback loops. A policy can become good and then get worse as the value estimates shift.

The main stability fields are:

`catastrophic_collapse_count`: how many times evaluation return dropped by a large fraction after the run had already crossed the success threshold.

`largest_eval_drop`: the largest absolute evaluation drop after a previous best.

`eval_return_std_across_time`: how much the evaluation return moved over training.

`rolling_return_std_mean`: how noisy the recent training returns were.

Always interpret collapse count with threshold reach rate. A run that never learns may have zero collapses only because it never reached a level from which it could collapse.

## 5. Seeds Matter More Than in Many Supervised Runs

RL training is sensitive to random seeds because the seed affects initialization, exploration actions, replay contents, and sometimes environment transitions. Do not trust one seed.

For each variant, compare:

Mean final return across seeds.

Seed standard deviation.

95 percent confidence interval of the seed mean.

How many seeds solved the task at the end.

How many seeds ever reached the threshold.

The per-seed table is also important. If one seed fails completely while four seeds are excellent, the mean may look acceptable but the method is unreliable.

When the same seed IDs are shared across variants, paired seed comparisons are useful. For seed 0, compare every variant against baseline seed 0; for seed 1, compare against baseline seed 1; and so on. This reduces noise from lucky or unlucky seeds.

## 6. Replay Buffer Diagnostics

Experience replay is one of the central DQN stabilizers. It breaks short-term correlation and allows each transition to be reused for multiple updates.

Important fields:

`replay_enabled`: whether replay was used.

`final_buffer_size`: how many transitions were in the buffer at the end.

`sample_age_mean_last_10pct`: how old sampled transitions were late in training. A large sample age means updates used a mix of older and newer experience.

`sample_consecutive_transition_fraction_last_10pct`: how often sampled transitions were adjacent in the original trajectory. Lower values mean better decorrelation.

If replay is removed, the agent usually trains on recent correlated data. That can make learning more like chasing a moving target from a narrow stream of experience. In the report, the no-replay ablation should be read mostly as a test of decorrelation and sample reuse.

## 7. Target Network Diagnostics

DQN uses a target network because the Bellman target contains a neural network prediction. If the same network is used for both prediction and target, the target moves every update. This can make bootstrapping unstable.

Important fields:

`target_network_enabled`: whether a separate target network was used.

`target_update_count`: how many hard or soft target updates occurred.

`online_target_param_l2_mean`: how far the online and target networks were from each other on average.

`target_q_mean_last_10pct`: the mean target-side Q-value late in training.

Removing the target network can sometimes still solve a simple environment, but the expected risk is more oscillation, larger collapses, or worse retention.

## 8. TD Error, TD Loss, and Gradient Metrics

TD error is the difference between the current Q estimate and the Bellman target. It is the DQN analogue of a regression residual, but the target itself is learned and non-stationary.

Important fields:

`td_loss_mean_last_10pct`: average TD loss late in training.

`td_error_abs_mean_last_10pct`: mean absolute TD error late in training.

`td_error_abs_p95_last_10pct`: tail TD error. This is often more useful than the mean because rare bad targets can destabilize training.

`grad_norm_mean_last_10pct` and `grad_norm_max`: gradient scale.

`gradient_clip_fraction`: fraction of updates where clipping was active.

High clipping is not automatically bad. It may mean the optimizer is being protected from large TD errors. But if clipping is high and returns are poor, it suggests the update targets are noisy or unstable.

## 9. Q-Value Diagnostics and Overestimation

DQN can overestimate action values because the max over noisy Q estimates tends to select overestimated actions. Double DQN tries to reduce this by using one network to select the action and the other network to evaluate it.

Important fields:

`online_q_mean_last_10pct`: average online Q-value late in training.

`online_q_max_mean_last_10pct`: average max action value late in training.

`target_q_mean_last_10pct`: average target Q-value late in training.

`q_overestimation_proxy_last10`: a proxy comparing online max values to target values.

`max_q_value_seen`: the largest Q-value observed during training.

Interpret Q metrics together with return. A higher Q scale is not better by itself. If Q-values rise while returns do not, the value function may be overconfident. If Double DQN improves return but the proxy does not decrease, that means this proxy is not capturing the whole effect, or the variant reached a different value scale.

## 10. Dueling Network Diagnostics

Dueling DQN separates Q(s, a) into a state-value stream and an action-advantage stream. This helps when many actions have similar value in a state, because the network can learn that the state is good or bad without having to relearn that fact independently for each action.

Useful logged fields in update metrics include:

`dueling.value_stream_mean` and `dueling.value_stream_std`.

`dueling.advantage_abs_mean`.

`dueling.action_gap_mean`.

The action gap is the difference between the best action value and the next alternatives. A larger useful gap can make the greedy policy more decisive, but a too-large gap with bad return can mean the network is confidently wrong.

## 11. LunarLander-Specific Behavior Metrics

Return tells you whether the agent did well. Environment-specific metrics help explain how.

Useful fields under `lunarlander_eval`:

`success_rate`: fraction of evaluation episodes with return >= 200.

`landing_success_rate`: fraction of episodes ending in a successful landing condition.

`crash_rate`: fraction of episodes ending in a crash.

`timeout_rate`: fraction of episodes that reached the time limit.

`both_legs_contact_rate`, `one_leg_contact_rate`, and `no_leg_contact_rate`: final contact behavior.

`main_engine_action_fraction_mean` and `side_engine_action_fraction_mean`: engine usage. High main-engine use may indicate hovering or inefficient control. High side-engine use may indicate strong attitude correction.

`episode_length_mean`: long episodes can mean stable hovering, slow landing, or timeouts. Use it with success, crash, and timeout rates.

## 12. A Practical Comparison Checklist

For every variant, check these in order.

First, final evaluation return and final success rate. This answers whether the final policy is useful.

Second, best return and best-final gap. This answers whether the run ever learned and whether it retained the learned behavior.

Third, normalized AUC and first reach step. This answers whether learning was sample-efficient.

Fourth, seed standard deviation and paired seed deltas. This answers whether the result is reliable or seed-dependent.

Fifth, collapse count and largest drop. This answers whether the method is stable after success.

Sixth, TD loss, TD error tail, gradient clipping, and Q-value scale. This helps explain training dynamics.

Seventh, LunarLander behavior metrics. This turns scores into a behavioral story: landing, crashing, hovering, timing out, or wasting fuel.

The figures generated by `analysis/analyze_lunarlander.py` follow this checklist: learning curves, final return distributions, sample efficiency, stability, optimization and Q diagnostics, and LunarLander behavior.
