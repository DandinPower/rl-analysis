# Metrics Tutorial for LunarLander DQN Experiments

This guide explains how to read the result logs in this repository and what to compare when analyzing DQN-style reinforcement learning runs. It assumes you already understand normal ML training ideas such as optimization loss, validation curves, overfitting, randomness, and ablation studies. The main difference is that in RL the model changes the data it will see next. A better policy visits different states, gets different rewards, and creates a different training distribution.

## 1. Start With the Question, Not the Loss

For supervised learning, a lower validation loss is often close to the final goal. For DQN, the TD loss is only the Bellman regression loss on sampled transitions. A low TD loss can mean the value function is accurate, but it can also mean the replay data is easy, narrow, or unhelpful. The first question should always be: does the policy actually get better return in evaluation?

In these logs, the main policy-quality fields are in `eval_metrics.jsonl` and `summary.json`.

When this guide uses a statistic word such as mean, average, standard deviation, percentile, minimum, maximum, median, or confidence interval, read it together with the group being summarized. The group may be evaluation episodes from one checkpoint, random seeds from one variant, evaluation checkpoints across one training run, or update batches from the last part of training. For example, a mean over evaluation episodes describes one checkpoint, while a mean over seeds describes one variant across repeated runs.

An evaluation checkpoint is not a training episode. It is a measurement taken after the agent has already collected some amount of training experience. For example, `global_env_step = 160000` means the agent has collected 160000 training environment steps, then the current policy is evaluated separately. The evaluation episodes are dedicated test runs of that checkpoint policy, usually without the same exploration behavior used during training. Therefore, `return_mean` is related to training only by timing: it tells you how well the current checkpoint performs after that much training experience.

`return_mean` is the arithmetic mean of the returns from one evaluation batch. If evaluation runs 20 dedicated evaluation episodes at one checkpoint, add the 20 episode returns and divide by 20. This value represents the center of that checkpoint's evaluation episode-return range. It is the closest metric to final task performance. For LunarLander, a return above 200 is usually treated as solved.

`return_std`, `return_p25`, `return_p75`, `return_min`, and `return_max` describe the spread of episode returns inside the same evaluation batch. `return_std` is the standard deviation across those evaluation episodes. `return_p25` and `return_p75` are the 25th and 75th percentiles across those episodes. `return_min` and `return_max` are the lowest and highest episode returns in that batch. High within-evaluation spread means the same checkpoint can sometimes land well and sometimes fail.

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

Does the mean curve hide large seed variance? Here the mean curve usually means the average of `return_mean` values across seeds at the same evaluation step.

`area_under_eval_curve` summarizes the whole evaluation learning curve for one run. The x-axis is `global_env_step`, which is how many environment interactions have happened. The y-axis is `return_mean`, which is the mean return across evaluation episodes at that checkpoint. The area is large when the policy reaches high evaluation return early and stays high for many environment steps.

The calculation uses the trapezoid rule. Take each logged evaluation point from `eval_metrics.jsonl` as a pair `(global_env_step, return_mean)`. For each neighboring pair, draw a straight line between the two returns. The area for that interval is the width in environment steps multiplied by the average height of the two endpoint returns:

`area = sum((step_i - step_{i-1}) * (return_i + return_{i-1}) / 2)`

You can calculate it by hand in three steps. First, sort the evaluation rows by `global_env_step`. Second, for each adjacent pair, compute `step gap = later step - earlier step` and `average return = (earlier return_mean + later return_mean) / 2`. Third, multiply `step gap * average return` for each pair and add the interval areas.

For example, suppose the logged evaluation points are `(100000, -100)`, `(200000, 50)`, and `(300000, 200)`. The first interval has width `200000 - 100000 = 100000` steps and average return `(-100 + 50) / 2 = -25`, so its area is `100000 * -25 = -2500000`. The second interval has width `300000 - 200000 = 100000` steps and average return `(50 + 200) / 2 = 125`, so its area is `100000 * 125 = 12500000`. The total `area_under_eval_curve` is `-2500000 + 12500000 = 10000000`.

The area can be negative if returns are negative for enough of training. Its unit is return-times-steps, not plain return. The implementation only uses intervals between logged evaluation points. It does not add an extra interval from step 0 unless step 0 is actually logged.

`normalized_area_under_eval_curve` divides that area by the configured training step budget:

`normalized AUC = area_under_eval_curve / total_env_steps`

If the training budget is `300000` steps in the example above, normalized AUC is `10000000 / 300000 = 33.33`. The result is in return units, so it is easier to compare with normal return values. It is not scaled to a 0 to 1 range. Because the numerator only includes logged intervals, the value is best read as the training-budget-normalized area from the logged evaluation curve. When evaluation logging covers the training run regularly, it behaves like a step-weighted average evaluation return: each return value influences the summary in proportion to how many environment steps it covers. A higher normalized AUC means the agent learned earlier, kept return high longer, or both.

## 3. Sample Efficiency: How Much Experience Was Needed?

Sample efficiency is about how many environment interactions were needed before the agent became good. In these logs, the main fields are:

`first_step_reaching_threshold`: the first evaluation checkpoint where `return_mean` reached the success threshold. If the threshold is 200 and the 160000-step evaluation has episode returns whose average is 205, then the run reached the threshold at 160000 steps, unless an earlier evaluation also averaged at least 200.

`first_step_sustained_threshold`: the first evaluation checkpoint where the run stayed above threshold for a window of evaluations. The window is a sequence of neighboring evaluation checkpoints, and each checkpoint is judged by its `return_mean` across evaluation episodes.

`never_reached_threshold`: whether the run failed to reach the threshold at all.

For a fair comparison, use the same threshold for every variant. For LunarLander, this project uses return >= 200. Compare the median first reach step across seeds. Here median means the middle first-reach value after sorting the successful seeds for the same variant. Also report how many seeds reached it. A variant with a fast median over only two successful seeds is not better than a variant that reaches the threshold in all seeds.

## 4. Stability: RL Can Learn and Then Forget

DQN is bootstrapped: it learns Q-values from targets that depend on other Q-values. This can create feedback loops. A policy can become good and then get worse as the value estimates shift.

The main stability fields are:

`catastrophic_collapse_count`: how many times `return_mean` at an evaluation checkpoint dropped by a large fraction after the run had already crossed the success threshold. The drop is measured along one run's evaluation checkpoints.

`largest_eval_drop`: the largest absolute drop in `return_mean` from a previous best checkpoint to a later checkpoint in the same run.

`eval_return_std_across_time`: the standard deviation of `return_mean` values across evaluation checkpoints in one run. It describes how much the evaluation curve moved over training.

`rolling_return_std_mean`: the mean of rolling standard deviations of recent training episode returns. The standard deviation is computed inside each rolling window, then those window-level values are averaged.

Always interpret collapse count with threshold reach rate. A run that never learns may have zero collapses only because it never reached a level from which it could collapse.

## 5. Seeds Matter More Than in Many Supervised Runs

RL training is sensitive to random seeds because the seed affects initialization, exploration actions, replay contents, and sometimes environment transitions. Do not trust one seed.

For each variant, compare:

Mean final return across seeds. This is the arithmetic mean of each seed's final evaluation return for the same variant.

Seed standard deviation. This is the standard deviation of final evaluation returns across seeds for the same variant.

95 percent confidence interval of the seed mean. This is the uncertainty range around the across-seed mean, not a range containing 95 percent of episodes.

How many seeds solved the task at the end.

How many seeds ever reached the threshold.

The per-seed table is also important. If one seed fails completely while four seeds are excellent, the mean may look acceptable but the method is unreliable.

When the same seed IDs are shared across variants, paired seed comparisons are useful. For seed 0, compare every variant against baseline seed 0; for seed 1, compare against baseline seed 1; and so on. This reduces noise from lucky or unlucky seeds.

## 6. Replay Buffer Diagnostics

Experience replay is one of the central DQN stabilizers. It breaks short-term correlation and allows each transition to be reused for multiple updates.

Important fields:

`replay_enabled`: whether replay was used.

`final_buffer_size`: how many transitions were in the buffer at the end.

`sample_age_mean_last_10pct`: the mean age of sampled replay transitions during the last 10 percent of updates. The age is measured in environment steps between when a transition was collected and when it was sampled for training. A large mean sample age means updates used a mix of older and newer experience.

`sample_consecutive_transition_fraction_last_10pct`: how often a replay batch contained transitions that came from right next to each other in the same original episode, measured during the last 10 percent of updates. A transition is one stored experience tuple, such as state, action, reward, next state, and done. Consecutive transitions are neighbors in time: for example, step 101 and step 102 from the same episode. If this fraction is high, the batch is using many near-duplicate moments from the same short time period, so the samples are still correlated. If it is low, the batch is mixed from different times or episodes, which is better decorrelation.

If replay is removed, the agent usually trains on recent correlated data. That can make learning more like chasing a moving target from a narrow stream of experience. In the report, the no-replay ablation should be read mostly as a test of decorrelation and sample reuse.

## 7. Target Network Diagnostics

DQN uses a target network because the Bellman target contains a neural network prediction. If the same network is used for both prediction and target, the target moves every update. This can make bootstrapping unstable.

Important fields:

`target_network_enabled`: whether a separate target network was used.

`target_update_count`: how many hard or soft target updates occurred.

`online_target_param_l2_mean`: the mean L2 parameter distance between the online and target networks across logged measurements. It describes the typical separation between the two networks.

`target_q_mean_last_10pct`: the mean target-side Q-value over update batches from the last 10 percent of training.

Removing the target network can sometimes still solve a simple environment, but the expected risk is more oscillation, larger collapses, or worse retention.

## 8. TD Error, TD Loss, and Gradient Metrics

TD error is the difference between the current Q estimate and the Bellman target. It is the DQN analogue of a regression residual, but the target itself is learned and non-stationary.

Important fields:

`td_loss_mean_last_10pct`: the mean TD loss over update batches from the last 10 percent of training.

`td_error_abs_mean_last_10pct`: the mean absolute TD error over update batches from the last 10 percent of training.

`td_error_abs_p95_last_10pct`: the 95th percentile of absolute TD errors over update batches from the last 10 percent of training. This describes the large-error tail and is often more useful than the mean because rare bad targets can destabilize training.

`grad_norm_mean_last_10pct` and `grad_norm_max`: gradient scale. The mean is over update batches from the last 10 percent of training, while the max is the largest logged gradient norm.

`gradient_clip_fraction`: fraction of updates where clipping was active.

High clipping is not automatically bad. It may mean the optimizer is being protected from large TD errors. But if clipping is high and returns are poor, it suggests the update targets are noisy or unstable.

## 9. Q-Value Diagnostics and Overestimation

DQN can overestimate action values because the max over noisy Q estimates tends to select overestimated actions. Double DQN tries to reduce this by using one network to select the action and the other network to evaluate it.

Important fields:

`online_q_mean_last_10pct`: the mean online Q-value over update batches from the last 10 percent of training.

`online_q_max_mean_last_10pct`: the mean of the maximum online action value over update batches from the last 10 percent of training.

`target_q_mean_last_10pct`: the mean target Q-value over update batches from the last 10 percent of training.

`q_overestimation_proxy_last10`: a proxy comparing online max values to target values.

`max_q_value_seen`: the largest Q-value observed during training.

Interpret Q metrics together with return. A higher Q scale is not better by itself. If Q-values rise while returns do not, the value function may be overconfident. If Double DQN improves return but the proxy does not decrease, that means this proxy is not capturing the whole effect, or the variant reached a different value scale.

## 10. Dueling Network Diagnostics

Dueling DQN separates Q(s, a) into a state-value stream and an action-advantage stream. This helps when many actions have similar value in a state, because the network can learn that the state is good or bad without having to relearn that fact independently for each action.

Useful logged fields in update metrics include:

`dueling.value_stream_mean` and `dueling.value_stream_std`: the mean and standard deviation of the dueling value-stream outputs over logged update batches.

`dueling.advantage_abs_mean`: the mean absolute advantage-stream output over logged update batches.

`dueling.action_gap_mean`: the mean gap between the best action value and the next alternatives over logged update batches.

The action gap is the difference between the best action value and the next alternatives. A larger useful gap can make the greedy policy more decisive, but a too-large gap with bad return can mean the network is confidently wrong.

## 11. LunarLander-Specific Behavior Metrics

Return tells you whether the agent did well. Environment-specific metrics help explain how.

Useful fields under `lunarlander_eval`:

`success_rate`: fraction of evaluation episodes in one evaluation batch with return >= 200.

`landing_success_rate`: fraction of evaluation episodes in one evaluation batch ending in a successful landing condition.

`crash_rate`: fraction of evaluation episodes in one evaluation batch ending in a crash.

`timeout_rate`: fraction of evaluation episodes in one evaluation batch that reached the time limit.

`both_legs_contact_rate`, `one_leg_contact_rate`, and `no_leg_contact_rate`: fractions of evaluation episodes in one evaluation batch with each final contact behavior.

`main_engine_action_fraction_mean` and `side_engine_action_fraction_mean`: mean fractions of actions that used the main engine or side engines across evaluation episodes. High main-engine use may indicate hovering or inefficient control. High side-engine use may indicate strong attitude correction.

`episode_length_mean`: the mean episode length across evaluation episodes in one evaluation batch. Long episodes can mean stable hovering, slow landing, or timeouts. Use it with success, crash, and timeout rates.

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
