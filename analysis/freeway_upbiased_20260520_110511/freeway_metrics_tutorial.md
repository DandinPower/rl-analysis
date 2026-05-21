# Freeway Metrics Tutorial

This guide explains the metrics used in the Freeway up-biased DQN analysis. In this run, score and return are the same logged value. A score of 15 is the configured success threshold. The reference scores in the run configuration are random 0.0, human 29.6, and DQN 30.3.

## Evaluation Score

`return_mean` in `eval_metrics.jsonl` is the mean score across the evaluation episodes for one checkpoint. `final_eval_return_mean` in `summary.json` is the last evaluation score. `best_eval_return_mean` is the best checkpoint score reached during training.

`normalized_area_under_eval_curve` summarizes the whole learning curve. Higher values mean the agent scored well earlier, stayed high longer, or both. It is in score units because the raw area is divided by the 1M-step training budget.

## Thresholds and Sample Efficiency

The report tracks first reach steps for score thresholds 5.0, 10.0, 15.0, and 22.5. The main success threshold is 15.0. The first reach step is the first evaluation checkpoint where the mean score reached the threshold.

The sustained threshold step uses the same configured rule as training: the score must stay above the threshold for a window of neighboring evaluation checkpoints.

## Stability

`best_final_gap` is best score minus final score. A large value means the run learned a better policy and then lost part of it by the final checkpoint.

`catastrophic_collapse_count` counts large drops after crossing the success threshold. Read it with the learning curve because a run can have few collapses simply if it spends little time in a high-score regime.

## Optimization and Q Diagnostics

TD loss and TD error describe how hard the Bellman regression problem was near the end of training. They are not policy-quality metrics by themselves. Use them after checking evaluation score.

The Q overestimation proxy compares the online max Q value with the target-side Q value. It is useful for relative diagnostics, but it is affected by Q-value scale and the replay states sampled late in training.

## Freeway Behavior Metrics

`zero_score_rate` is the fraction of evaluation episodes that scored zero. In this result set it should be near zero for trained agents.

`action_up_fraction_mean`, `action_down_fraction_mean`, and `action_noop_fraction_mean` summarize the final policy's action mix during evaluation. Because this experiment used an up-biased exploration setting during training, these action fractions help check whether the final greedy policy still relies heavily on upward movement or also learns when to wait.
