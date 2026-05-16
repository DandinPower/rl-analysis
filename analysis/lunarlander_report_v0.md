# LunarLander Report v0

## Data and Method

This report analyzes the completed `output/lunarlander_v3/` runs. There are 30 runs: six DQN variants with 5 seeds each. Every run used a budget of 500,000 environment steps, and the success threshold is return >= 200. The analysis uses the latest completed run under each variant/seed directory.

The statistics below are computed with numpy and pandas from `summary.json`, `eval_metrics.jsonl`, and the update logs. Intervals are 95 percent t-intervals across seeds. With only five seeds, the intervals should be read as uncertainty estimates, not as formal proof.

Figures:

![Evaluation learning curves](figures/lunarlander/fig_eval_learning_curves.png)

![Final return distribution](figures/lunarlander/fig_final_return_distribution.png)

## Overall Performance Summary

| Variant | Seeds | Final return mean +/- CI | Seed std | Best return mean +/- CI | Norm. AUC mean +/- CI | Reached 200 | Median first reach | Collapses/run | Best-final gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DQN baseline | 5 | 180.2 +/- 131.7 | 106.1 | 258.8 +/- 28.6 | 88.1 +/- 47.3 | 5/5 | 160,000 | 12.6 | 78.6 |
| DQN without target network | 5 | 167.9 +/- 138.1 | 111.2 | 255.3 +/- 14.1 | 55.7 +/- 41.3 | 5/5 | 220,000 | 17.8 | 87.4 |
| DQN without replay buffer | 5 | -113.1 +/- 145.6 | 117.3 | 108.7 +/- 40.4 | -179.3 +/- 130.5 | 0/5 | n/a | 0.0 | 221.9 |
| Double DQN | 5 | 227.9 +/- 58.4 | 47.1 | 267.3 +/- 9.9 | 60.2 +/- 44.7 | 5/5 | 230,000 | 11.8 | 39.4 |
| Dueling DQN | 5 | 229.7 +/- 34.8 | 28.0 | 271.8 +/- 6.7 | 62.5 +/- 52.1 | 5/5 | 250,000 | 7.6 | 42.2 |
| Double + Dueling DQN | 5 | 210.0 +/- 146.9 | 118.3 | 247.2 +/- 63.6 | 59.2 +/- 72.6 | 4/5 | 215,000 | 8.4 | 37.2 |

The strongest final mean return is from Dueling DQN at 229.7 +/- 34.8. The strongest whole-training average is from DQN baseline at 88.1 +/- 47.3 normalized AUC. This difference is important: final score rewards the last checkpoint, while normalized AUC rewards being good for more of training.

Per-seed final returns:

| Seed | DQN | No target | No replay | Double | Dueling | Double+ Dueling |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 250.4 | 199.2 | -20.3 | 264.2 | 251.5 | 264.9 |
| 1 | 231.4 | 233.1 | -107.7 | 260.7 | 257.1 | 255.1 |
| 2 | 7.8 | 163.9 | 7.1 | 232.7 | 186.6 | 263.1 |
| 3 | 262.4 | 263.0 | -158.9 | 147.8 | 222.2 | -1.5 |
| 4 | 148.9 | -19.7 | -285.8 | 234.3 | 231.0 | 268.5 |

Paired comparison against baseline:

| Variant vs DQN | Final return delta | Seed win rate | Norm. AUC delta | First-reach step delta | Collapse delta |
| --- | --- | --- | --- | --- | --- |
| DQN without target network | -12.3 +/- 145.1 | 60% | -32.4 +/- 52.0 | 18000 +/- 118233 | 5.2 +/- 7.3 |
| DQN without replay buffer | -293.3 +/- 219.1 | 0% | -267.4 +/- 127.0 | n/a | -12.6 +/- 9.0 |
| Double DQN | 47.7 +/- 152.8 | 80% | -27.9 +/- 51.2 | 70000 +/- 96163 | -0.8 +/- 11.3 |
| Dueling DQN | 49.5 +/- 105.2 | 80% | -25.6 +/- 44.5 | 90000 +/- 58888 | -5.0 +/- 5.9 |
| Double + Dueling DQN | 29.8 +/- 236.7 | 80% | -28.9 +/- 102.0 | 25000 +/- 117274 | -4.2 +/- 8.7 |

![Paired final return deltas](figures/lunarlander/fig_paired_final_delta_vs_dqn.png)

## Baseline DQN Performance Study

The baseline DQN solved the task at least once in all 5 seeds. Its final mean return was 180.2 +/- 131.7, with a large seed standard deviation of 106.1. Its best mean return was 258.8 +/- 28.6, and its median first step reaching the success threshold was 160,000.

The baseline's main strength is early sample efficiency. Its normalized AUC was 88.1 +/- 47.3, the highest among these variants. The learning-curve and phase-return figures show that baseline DQN became useful earlier than most extensions.

The weakness is retention. The average best-final gap was 78.6, and the average collapse count was 12.6 per run. Two baseline seeds finished below the success threshold: 2, 4. This means the baseline often found a good policy, but the final checkpoint was not always the best policy.

![Sample efficiency and AUC](figures/lunarlander/fig_sample_efficiency_auc.png)

![Training phase returns](figures/lunarlander/fig_phase_returns.png)

## Ablation: Removing the Target Network

Removes the target network, so the bootstrap target moves with the online network. The final mean return was 167.9 +/- 138.1, lower than baseline. It still reached the 200 threshold in 5/5 seeds, but its median first reach step was 220,000, slower than baseline's 160,000.

Against baseline on matched seeds, it changed final return by -12.3 +/- 145.1, won 60% of seeds, had -32.4 +/- 52.0 normalized-AUC delta, and had a first-reach delta of +18000 steps. The stability diagnostics are the clearest warning sign: the no-target variant averaged 17.8 collapses per run, compared with 12.6 for baseline. This matches the expected role of a target network. Without a slower-moving bootstrap target, the value update target is less stable.

## Ablation: Removing Replay

Removes replay, so updates use recent correlated transitions with little sample reuse. This was the clearest failure. Final mean return was -113.1 +/- 145.6, and zero out of 5 seeds reached the success threshold. Its normalized AUC was -179.3 +/- 130.5, far below every replay-based variant.

Against baseline on matched seeds, it changed final return by -293.3 +/- 219.1, won 0% of seeds, had -267.4 +/- 127.0 normalized-AUC delta, and had no paired threshold comparison because the variant did not reach the threshold. The collapse count is 0.0, but that is not a sign of stability. The variant never crossed the threshold, so it had no successful regime from which to collapse. The result shows that replay is not just an implementation detail here; it is the core mechanism that gives DQN enough decorrelated and reused experience to learn LunarLander.

## Extension: Double DQN

Uses the online network to select the next action and the target network to evaluate it. Double DQN produced a final mean return of 227.9 +/- 58.4, above baseline. It reached the threshold in all seeds and had a final solved count of 4/5. Its median first reach step was 230,000, so it was slower to cross the threshold than baseline, but it ended with a better final policy.

Against baseline on matched seeds, it changed final return by 47.7 +/- 152.8, won 80% of seeds, had -27.9 +/- 51.2 normalized-AUC delta, and had a first-reach delta of +70000 steps. The Q-overestimation proxy was 0.213 for Double DQN and 0.140 for baseline. This proxy did not decrease, so the performance gain should not be described as proven lower overestimation from this single metric. A safer interpretation is that Double DQN improved final policy quality and reliability even though the logged proxy is sensitive to Q-value scale and late-training state distribution.

## Extension: Dueling DQN

Splits the network head into state-value and action-advantage streams. Dueling DQN had the best final mean return: 229.7 +/- 34.8. It solved at the end in 4/5 seeds and reached the threshold in all seeds. Its average collapse count was 7.6, lower than the baseline value of 12.6.

Against baseline on matched seeds, it changed final return by 49.5 +/- 105.2, won 80% of seeds, had -25.6 +/- 44.5 normalized-AUC delta, and had a first-reach delta of +90000 steps. The tradeoff is speed. Its median first reach step was 250,000, slower than baseline. The result suggests that separating state value from action advantage helped late policy quality and stability more than early learning speed.

## Extension: Double + Dueling DQN

Combines Double DQN target selection with the dueling value/advantage architecture. The combined variant had final mean return 210.0 +/- 146.9, but with very high seed standard deviation of 118.3. Four seeds finished near or above strong solved performance, while one seed failed badly.

Against baseline on matched seeds, it changed final return by 29.8 +/- 236.7, won 80% of seeds, had -28.9 +/- 102.0 normalized-AUC delta, and had a first-reach delta of +25000 steps. This means the combination can work very well, but in this result set it is less reliable than using the dueling head alone. The mean hides a bimodal-looking behavior: most seeds are strong, one seed is a clear failure.

![Stability and regression diagnostics](figures/lunarlander/fig_stability_regression.png)

## Optimization, Q-Value, and Behavior Diagnostics

The optimization figure shows that the no-replay variant had the largest late TD loss and a poor final policy. This supports the interpretation that removing replay makes the Bellman regression problem harder and less useful. Dueling DQN had the lowest Q-overestimation proxy among the strong final performers, while Double DQN had high final return even though this proxy was not lower than baseline.

![Optimization and Q diagnostics](figures/lunarlander/fig_optimization_q_diagnostics.png)

Final evaluation behavior summary:

| Variant | Success | Landing success | Crash | Timeout | Main engine | Side engines |
| --- | --- | --- | --- | --- | --- | --- |
| DQN baseline | 62% | 62% | 0% | 27% | 49% | 23% |
| DQN without target network | 52% | 52% | 0% | 42% | 40% | 24% |
| DQN without replay buffer | 3% | 3% | 60% | 25% | 51% | 34% |
| Double DQN | 78% | 78% | 0% | 7% | 40% | 35% |
| Dueling DQN | 84% | 84% | 1% | 5% | 43% | 20% |
| Double + Dueling DQN | 79% | 79% | 0% | 20% | 47% | 21% |

The behavior figure connects return to LunarLander outcomes. Good variants show higher final success and lower crash/timeout rates. Engine-use fractions help distinguish controlled landings from hovering or unstable correction.

![LunarLander behavior diagnostics](figures/lunarlander/fig_lunarlander_behavior.png)

## Main Conclusions

Experience replay is essential in this experiment. Removing replay prevented every seed from reaching the success threshold.

The target network matters for stability. Removing it did not make learning impossible, but it lowered final performance and increased collapse frequency.

Baseline DQN learned early, which gave it the best normalized AUC, but it often failed to retain its best policy until the final checkpoint.

Dueling DQN was the strongest final performer and had fewer collapses than baseline, suggesting better late-training value representation for LunarLander.

Double DQN improved final return, but the logged overestimation proxy alone does not prove a clean reduction in overestimation for these runs.

Double + Dueling DQN showed high upside but worse reliability than Dueling DQN alone because one seed failed badly.
