# Experiment protocol

All shipped Python training experiments run in `crafting_arena_v1`, a fixed-map simulator. Keep these separate from actual Minecraft development arenas and full Survival evaluations.

## Protocol implemented by the runner

1. Enumerate the declared algorithm, reward, learning-rate, discount, and exploration-decay grid.
2. Train every candidate using every listed independent training seed. Each candidate gets the same number of episodes and maximum decisions per episode. This is an **episode-budget comparison**, not an equal-transition or equal-wall-clock comparison: early successes/deaths change the number of interactions. The CSV records total transitions so sample efficiency can also be examined.
3. Evaluate frozen policies on the configured validation seed range. No updates, exploration, or Q-table mutation occur; exact-value ties are broken using a separate seeded RNG.
4. Select one hyperparameter configuration per algorithm/reward pair by mean validation success across training seeds, then mean base reward, then fixed candidate order. Do not select only the most successful training seed.
5. Save this selection before evaluating the configured test episode seeds. Retain every selected training seed and report its individual performance and between-seed sample standard deviation.
6. Run the random-action baseline on the same test episode-seed set. Its policy is not trained. All methods share the same action set and map.

The environment seeds used during training are sampled from `[0, 1,000,000)`, distinct from validation and test. Diagnostic train/demo/evaluate commands use seeds beginning at 1,000,000. The smoke configuration uses validation at 1,000,000 and exploratory test at 2,000,000; the larger course configuration reserves validation at 3,000,000 and final test at 4,000,000. Configured evaluation ranges are checked for overlap and against the training range. Separate RNG seeds are controls for reproducibility, not guarantees of statistical independence. The same seed across policies provides the same initial start; different action trajectories consume movement randomness differently.

`train` produces a validation preview for development. `evaluate` permits an explicit seed range and should not be repeatedly used on the final test set while tuning. The experiment runner enforces the sequence inside one run but cannot prevent a person from inspecting tests and modifying later experiments. Declare changes and use a genuinely new reserved test set for a later confirmatory study.

## Outputs and interpretation

- `config.json` and `provenance.json`: requested grid, code revision/status, source SHA256 fingerprints, Python and platform.
- Per candidate/seed: Q table, episode CSV, learning curve, validation CSV, elapsed training time and transition count.
- `validation_summary.json`: all candidates, including weaker results.
- `selected_configs.json`: the frozen selection before test evaluation.
- Selected runs' `test.csv`: each actual test episode's seed, return, steps, success, death, timeout, and unseen-state steps.
- `summary.json` and `comparison.csv`: held-out simulator results and seed dispersion.

Use success rate, death rate, base return, and steps for comparisons. Do not compare shaped totals against sparse totals as though they were the same objective. Successful-episode length excludes failures and must be read alongside success count/rate. Wilson 95% intervals refer to episodes for one checkpoint, conditional on that checkpoint; they do not measure training uncertainty. Use dispersion across independent training seeds for that separate question. Three smoke-run seeds give only preliminary evidence; the larger course configuration uses five.

The SVG shows exploratory training behavior in bins. It is not frozen-policy evaluation and is not a real-game success plot. The map is the same during training and evaluation. Held-out episode seeds do not mean held-out Minecraft worlds.

## Larger course study

`configs/course_experiment.json` specifies 32 configurations with five training seeds each: two algorithms, two rewards, two learning rates, two discounts, and two decay fractions. It is supplied as a reproducible plan, not claimed to have run. Add a measured equal-transition-budget comparison and new-map tasks when the implementation matures. Record runtimes and CPU/GPU use; do not claim a GPU speedup for the standard-library tabular learner.

For each real-world stage, keep the same experimental questions but replace simulator success with observed in-game achievements. Compare the hybrid controller against a scripted baseline and a version whose learned decisions are replaced by random decisions. Log interventions, resets, failed macros, and all failed attempts. A single completed final demo is useful evidence of possibility, not a general success-rate estimate.
