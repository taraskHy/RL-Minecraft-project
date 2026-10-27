# What data we have, and where

Everything below is in Git under `results/` unless it says *local only*.
`python -m parkour.report` refreshes `results/` from the run folders in `runs/`
(which stay on the training PC, inside OneDrive, and also hold every
checkpoint).

## Training: every iteration of every run

`results/runs/NAME/` for each training run (and `results/runs/_v1_unstable/`
for the first three runs, discarded after PPO collapsed; kept as evidence):

| File | One row per | Columns (the ones you will want) |
| --- | --- | --- |
| `train_log.csv` | PPO iteration (524,288 game ticks) | `success`, `fell`, `truncated` (fractions of finished episodes), `ep_return` (mean reward per episode), `ep_len`, `progress`, `reward_per_tick`, `frontier` (curriculum level), `pi_loss` (policy loss), `v_loss` (value loss), `entropy`, `kl`, `clipfrac`, `lr`, `stopped`, `train_success_L0..L8` |
| `eval_log.csv` | 50 iterations | held-out test success overall and per level (`success_L*`), course progress (`progress_L*`) and falls (`fell_L*`) |
| `eval_val_log.csv` | 50 iterations | validation success (used to pick `best.pt`) |
| `eval_obstacles_log.csv` | 50 iterations | held-out obstacle-course success (obstacle runs) |
| `config.json` | run | every setting, including seed, reward weights and curriculum |
| `summary.json` | run | last evaluation and the validation-selected best iteration |

Checkpoints (`runs/NAME/checkpoints/best.pt`, `latest.pt`, and
`updates_*.pt` snapshots for the progress run) are *local only* (about 14 MB
per run). The final policy is in Git: `results/final_policy.pt` (weights) and
`results/final_policy.json` (what the bot loads).

## Figures (`results/figures/`)

| Figure | Shows |
| --- | --- |
| `training_main-s0.png` | the main run's 10 training curves: success, reward, falls, episode length, curriculum level, entropy, policy loss, value loss, KL, learning rate |
| `training_obstacles-ft.png` | the same for the final (obstacle) policy |
| `training__v1_unstable_main-s0.png` | the collapse before the PPO fixes |
| `main_learning_curve.png` | held-out success per difficulty level over training |
| `ablation_curves.png` | reward and curriculum ablation, 3 seeds per arm |
| `jumps.png` | every jump type: the policy against scripted strategies |

## Games the bot played, tick by tick

`results/games/NAME/`: `courses.json` (the courses), `episodes.jsonl.gz`
(one line per episode: result, platforms reached, start state, every action
and every tick's position and velocity), `summary.md` / `summary.json`
(outcomes, comparison with the simulator) and, for real-game runs,
`server.log.gz`.

| Folder | What |
| --- | --- |
| `live-final`, `live-final-obstacles`, `live-showcase`, `live-smoke` | **real Minecraft 1.21.4**, final policy: 59/60, 58/60, 3/3, 6/6 |
| `offline-*`, `final-*` | the bot's JavaScript on Mineflayer's physics engine, no server |
| `neo-preview__bot` | the first neo attempts during neo training: stalls at the wall and falls |
| `progress-demo-games__*` | the same courses at every stage of a training run, from untrained to trained |

## Pages to show

| Page | Contents |
| --- | --- |
| `results/replay_viewer.html` | 3D replays of all 123 real-Minecraft episodes |
| `results/progress.html` | learning progress: the same courses from untrained to trained, failures included |
| `results/neo_preview.html` | neo training in progress |

Open them in a browser; they need no server.

## Tables

`results/README.md` (generated) has every number with its source run;
`docs/report_draft.md` has the report text.
