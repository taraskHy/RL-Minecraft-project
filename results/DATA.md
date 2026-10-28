# What data we have, and where

Everything below is in Git under `results/` unless it says *local only*.
`python -m parkour.report` refreshes `results/` from the run folders in `runs/`
(which stay on the training PC, inside OneDrive, and also hold every
checkpoint).

## Training: every iteration of every run

`results/runs/NAME/` for each training run (and `results/runs/_v1_unstable/`
for the first three runs, discarded after PPO collapsed; kept as evidence;
`final-v2` is an abandoned run that no result uses):

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

"Live" folders are **real Minecraft 1.21.4**; "offline" ones are the bot's
JavaScript on Mineflayer's physics engine without a server. Every folder's
`summary.md` names its checkpoint. The generated snapshot at the end of
`HANDOFF.md` lists every folder with its score.

| Folder | What |
| --- | --- |
| `v2-live`, `v2-live-obstacles`, `v2-live-showcase` | real Minecraft, **final policy** (discount 0.995): 59/60 standard, 57/60 obstacles, 3/4 hand-designed (misses the neo course) |
| `v2-offline`, `v2-offline-obstacles`, `v2-offline-showcase` | the same courses offline: identical outcomes |
| `v2-planner-live`, `v2-planner-live-obstacles`, `v2-planner-offline`, `v2-planner-offline-obstacles` | final policy + route planner, no course list: 50/60, 55/60 (live and offline identical) |
| `v2-tuned-planner-offline`, `v2-tuned-planner-offline-obstacles` | the same courses offline with the retuned planner (`cost=distance, align=2.0`): 51/60, 56/60 |
| `live2-final`, `live2-obstacles`, `live2-showcase` | real Minecraft, first final policy, re-run with the collision-tolerance patch: 59/60, 58/60, 3/4 |
| `recheck-std`, `recheck-obs`, `recheck-show` | the same re-check offline: identical |
| `live-final`, `live-final-obstacles`, `live-showcase`, `live-smoke` | real Minecraft, first final policy, before the collision-tolerance patch: 59/60, 58/60, 3/3, 6/6 |
| `offline-*`, `final-*` | offline runs of the main run's policy (`offline-*`) and the first final policy (`final-*`) |
| `planner-live`, `planner-live-obstacles`, `planner-offline`, `planner-offline-obstacles` | first final policy + route planner, no course list: 52/60, 53/60 |
| `noroute-live`, `noroute-live-obstacles`, `noroute-offline`, `noroute-offline-obstacles` | learned no-route model `noroute-v2`: 55/60, 37/60 |
| `neo-live-final`, `neo-offline-final` | neo policy `neo-dagger3` on the neo course set (`configs/neo_live_courses.json`): 11/31 |
| `neo-full`, `cheat-check` | an early neo checkpoint (`runs/neo-ck.pt`, source run not recorded) on the neo course set (1/31) and on one test course (0/1) |
| `neo-preview__bot` | the first neo attempts during neo training: stalls at the wall and falls |
| `film-*` | single-course real-Minecraft runs recorded for the first-person videos (`scripts/film_pov.py`); `film-neo_level7-*` is the neo video |
| `live-rainbow-test`, `viewer-test` | test runs: the four hand-designed courses (3/4) and one course with the spectator viewer (1/1) |
| `progress-demo-games__*` | the same courses at every stage of a training run, from untrained to trained |

## Evaluations (`results/`)

| File | What |
| --- | --- |
| `final_policy.pt` / `.json` | the final policy (`runs/obstacles-g995`, discount 0.995) |
| `final_policy_v1.pt` / `final_policy_v1.json` | the first final policy (`runs/obstacles-ft`, discount 0.99) |
| `jump_eval.json`, `jump_eval_v1.json`, `jump_eval_main.json` | every jump type from a standing start at 16 angles: final, first final and main-run policy |
| `stress.json`, `stress_v1.json`, `stress_main.json` | course styles outside training, 100 courses per level and condition, same three policies |
| `offline_summary.md`, `live_summary.md` (and `offline_summary_v1.md`, `live_summary_v1.md`) | the bot's per-level tables offline and in real Minecraft, final and first final policy |
| `physics_parity.json` | the simulator against prismarine-physics 1.11.1: 99,000 ticks, largest error 1.4e-14 blocks |
| `hparams_defaults_validation.json` | the defaults' validation score for the hyperparameter table |
| `audit_claims.json` | the audit of 829 claims in the documents against the run files (28 Oct) |
| `results/planner/` | route planner on the 1,200 + 1,200 test courses: `sim_test_*.json` (`_v1`: with the first final policy; `_tuned`: the retuned setting `cost=distance, align=2.0`); `validation_tuning.jsonl`: every planner variant tried on the validation courses (one JSON line each) |
| `results/neo/` | the neo numbers (written by `scripts/neo_evidence.py` and the evaluations), below |

`results/neo/`:

| File | What |
| --- | --- |
| `course_eval.json`, `round3_eval.json` | each neo policy on the 210 held-out level 6-8 courses with a neo (and the validation ones) |
| `by_neo_count.json` | the chosen neo policy split by the number of neos per course |
| `jump_eval_*.json` | every advanced jump type alone (`parkour.jumpeval --advanced`), one file per neo policy |
| `demonstrations.json`, `demos_collection.txt` | the search demonstrations: 50,706 collected, 12,800 used |
| `neo_dagger_r1b.txt`, `neo_dagger_r2.txt`, `neo_dagger_r3.txt` | the DAgger collection logs per round |
| `label_agreement.json` | how often DAgger labels agree with the bot, before and after simplification |
| `detector_check.json` | the first neo-takeoff detector against the fixed one |
| `wide_takeoff_search.json` | which neos are landable from 2x2 and 3x3 takeoffs |

## Videos (`results/videos/`, 1080p MP4)

| File | What |
| --- | --- |
| `ghost_race_*.mp4` | ten training stages of one agent on the same unseen course, seen from above |
| `real_minecraft_*.mp4` | the bot's first-person view in real Minecraft (`_no_course_list`: with the route planner; `_neo_level7`: the neo policy) |
| `replay_*.mp4` | replays from the theatre (no-course-list runs draw the plan; `replay_137_*` is a neo) |

## Pages to show

| Page | Contents |
| --- | --- |
| `results/replay_viewer.html` | 3D replays of all 123 real-Minecraft episodes |
| `results/progress.html` | learning progress: the same courses from untrained to trained, failures included |
| `results/neo_preview.html` | neo training in progress |
| `results/theatre.html` | Minecraft-style 3D replays and the ghost races |
| `results/showcase.html` | the results page for the presentation ("Learning Minecraft Parkour") |
| `results/overnight_plan.html` | the overnight decision tree of 28 October (`docs/overnight_plan.md`) |

Open them in a browser; they need no server.

## Tables

`results/README.md` (generated) has every number with its source run;
`docs/report_draft.md` has the report text. `results/report_draft.docx` and
`results/report_draft.pdf` are that text in the submission format (Arial 12,
1.5 spacing, one-inch margins), written by `scripts/report_docx.py --pdf`.
