# What data we have, and where

Everything below is in Git under `results/` (the course files under
`configs/`) unless it says *local only*. `python -m parkour.report` refreshes `results/` from the run
folders in `runs/` (which stay on the training PC, inside OneDrive, and also
hold every checkpoint).

## Training: every iteration of every run

`results/runs/NAME/` for each training run (and `results/runs/_v1_unstable/`
for the first three runs, discarded after PPO collapsed; kept as evidence;
`final-v2` is an abandoned run that no result uses):

| File | One row per | Columns (the ones you will want) |
| --- | --- | --- |
| `train_log.csv` | PPO iteration (524,288 game ticks; 131,072 in `hp-envs4096`) | `success`, `fell`, `truncated` (fractions of finished episodes), `ep_return` (mean reward per episode), `ep_len`, `progress`, `reward_per_tick`, `frontier` (curriculum level), `pi_loss` (policy loss), `v_loss` (value loss), `entropy`, `kl`, `clipfrac`, `lr`, `stopped` (the three `_v1_unstable` logs have no `lr` or `stopped`), `train_success_L0..L5` (`L0..L8` in the level 0-8 runs: `advanced-ft*` and the neo runs); `neo-bc` and the DAgger runs also have `bc_loss` (imitation loss; `main-g995` and `obstacles-g995` have the column too, always 0) |
| `eval_log.csv` | 50 iterations (200 in `hp-envs4096`, 25 in `progress-demo`) | held-out test success overall and per level (`success_L*`), course progress (`progress_L*`) and falls (`fell_L*`); levels 0-8 in the neo runs |
| `eval_val_log.csv` | 50 iterations (200 in `hp-envs4096`, 25 in `progress-demo`) | validation success (used to pick `best.pt`) |
| `eval_obstacles_log.csv` | 50 iterations | held-out obstacle-course success (obstacle runs) |
| `config.json` | run | every setting, including seed, reward weights and curriculum |
| `summary.json` | run | last evaluation and the validation-selected best iteration |

The runs:

| Run | What it is |
| --- | --- |
| `main-s0` | the main run: 3,000 iterations, discount 0.99 |
| `full-s1`, `full-s2`, `sparse-*`, `nolanding-*`, `nofall-*`, `uniform-*`, `noroute-s0..s2` | the reward and curriculum ablation, 1,000 iterations per seed (the `full` arm's seed 0 is the first 1,000 iterations of `main-s0`) |
| `obstacles-ft` | the first final policy: `main-s0` fine-tuned with obstacles for 1,000 iterations |
| `hp-*` | the hyperparameter study: one setting changed around the defaults, seed 0, 1,000 iterations (`hp-envs4096`: 4,096 environments for 4,000 iterations, the same 0.52 B ticks); `hp-gamma0.995-s1` and `hp-gamma0.995-s2` repeat discount 0.995 with seeds 1 and 2 |
| `main-g995` | the main run's recipe with discount 0.995, 3,000 iterations |
| `obstacles-g995` | `main-g995` fine-tuned with obstacles for 1,000 iterations; its iteration 699 is the final policy |
| `noroute-v2` | the learned no-route model: 3,000 iterations from scratch, obstacles on 30% of jumps, without the platform slots or the progress input |
| `final-v2` | a retrain from scratch with obstacles on 30% of jumps and learning rate 0.0003, stopped after 534 of 3,000 iterations |
| `advanced-ft`, `advanced-ft2`, `advanced-ft3` | the first final policy fine-tuned on levels 0-8 (discount 0.99, obstacles on 30% of jumps): three chained runs of 820, 335 and 612 iterations, each stopped before the 1,500 configured; the neo runs start from `advanced-ft3` |
| `neo-stall` | `advanced-ft3` (levels 0-8, discount 0.99) trained 1,000 more iterations with a stall penalty of 5 |
| `neo-control-aborted` | a control run from `advanced-ft3` without the stall penalty, stopped after 5 iterations |
| `neo-bc` | `neo-stall` trained 800 more iterations with PPO plus imitation of the search demonstrations (imitation weight 1, constant) |
| `neo-dagger-collapsed` | the first DAgger training; it collapsed because the search's labels were noisy (kept as evidence) |
| `neo-dagger`, `neo-dagger2`, `neo-dagger3` | DAgger rounds 1 to 3, 500 iterations each, imitation weight 0.3 (constant); each round trains on the imitation pairs of all rounds so far; `neo-dagger3` is the neo policy |
| `progress-demo` | the progress run: 400 iterations from scratch (seed 3); its snapshots play the `progress-demo-games__*` games, the ghost-race videos and `progress.html` |
| `film-pov` | not a training run: the real-Minecraft episodes (tour, stairs, sprint) recorded with the first final policy for the first-person videos |

Checkpoints (`runs/NAME/checkpoints/best.pt`, `latest.pt`, and
`updates_*.pt` snapshots for the progress run) are *local only* (about 14 MB
per run). The neo policy is `runs/neo-dagger3/checkpoints/best.pt` (*local
only*). The final policy is in Git; see the next section.

## Policies and evaluation files (`results/`)

| File | What |
| --- | --- |
| `final_policy.pt`, `final_policy.json` | the final policy since 28 October 10:41: `obstacles-g995` iteration 699, discount 0.995 (weights, and the file the bot loads) |
| `final_policy_v1.pt`, `final_policy_v1.json` | the first final policy: `obstacles-ft` iteration 999, discount 0.99 |
| `jump_eval.json`, `jump_eval_v1.json` | every jump type of levels 0-5 and the 49 impossible types, 16 start angles each, in the simulator: final policy and first final policy (`jump_eval_main.json`: `main-s0`) |
| `stress.json`, `stress_v1.json` | the standard test course style and six styles that differ from training (long, zigzag, straight, any start yaw, single blocks, obstacles on 50% of jumps), 100 courses per level and condition, in the simulator: final policy and first final policy (`stress_main.json`: `main-s0`) |
| `offline_summary.md`, `live_summary.md` | the offline bot and the real-Minecraft bot with the final policy; `results/README.md` copies them |
| `offline_summary_v1.md`, `live_summary_v1.md` | the same checks with the first final policy, except the offline showcase check, which has only the 3 courses without the neo course (3/3) |
| `physics_parity.json` | the simulator against prismarine-physics 1.11.1 (run through Node) on random block fields and random inputs: 99,000 ticks on CPU and GPU, largest position error 1.42e-14 blocks |
| `hparams_defaults_validation.json` | validation success of the defaults at iteration 999 (`full-s1`, `full-s2`), the reference row of the hyperparameter table |

Game summaries name the checkpoint path used at the time. Before 28 October
10:41, `results/final_policy.pt` held the first final policy.

## Route planner (`results/planner/`)

| File | What |
| --- | --- |
| `sim_test_standard.json`, `sim_test_obstacles.json` | the route planner with the final policy on the 1,200-course standard and obstacle test sets, in the simulator: success with the course list given and with the planner, both fall rules, per level, failure modes, every plan and course |
| `sim_test_standard_v1.json`, `sim_test_obstacles_v1.json` | the same with the first final policy |

## Neo jumps (`results/neo/`)

| File | What |
| --- | --- |
| `course_eval.json` | 210 held-out test courses of levels 6-8 with a neo (337 neo jumps), in the simulator: courses completed, neo jumps reached and passed, from `neo-stall` to DAgger round 2 |
| `round3_eval.json` | the same counts on the validation and test neo courses for round 2 (`best.pt`) and round 3 (`best.pt`, `latest.pt`); used to choose the neo policy |
| `by_neo_count.json` | the neo policy on the same 210 test courses, by number of neos in the course, in the simulator: one neo 56/129 completed, two 3/47, three 2/25, four or more 0/9 |
| `jump_eval_NAME.json` | every jump type of the advanced feasibility study, 16 start angles each, in the simulator, for `advanced-ft3` (before neo training), `neo-stall`, `neo-bc` and the three DAgger rounds: the 78 types the study puts in levels 6-8 (13 single neos among them) and the 132 its scripted strategies never made. Three of those 132 are single neos the search later landed, and level 6 of the course generator includes them |
| `demos_collection.txt` | the demonstration search log: 50,706 verified landing sequences on the 16 single-neo layouts, both wall sides |
| `demonstrations.json` | the demonstration counts: 50,706 over 32 layout-sides (1,350 to 1,600 each), 12,800 used for training (400 per layout-side) |
| `wide_takeoff_search.json` | the search on one neo after a 1x1, 2x2 or 3x3 takeoff (3 layouts, both sides): 200 of 200 wanted landings found when the landing is 1 lane from the wall's open end, 0 when it is 2 lanes away |
| `neo_dagger_r1b.txt`, `neo_dagger_r2.txt`, `neo_dagger_r3.txt` | label collection logs for DAgger rounds 1 to 3: neo takeoffs reached, how many the search continued to a landing, sequences and imitation pairs |
| `detector_check.json` | the neo-takeoff detector on the training courses of DAgger round 1 (`runs/neo_dagger_r1b.pt`): of the 2,526 takeoffs the first detector found, 1,090 (43%) were real neos |
| `label_agreement.json` | per round, the share of imitation pairs whose action part (move, strafe, jump, sprint, turn) equals the collecting policy's own choice, and the imitation loss: 82 to 99% per part in DAgger rounds 1 to 3 (89 to 99% in round 1), 19 to 82% for the unsimplified labels of the collapsed run |

The demonstrations and labels themselves are *local only*:
`runs/neo_demos.json` (all 50,706), `runs/neo_demos_train.json` (the 12,800
used), `runs/neo_dagger_r1.pt` (the noisy labels of the collapsed run) and
`runs/neo_dagger_r1b.pt`, `_r2.pt`, `_r3.pt` (the imitation pairs).

## Figures (`results/figures/`)

| Figure | Shows |
| --- | --- |
| `training_main-s0.png` | the main run's 10 training curves: success, reward, falls, episode length, curriculum level, entropy, policy loss, value loss, KL, learning rate |
| `training_obstacles-ft.png` | the same for the first final policy (`obstacles-ft`) |
| `training__v1_unstable_main-s0.png` | the collapse before the PPO fixes |
| `main_learning_curve.png` | held-out success per difficulty level over training |
| `ablation_curves.png` | reward and curriculum ablation, 3 seeds per arm |
| `jumps.png` | every jump type: the policy against scripted strategies |

## Games the bot played, tick by tick

`results/games/NAME/`: `courses.json` (the courses), `episodes.jsonl.gz`
(one line per episode: result, platforms reached, start state, every action
and every tick's position and velocity, and the plans in planner runs),
`summary.md` / `summary.json` (outcomes, comparison with the simulator) and,
for real-game runs, `server.log.gz`.

"Live" folders are **real Minecraft 1.21.4**; "offline" ones are the bot's
JavaScript on Mineflayer's physics engine without a server. Every folder's
`summary.md` names its checkpoint. The generated snapshot at the end of
`HANDOFF.md` lists every folder with its score.

| Folder | What |
| --- | --- |
| `v2-tuned-planner-offline`, `v2-tuned-planner-offline-obstacles` | the same courses offline with the retuned planner (`cost=distance, align=2.0`): 51/60, 56/60 |
| `live2-final`, `live2-obstacles`, `live2-showcase` | real Minecraft, first final policy, re-run with the collision-tolerance patch: 59/60, 58/60, 3/4 |
| `recheck-std`, `recheck-obs`, `recheck-show` | the same re-check offline: identical |
| `neo-full`, `cheat-check` | an early neo checkpoint (`runs/neo-ck.pt`, source run not recorded) on the neo course set (1/31) and on one test course (0/1) |
| `live-rainbow-test`, `viewer-test` | test runs: the four hand-designed courses (3/4) and one course with the spectator viewer (1/1) |
| `live-final`, `live-final-obstacles`, `live-showcase`, `live-smoke` | **real Minecraft 1.21.4**, first final policy: 59/60, 58/60, 3/3, 6/6 |
| `v2-live`, `v2-live-obstacles`, `v2-live-showcase` | **real Minecraft 1.21.4**, final policy, the same 60 + 60 held-out courses and the four hand-designed courses: 59/60, 57/60, 3/4 |
| `v2-offline`, `v2-offline-obstacles`, `v2-offline-showcase` | the same courses on the offline bot: 59/60, 57/60, 3/4 |
| `v2-planner-live`, `v2-planner-live-obstacles`, `v2-planner-offline`, `v2-planner-offline-obstacles` | the route planner instead of the course list, final policy, real Minecraft and offline: 50/60 standard and 55/60 obstacles in both |
| `planner-live`, `planner-live-obstacles`, `planner-offline`, `planner-offline-obstacles` | the same with the first final policy: 52/60 and 53/60 in both |
| `noroute-live`, `noroute-live-obstacles`, `noroute-offline`, `noroute-offline-obstacles` | the learned no-route model `noroute-v2`, real Minecraft and offline: 55/60 standard and 37/60 obstacles in both |
| `neo-live-final`, `neo-offline-final` | the neo policy on the neo course set (`configs/neo_live_courses.json`), real Minecraft and offline: 11/31 in both; on the showcase neo course it passes the first neo (from a 2x2 block) and falls at the second, after 3 of 11 jumps |
| `film-tour-*`, `film-stairs-*`, `film-sprint-*` | **real Minecraft**, one hand-designed course per folder, recorded for the first-person videos with the first final policy; the folder name ends in the date and time (MMDD-HHMMSS); the runs from 01:40 to 01:43 have the course list (two takes of the tour), the runs from 02:48 to 02:50 use the route planner |
| `film-neo_level7-*` | **real Minecraft**, the neo policy on `configs/neo_film_course.json`, recorded for `real_minecraft_neo_level7.mp4`: completed in 164 ticks |
| `offline-*`, `final-*` | the bot's JavaScript on Mineflayer's physics engine, no server |
| `neo-preview__bot` | the first neo attempts during neo training: stalls at the wall and falls |
| `progress-demo-games__*` | the same courses at every stage of the `progress-demo` run, from untrained to trained |

In every run from `v2-*` to `film-*` above, the outcome matched the simulator
on every course, and the real-Minecraft runs had 0 server corrections.

## Course files (`configs/`)

| File | What |
| --- | --- |
| `neo_live_courses.json` | the neo course set: the first 10 held-out test courses with a neo of levels 6, 7 and 8, and the showcase neo course (31 courses) |
| `neo_film_course.json` | the held-out level-7 course with one neo in `real_minecraft_neo_level7.mp4` (it is also in the neo course set) |

## Videos (`results/videos/`, 1080p MP4)

| Video | Shows |
| --- | --- |
| `ghost_race_00_tour.mp4`, `ghost_race_01_level_0_course_0.mp4` to `ghost_race_06_level_5_course_0.mp4` | ten stages of the `progress-demo` training run (untrained to 400 updates; not a final policy) on the same unseen course at once, seen from above; recorded games of the offline bot |
| `real_minecraft_tour.mp4`, `real_minecraft_stairs.mp4`, `real_minecraft_sprint.mp4`, `real_minecraft_showcase.mp4` | the bot's first-person view in real Minecraft 1.21.4 on three hand-designed courses (the showcase joins them), first final policy, course list given |
| `real_minecraft_*_no_course_list.mp4` | the same four videos with the route planner instead of the course list, first final policy |
| `replay_233_no_course_list_level_4_course_9_obstacles.mp4`, `replay_240_no_course_list_level_5_course_6_obstacles.mp4` | 3D replays of two real-Minecraft planner games on obstacle courses, the plan drawn as numbered cyan platforms, first final policy |
| `replay_295_learned_no_course_list_level_5_course_1.mp4` | 3D replay of the learned no-route model in real Minecraft on a level-5 course the planner failed with the first final policy |
| `real_minecraft_neo_level7.mp4` | first person, real Minecraft: the neo policy completes a held-out level-7 course with one neo (164 ticks, 0 server corrections) |

## Evaluations (`results/`)

| File | What |
| --- | --- |
| `final_policy.pt` / `.json` | the final policy (`runs/obstacles-g995`, discount 0.995) |
| `final_policy_v1.pt` / `final_policy_v1.json` | the first final policy (`runs/obstacles-ft`, discount 0.99) |
| `jump_eval.json`, `jump_eval_v1.json`, `jump_eval_main.json` | every jump type from a standing start at 16 angles: final, first final and main-run policy |
| `stress.json`, `stress_v1.json`, `stress_main.json` | course styles outside training, 100 courses per level and condition, same three policies |
| `offline_summary.md`, `live_summary.md` (and `offline_summary_v1.md`, `live_summary_v1.md`) | the bot's per-level tables offline and in real Minecraft, final and first final policy |
| `physics_parity.json` | the simulator against prismarine-physics 1.11.1: 99,000 ticks, largest error 1.4e-14 blocks |
| `ice_preview.json` | the final policy on the standard test courses with every platform (or every platform after the start) made of ice: 95.6% / 96.5% against 98.4% (`scripts/ice_preview.py`) |
| `physics_parity_ice.json` | the same check with ice blocks (optional slipperiness grid): 60,000 ticks, largest error 7.1e-15 blocks |
| `hparams_defaults_validation.json` | the defaults' validation score for the hyperparameter table |
| `audit_claims.json` | the audit of 829 claims in the documents against the run files (28 Oct) |
| `level5_by_position.json` | failure rate by position in the course on 3,000 extra level-5 courses (seed family 11, analysis only; `scripts/level5_positions.py`) |
| `level5_pause_test.jsonl` | a test-time pause after each landing (0, 3, 6, 10 ticks) on 2,000 level-5 courses of family 11: worse (`scripts/level5_pause_test.py`) |
| `jump_failures.json` | which jump each failed test course failed on, with the final policy: failure rate per jump type, by level and jump index, with and without a beam (`scripts/jump_failures.py`) |
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
| `one_policy_comparison.json` | the neo policies on levels 0-5 (the same courses as the standard test set) and the final policies on levels 6-8 and the 210 neo courses (`scripts/one_policy_check.py`) |

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
`docs/report_draft.md` has the report text. `results/audit_claims.json` has
the audit of the documents' claims against the data: 829 claims, 111 flagged
wrong, 60 confirmed wrong, 14 disputed, 37 overturned.

The cloud `report_draft.docx` and `.pdf` were generated before the PC merge. Rebuild them from the merged report and verify the page count before submission.

## PC integration and new training

`neo_policy.pt` / `neo_policy.json`: validation-selected neo-dagger3 checkpoint; `neo_policy_provenance.json` records its source and hashes. `runs/g995-*` archives are produced by the report as those runs finish. Their `test_once` protocol records validation every 50 iterations and tests the validation-selected best only at completion.
