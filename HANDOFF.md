# Handoff for the next assistant session

Read this, then [TEAM_STATUS.md](TEAM_STATUS.md), [docs/design.md](docs/design.md)
and [results/DATA.md](results/DATA.md). Current result tables:
[results/README.md](results/README.md) (`python -m parkour.report`).

## Goal and how the user works

- Goal (since 27 Oct 2026): RL bot that completes **unseen** Minecraft
  parkour courses and reaches the target. Requested on 27 Oct: harder jumps
  (neo, double neo, 4-block jumps). The earlier survival project is archived
  (`archive/survival` tag; `codex/*` branches untouched).
- The user manages at a high level and wants as few questions as possible.
  Ask only about direction or the user's own decisions.
- **One session changes the project.** On 28 Oct the user decided that only
  the main session edits files, commits and starts or stops runs; other
  sessions are for questions only. A side chat once committed and restarted a
  run in parallel; check `git log` and running processes before acting.
- Team of three students; course-level methods (PPO, actor-critic, GAE are
  in the course material). They need data for a presentation: keep
  `results/` complete (`python -m parkour.report` refreshes it).
- Repository: <https://github.com/taraskHy/RL-Minecraft-project>, branch
  `main`; pushing works (the user signed in on this PC). Never force-push.
  Push and update TEAM_STATUS after each result.

## Machine

Intel Core Ultra 7 270K Plus (24 cores), 128 GB RAM, RTX 5080 16 GB,
Windows 11, clock in Israel time (UTC+2; Git Bash labels it JST). Checkout:
`C:\Users\ethan\OneDrive\Desktop\etgar\RL\RL-Minecraft-parkour` (OneDrive
folder). Python 3.12 venv in `.venv` (torch 2.11.0+cu128). Portable Node,
Java and the 1.21.4 server jar in `.local` (ignored by Git). The Minecraft
EULA was accepted by the user on 27 Oct 2026
(`.local/minecraft-eula-consent.json`).

**GPU memory is the limit:** two training processes fit (since 28 Oct, with
the desktop apps open); a third spills into system RAM and everything slows
5-10 times. Windows sleeps after 15 idle minutes: `scripts/keep_awake.py`
holds it awake while training runs.

**GPU temperature rule (user, 28 Oct):** above 65 C pause the GPU work,
resume when cool. `scripts/gpu_watchdog.py` runs in the background: every
10 s it reads nvidia-smi; above 65 C it suspends the project's GPU jobs
by name (train, neo_demos, cem, planner, jumpeval, stress, feasibility,
cem_momentum_neo; suspended, so no state is lost) and resumes them below
58 C. It does not cover `parkour.neo_dagger` (also a GPU job) or the live
bot. Log: `runs/gpu_watchdog.log` (no pause so far on 28 Oct). If the
watchdog is ever killed while jobs are paused:
`python scripts/gpu_watchdog.py --resume-all`.

## Key facts learned

- Physics: prismarine-physics 1.11.1 ported exactly, **plus vanilla's 1e-7
  collision tolerance** in both the simulator (`COLLISION_EPSILON`) and the
  bot (`bridge/physics_patch.js`). Without it a player resting against a face
  can slip into a block corner through rounding. Parity with the patched
  engine: 1.4e-14 blocks over 99,000 ticks (`results/physics_parity.json`).
- Searches must never start the player inside a block
  (`physics.inside_solid`): the engine lets a player walk out of a block it
  already overlaps (this faked three "double neo proofs" once).
- Live bot: `/tp` coordinates must be decimals (whole numbers are centred);
  the bot deletes the server's movement-speed attribute each tick; it sets
  `bot.entity.yaw` directly.
- PPO needed safeguards (KL stop, dual clip, KL-adaptive LR); the first runs
  collapsed (`runs/_v1_unstable`). One short dip remains at the 1e-3 LR cap.
- Checkpoint selection uses the validation family (2); test families 1
  (standard) and 4 (obstacles) are only reported.
- Discount 0.995 was the only setting that beat the defaults by more
  than 3 points on validation (all three seeds, simulator). It is now the
  final policy's setting.
- Neo layouts: a single neo (wall flush with a 1-wide takeoff) is possible
  in 16 of the 18 layouts tried (none for gap 3, dy +1 with the wall at gap
  cell 1 or 3). The search found no double or triple neo in 64 layouts (36
  from a standing start, `configs/jump_cem.json`; 28 with a 3-4 block
  run-up, `configs/jump_cem_momentum.json`), so we treat them as impossible
  (not proven). Standing on any obstacle ends an episode as a fall.
- Wide-takeoff neos (user correction): the 1-wide takeoff before a neo is a
  simplification of our generator. Parkour allows wider ones. The search
  (simulator, `results/neo/wide_takeoff_search.json`, 3 layouts per takeoff
  size, both sides) lands neos from a 2x2 takeoff when the landing is next
  to the wall's open end and finds none when it is two lanes away (3x3
  middle lane: none). `courses._place(wide_neo=True)` allows them;
  `courses.from_specs` (hand-designed courses), the DAgger training courses
  (`neo_dagger.training_neo_courses`) and `python -m parkour.train
  --wide-neo` use it. The default generator and every test and validation
  set are unchanged.
- PPO alone did not learn neos: with or without the stall penalty the bot
  made 0 of the 13 single-neo jump types (simulator, standing start, 16
  angles) and mostly stood still at them (stalled in 95% of tries without
  the stall penalty, 89% with it; `results/neo/jump_eval_advanced-ft3.json`,
  `jump_eval_neo-stall.json`). Demonstrations plus imitation changed that
  (pipeline below).
- Noisy search labels: the first DAgger training
  (`runs/neo-dagger-collapsed`) fell to 59% held-out success (simulator,
  levels 0-8). The search filled ticks that do not matter with random
  actions: only 3.7% of its pairs matched the bot's own choice on all five
  action parts (`results/neo/label_agreement.json`). Two fixes in
  `parkour.neo_dagger`, both on by default:
  `--prior` starts the search from the policy's own action probabilities,
  and `--simplify` puts the bot's own choice back wherever the neo still
  lands. Since then the labels agree with the bot 82-99% per action part
  in rounds 1-3 (89-99% in round 1; turning lowest). The collapsed run
  also used imitation weight 1 and the later rounds 0.3, so the data do
  not separate the two causes.
- Neo detection: in round 1 of DAgger the detector also counted
  one-block-up jumps as neo takeoffs. On that round's 1,024 training
  courses, 1,090 of the 2,526 takeoffs it found (43%) were real neos
  (`results/neo/detector_check.json`, seed 2, the courses of
  `runs/neo_dagger_r1b.pt`; the file counts the collapsed attempt as round
  1 and calls this round 2). Since round 2 only wall blocks (blocks that
  are not part of a platform) count.

## Final policy (discount 0.995)

- `results/final_policy.pt` / `.json`: `obstacles-g995` iteration 699
  (`main-g995`, 3,000 iterations, then a 1,000-iteration obstacle
  fine-tune). The first final policy (`obstacles-ft`, discount 0.99) is
  `results/final_policy_v1.pt` / `.json`. Validation: 97.6 vs 96.6.
- Test, simulator (1,200 courses each): standard 98.4 (L5 93.5) vs 97.1
  (90.5); obstacles 95.2 (L5 81.0) vs 93.6 (75.5).
- Bot on the same 60 + 60 held-out courses and the 4 showcase courses:
  offline and real Minecraft identical, 59/60 standard, 57/60 obstacles,
  3/4 showcase (the 4th is the neo course, outside this policy's
  training), 0 server corrections. First policy: 59/60, 58/60, 3/4. The
  60-course samples are within noise; the 1,200-course simulator test sets
  show the new policy better.
- Route planner with the new policy: simulator 90.9 standard / 88.1
  obstacles (first policy 89.6 / 86.2); offline and real Minecraft 50/60
  standard, 55/60 obstacles, identical, 0 corrections (first policy 52/60,
  53/60).
- Videos in `results/videos/`: the first-person showcase videos
  (`real_minecraft_*.mp4` except the neo one, with their
  `_no_course_list` planner versions) and the planner replays
  (`replay_233_*`, `replay_240_*`) were made with the first final policy.
  Label them so. The other two replays are named for their policy:
  `replay_137_neo_policy_*` (neo policy) and `replay_295_learned_*`
  (learned no-route model `noroute-v2`). The ghost races
  (`ghost_race_00`..`06`) show ten training stages (untrained to 400
  updates) of the separate `progress-demo` run.

## Neo pipeline (separate neo policy)

The neo policy is a fine-tune of the discount-0.99 line on levels 0-8:
`advanced-ft3` -> `neo-stall` (stall penalty 5) -> `neo-bc` ->
`neo-dagger` -> `neo-dagger2` -> `neo-dagger3`. The final policy never
trained on neos. Each run started from the previous one's checkpoint
(see `runs/<name>/config.json`).

1. `python -m parkour.neo_demos`: cross-entropy search in the bot's own
   discrete action space on the 16 landable single-neo layouts, both wall
   sides. 50,706 verified landing sequences (`runs/neo_demos.json`); 12,800
   used (`runs/neo_demos_train.json`, 400 per layout and side).
2. `python -m parkour.train --demos runs/neo_demos_train.json --bc-coef 1
   --demo-courses 1024`: a behaviour-cloning loss next to PPO (weight 1 for the whole run;
   the decay option was off, and `bc_decay` now defaults to off in
   `parkour/train.py`) and 1,024 pool courses that start where a
   demonstration starts. `neo-bc`, 800 iterations.
3. `python -m parkour.neo_dagger --checkpoint <ckpt> --out
   runs/neo_dagger_rN.pt`: the policy plays training courses with neos
   (levels 6-7, never held-out ones); at each neo takeoff it reaches, its
   exact state is recorded and a batched search continues to a landing. The
   observations along those sequences become (observation, action) pairs.
   Then `python -m parkour.train --bc-pairs
   runs/neo_dagger_r1b.pt,runs/neo_dagger_r2.pt,... --bc-coef 0.3
   --wide-neo` (weight 0.3, constant; same demonstrations as
   step 2), 500 iterations per round; each round adds its own pair file.
4. Chosen neo policy: `runs/neo-dagger3/checkpoints/best.pt` (chosen on
   the validation neo courses: 34.4% vs 33.0% for round 2's best).

Results (README section "Neo jumps", data in `results/neo/`):

- 210 held-out test courses of levels 6-8 with a neo (337 neo jumps),
  simulator, courses completed: 0% before, 23.3% imitation, 29.5% DAgger 1,
  30.0% DAgger 2, 29.0% DAgger 3 (chosen). Rounds 2-3 level off at about
  30% with the checkpoints chosen on validation (round 2's last
  checkpoint: 35.2%, simulator). By neos per course (chosen policy, simulator,
  `results/neo/by_neo_count.json`): one 56/129, two 3/47, three 2/25,
  four or more 0/9.
- Single neos from a standing start (simulator): all 13 types made, 60.6%
  mean (`results/neo/jump_eval_neo-dagger3.json`; round 1 had 70.2%).
- Neo course set (`configs/neo_live_courses.json`, 30 held-out test
  courses and the showcase neo course): 11/31 offline and 11/31 in real
  Minecraft, identical to the simulator, 0 corrections. Every completed
  course had one neo. The showcase neo course is not completed: the chosen
  neo policy passes its first neo (from a 2x2 block, by jumping around the
  wall's open end) and falls at the second neo after 3 of 11 jumps, the
  same in the simulator, offline and in real Minecraft (episode 30 of
  `runs/neo-offline-final` and `runs/neo-live-final`).
- Video: `results/videos/real_minecraft_neo_level7.mp4` (first person,
  real Minecraft, a held-out level-7 course with one neo, 164 ticks, 0
  corrections).

## Where things are

- Final policy: `results/final_policy.pt` / `.json` (`obstacles-g995`);
  first one: `results/final_policy_v1.*` (`obstacles-ft`).
- Neo policy: `runs/neo-dagger3/checkpoints/best.pt`; neo data and logs in
  `results/neo/` (logs saved as `.txt`, because Git ignores `.log`).
- Runs: `runs/<name>/` (ignored by Git; logs copied to `results/runs/`).
- Sweeps: `scripts/sweep.py configs/<sweep>.json --slots N` (skips
  finished runs).
- Pages: `results/replay_viewer.html`, `results/progress.html`,
  `results/neo_preview.html` (published as private artifacts).
