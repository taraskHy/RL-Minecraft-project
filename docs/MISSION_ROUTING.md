# Mission: RL routing (choose the next platform without the course list)

A self-contained brief for an autonomous agent working on a teammate's
Windows PC with an NVIDIA GPU, from a fresh clone of
<https://github.com/taraskHy/RL-Minecraft-project>. Written on 29 Oct 2026 by
the main session on the training PC. Read this file to the end before you
start; it is meant to answer every question you would otherwise ask.

## 0. In one paragraph

The project's final policy (`results/final_policy.pt`) completes 98.4% of
unseen parkour courses in the simulator and 59 of 60 in real Minecraft, but
only when it is told the course: which platform it stands on and which three
come next. On a new map nobody tells it. Today a hand-written greedy planner
picks the next platforms from the blocks around the bot, and the same
policy then completes 92.3% instead of 98.4%. Most of the lost courses are
route choices, not failed jumps. **Your mission is to replace that planner
with a policy trained by reinforcement learning** that picks the next
platform from what the bot can see, beats the greedy planner on the
validation courses, and confirms it once on the test courses, on the offline
bot and in real Minecraft.

## 1. Why it matters

- **It runs on new maps.** With a route policy the bot needs only the goal's
  position (a beacon), not a hand-made list of platforms, so it can play any
  course built in Minecraft.
- **The presentation.** The course is about RL. The greedy planner is a
  heuristic; a learned router is the RL answer to the project's biggest
  remaining gap, and the results (simulator, offline bot, real game) slot
  straight into the results page and the report.

## 2. Baselines: the numbers to beat

All held-out courses have 10 jumps; levels 0-5 (`configs/jump_levels.json`).
"Standard" courses have no obstacles; "obstacle" courses have a head-hitter
beam or a decoy pillar on 50% of the jumps. Fall rule `local` (the default:
an episode fails when the feet drop 1 block below every nearby platform top)
unless stated. The jumper is always the final policy, unchanged.

### 2.1 Validation (tune on these; seed family 2 standard, family 5 obstacles)

300 courses per level (1,800 per set), simulator, from
`results/planner/validation_tuning.jsonl` (rows written on 28 Oct by a tuning
script in a cloud session, on the CPU, with the same evaluation as
`parkour.planner`'s `evaluate_planned`; fall rule `local`):

| Router | Standard (family 2) | Obstacles (family 5) | Failures: wrong platform / nothing in view / jump |
| --- | --- | --- | --- |
| **greedy planner, current default** (`cost=distance`, `align=2.0`) | **92.78%** (L0-L5: 97.7, 95.3, 94.3, 94.0, 90.3, 85.0) | **89.00%** (97.0, 93.0, 94.7, 88.0, 85.0, 76.3) | standard 112 / 0 / 18; obstacles 145 / 8 / 45 |
| greedy planner, earlier setting (`cost=difficulty`, `align=3.0`) | 91.28% | 87.39% | standard 139 / 0 / 18; obstacles 178 / 8 / 41 |

**Done means beating the greedy planner on these two sets** (section 9):
these 92.78% and 89.00%, or the planner's numbers re-measured on your
machine, which is the fair comparison (section 8).

### 2.2 Test (report once, at the end; seed family 1 standard, family 4 obstacles)

1,200 courses per set (200 per level), simulator, `results/README.md`
("Without the course list") and `results/planner/sim_test_*_tuned.json`:

| Router | Standard | Obstacles | Source |
| --- | --- | --- | --- |
| course list given (upper bound) | 98.42% (L0-L5: 100.0, 98.5, 100.0, 99.5, 99.0, 93.5) | 95.25% (99.5, 98.5, 97.5, 98.0, 97.0, 81.0) | `results/planner/sim_test_*_tuned.json`, `given_route` |
| **greedy planner, current default** | **92.33%** (98.5, 95.0, 95.5, 92.0, 90.5, 82.5) | **90.50%** (96.5, 95.5, 91.5, 93.0, 88.0, 78.5) | same files, `planner_route` |
| greedy planner, `remaining` fall rule (drops onto a later, lower platform count as landings) | 94.67% | 92.42% | same files |
| greedy planner, earlier setting | 90.9% | 88.1% | `results/README.md` |
| learned no-route model `noroute-v2` | 85.4% (97.0, 91.5, 94.5, 90.5, 87.0, 52.0) | 70.9% (82.5, 76.0, 76.5, 73.5, 71.5, 45.5) | `results/README.md`, `results/runs/noroute-v2/` |

Test failure counts of the current planner (local rule): standard 1,108
completed, 80 wrong platform, 0 nothing in view, 12 jump failures;
obstacles 1,086 completed, 80 wrong platform, 4 nothing in view, 30 jump
failures (`results/planner/sim_test_*_tuned.json`, `failure_modes`).

### 2.3 Offline bot and real Minecraft (the same 60 + 60 test courses)

The first 10 test courses of each level, standard and obstacle sets; course
files `results/games/v2-tuned-planner-live/courses.json` and
`results/games/v2-tuned-planner-live-obstacles/courses.json`:

| Router | Offline bot (prismarine-physics, no server) | Real Minecraft 1.21.4 | Source |
| --- | --- | --- | --- |
| course list given | 59/60, 57/60 | 59/60, 57/60 | `results/games/v2-offline*`, `v2-live*` |
| **greedy planner, current default** | **51/60, 56/60** | **51/60, 56/60** | `results/games/v2-tuned-planner-*` |
| learned no-route model `noroute-v2` | 55/60, 37/60 | 55/60, 37/60 | `results/games/noroute-*` |

In every one of these runs the outcome matched the simulator on every course,
and the real-Minecraft runs had 0 server corrections (`results/DATA.md`).
Hold your router to the same standard.

### 2.4 The learned no-route model, in brief

`noroute-v2` (`results/runs/noroute-v2/config.json`): trained from scratch for
3,000 iterations with obstacles on 30% of jumps, discount 0.99, the platform
slots and the progress input hidden (`--hide platforms,progress`), and the
extended observation (`--extended-obs`: a wide coarse block scan and its last
three landing spots). The route-based progress reward is used in training
only. Validation-selected iteration 2849 (validation 83.2%). Its weights are
in Git as `results/noroute_policy.pt` (provenance in
`results/noroute_policy_provenance.json`). It is weaker than the planner,
and decoy pillars fool it.

## 3. How the greedy planner works and fails

`parkour/planner.py` (mirrored in `bridge/planner.js`; tests keep them equal):

- **Perception** (`perceive`): solid blocks within `radius` 12 blocks
  (horizontal square), 10 below and 4 above the feet. A standable cell is a
  solid block with two air blocks above it and (`slab_only`) air below it, so
  wall and pillar tops are not platforms. Standable cells at one height that
  touch side by side form a platform, used as its bounding rectangle
  `(x0, x1, z0, z1, y)`. Slabs 4 blocks above a platform they touch are head
  hitters, not targets.
- **Choice** (`options`, `choose`): from the platform it stands on, the next
  one is the unvisited platform that minimises `cost - align * cos(angle to
  the goal)`, with `cost` the horizontal air gap (`cost=distance`), among the
  jumps within limits (air gap up to `MAX_GAP[dy] + slack` from the
  feasibility study, height change -3 to +1, straight flight path clear of
  blocks). The rule repeats from the chosen platform to fill the slots after
  it and stops at the goal platform.
- **When** (`GreedyPlanner.update`): at the start and after every landing on
  a new platform. Landing beyond the planned next platform marks the skipped
  ones as passed (`mark_passed`).
- **Slots** (`GreedyPlanner.slots`): the platform stood on plus up to three
  planned ones fill the policy's 4 platform slots; `progress` (the last
  observation number) becomes `1 - distance to goal / distance at start`.

**Failure modes** (`failure_modes()` in `parkour/planner.py`): *wrong
platform* means that at some re-plan the planner's next platform was not the
course's next platform (it marks a deviation, not necessarily the cause);
*nothing in view* means no candidate was in range; *jump* (`motor`) means
the plan followed the course and the jump failed. On validation,
wrong-platform failures are 112 of 130 failed standard courses and 145 of 198
failed obstacle courses (section 2.1). So the router, not the jumper, loses
most courses. What goes wrong, from the recorded plans and games:

- **Shortcuts are usually fine.** The planner often skips ahead to a later
  platform that is in reach and closer to the goal. In the current
  planner's real-Minecraft runs, 30 of the 60 standard games left the
  course's order (each first time by skipping 1 to 7 platforms ahead) and 23
  of those 30 still finished; with obstacles 33 left it and 30 finished
  (computed from `results/games/v2-tuned-planner-live*/episodes.jsonl.gz`
  and `courses.json`).
- **Dead ends.** After a skip, the greedy chain can later come back to a
  skipped platform and then find no unvisited platform in reach: an empty
  plan, and the jumper, with only its current slot filled, falls. Every
  test run that ever had an empty plan failed (11 of the 109 failed standard
  runs and 14 of 143 obstacle runs with the earlier setting; computed from
  the plans in `results/planner/sim_test_*.json`).
- **Long drops count as falls.** The `local` fall rule only looks at the
  course platforms from one before to two after the current one
  (`EnvConfig.fall_rule`, `ParkourEnv.fall_range`), so dropping toward a
  lower platform further ahead ends the episode while the bot is still above
  its target. The `remaining` rule, which counts such drops as landings,
  turns 28 more standard and 23 more obstacle test courses into successes
  (`results/planner/sim_test_*_tuned.json`). The bots use `local` too, so a
  router must learn to live with it.
- **Stalls.** The stall counter (200 ticks) resets only when the player
  reaches a new course platform index, so landing back on an earlier
  platform does not reset it; refusing a planned jump ends in a truncation.
- **Hard jumps** (`motor`): 12 standard and 30 obstacle test courses, mostly
  at levels 4 and 5 (level 5: 82.5% with the planner, 93.5% with the course
  list).
- **Obstacles.** Head-hitter beams are filtered out by the head-hitter rule,
  and decoy pillars, being 4-block columns, never become platforms
  (`slab_only`); the learned no-route model has neither rule, and 10 of its
  23 real-Minecraft obstacle failures end standing on an obstacle.
- **Clipping.** A platform that crosses the 12-block window is seen only up
  to the window's edge until the bot comes closer (visible in the recorded
  plans; usually in slots 2 and 3).

## 4. Recommended approach: hierarchical RL

A **high-level route policy** chooses the next target platform; the **frozen
final policy** jumps to it. Train the route policy with PPO in the batched
simulator.

**Decision points.** At the start of an episode and every time the player
lands on a new platform (the planner's `update` rule). Between decisions the
final policy acts every tick, as now.

**Action.** An index into the top-K candidate platforms (K about 8 to 16),
masked where fewer exist. Fill the jumper's slots consistently: slot 0 is the
platform stood on, slot 1 the chosen target; fill slots 2 and 3 either by
asking the route policy again from the chosen platform (autoregressive) or,
to start, with the greedy rule from it. The final policy was trained with a
real chain in its slots, so keep the chain plausible.

**Candidates.** Take them from the planner's own perception (`perceive`,
head-hitter filter, jump limits from `options`), so that training sees the
same candidates as the bot. Rank them (for example by the greedy cost) and
keep the top K. Courses are static, so you may precompute every course's
platforms once when a pool is built and filter by the 12-block window at each
decision on the GPU; if you do, test that the result equals `perceive` on the
validation courses.

**Observation of the route policy.** Per candidate: its four top-face corners
and top height in the player's frame (the same encoding as `Observer.slots`
in `parkour/env.py`), size, air gap and height change from the current
platform, the greedy cost, the cosine of the angle to the goal, distance from
the candidate to the goal, a visited flag, an is-goal flag. Global: goal
direction and distance, the current platform, the block scan around the
player (the observation's `scan` part), and a memory of the visited platforms
(for example the last 3 to 5 landing spots, like `EnvConfig.extended_obs`'s
memory). Sort or permutation-encode the candidates; a small shared MLP per
candidate with a softmax over candidates (pointer-style scoring) works well
for a variable set.

**Reward.** Per decision: course progress (decrease of the remaining path
length to the goal; in training you may use the true course for this, as
`noroute-v2` did, never as an input), a bonus when the goal is reached, a
penalty on a fall, a small cost per decision or per tick. Discount per
decision (semi-MDP; or `gamma ** ticks`). Episodes end on goal, fall, the
stall limit (`stall_ticks`, 200 ticks) or the time limit, as in `ParkourEnv`.

**Training.** PPO in `ParkourEnv` with many environments; the low-level
policy is `results/final_policy.pt`, frozen and deterministic (argmax), as on
the bot. Train on generated courses (seed family 0, `train.CourseFactory`)
with obstacles on 30% of jumps, like the final policy. Select checkpoints on
validation (families 2 and 5). Start simple: K small, greedy fill of slots 2
and 3, then grow.

**Speed.** `evaluate_planned` runs the planner on the CPU, one environment at
a time in Python. Decisions are rare: about 8 re-plans per course, one every
15 ticks on average (computed from the plans in
`results/planner/sim_test_*.json`), but at thousands of environments the
Python loop dominates. Either use fewer environments (512 to 2,048), or
batch the candidate extraction (precomputed platforms per course, filtered on
the GPU), or both.

**The judge stays the environment.** Success, falls and stalls are scored on
the true course by `ParkourEnv` (`evaluate_planned` does the same), with the
fall-rule window and the stall counter described in section 3. Your training
reward may be your own, computed in your code from the environment's state,
but do not change the environment's reward or its termination rules.

**Deployment.** The bot plays in real time from JavaScript (`bridge/`). Two
ways to run the route policy there: port its feature code and network to
JavaScript (the pattern of `bridge/policy.js`, with a parity test against
Python like `tests/test_bridge_parity.py`), or serve it from Python, as the
neo assist does (`parkour.neo_assist --serve`, called by the bot;
`tests/test_planner_service.py`). Either way, add an opt-in flag to
`parkour.live` (next to `--planner`) and keep the defaults unchanged.

### 4.1 Implementation notes from the code

- **Leave `parkour/env.py` alone; wrap it.** A new module (for example
  `parkour/route_env.py`) owns a `ParkourEnv` and, every tick, does what
  `evaluate_planned` does: `obs = env.observe()`, overwrite the slot part
  `obs[:, 11:55]` with `env.obs_fn.slots(env.state, R, V, G)` and the
  progress input `obs[:, 254]` with your progress estimate, then
  `jumper.act(obs, deterministic=True)` and `env.step`. `R` (rects, N x 4 x
  5), `V`, `G` (valid and goal flags) and `P` stay on the GPU and change only
  at decisions. Load the jumper with `parkour.live.load_model`, run it under
  `torch.no_grad()` and never update its normaliser.
- **Decision trigger** (`GreedyPlanner.update`): a row that is on the
  ground, with its feet within 1e-6 of a whole block height, standing on a
  platform other than its current one (landing anywhere new, not only on the
  target), plus every row that `env.step` just reset (`info["done"]`; the
  reset happens inside `step`, so copy what you need first), which needs a
  start decision.
- **Candidates on the GPU.** For the generator's courses, the platforms in
  `pool.plat[:pool.n]` should be exactly what `perceive` finds once the
  head-hitter rule drops the beams, because platforms and obstacles keep
  clear of each other (`courses._fits`, `_clear_of`); prove it with a test
  on the validation courses before relying on it. Then filter per decision as `perceive` and `options`
  do: the 25 x 25 window around the player's block, rows 10 below to 4 above
  the feet, rects clipped to the window, height change -3 to +1, air gap up
  to `MAX_GAP[dy] + 0.5`, `path_clear`, the head-hitter rule, a visited mask,
  the goal flag. Courses swapped in by a pool refresh need their tables
  rebuilt and their rows' route state reset.
- **Never feed the course list.** `env.cp`, `pool.rem` and the order of
  `pool.plat` are privileged: sort candidates by a geometric key with fixed
  tie-breaks and never use a course index as a feature. Your training
  reward may use the true course.
- **Keep the jumper in its training distribution.** It saw a real chain in
  slots 1 to 3; a target followed by empty slots looks to it like the goal.
  Fill slots 2 and 3 (`planner.choose(..., first=target)` does it greedily in
  Python). Its jumps were trained on course jumps only, so long shortcuts are
  unseen.
- **Encoding details.** `Observer.slots` encodes exactly 4 slots per call
  (`PLATFORM_SLOTS`); with `extended_obs` the progress input is still index
  254, not the last one; `evaluate_planned` cannot evaluate
  extended-observation checkpoints, `train.evaluate` can.
- **Semi-MDP PPO.** One transition per decision per environment, with the
  environment's own reward summed between decisions (or your own), discount
  `gamma ** ticks` or per decision; success and fall are terminal, stall and
  the time limit count as failures, as in evaluation. Reuse `train.py`'s
  pieces: GAE, the dual clip, the KL stop and KL-adaptive learning rate,
  `RunningNorm` with a shadow copy, `train._save`.
- **Memory.** A `CoursePool` keeps a 64 x 32 x 64 block grid per course
  (128 KiB): 1 GiB for 8,192 courses, about 230 MiB for a 1,800-course
  validation set.
- **In the bots** (`bridge/policy.js`): `Episode` takes a planner object
  with `start(pos)`, `update(pos, onGround, tick)`, `slots()`, `progress`
  and `events`; a JavaScript router implementing that interface plugs in
  next to `GreedyPlanner`, behind an opt-in option, with flags in
  `offline_bot.js`, `parkour_bot.js` and `parkour.live`. The bots' `Policy`
  accepts only the jumper's JSON format, so give the router its own export.
  A Python service is the other way (`bridge/assist.js` spawns
  `parkour.neo_assist --serve`; the module name is hard-coded there; live
  services need `--realtime` so the watchdog never pauses them mid-game, and
  the bot freezes its physics while it waits).
- **Parity tests to copy.** `tests/test_planner.py::PlannerParity` (24
  generated courses, half with obstacles: the Python evaluation with settle
  10 and the offline bot make identical plans, outcomes and tick counts),
  `tests/test_bridge_parity.py` (observation within 1e-5, logits within
  1e-3) and `tests/test_planner_service.py` for a service. Do the geometry in
  float64 with the planner's own operation order.

## 5. Alternatives (if hierarchical RL stalls)

- **Extend the end-to-end no-route model.** Fine-tune `results/noroute_policy.pt`
  (`--init-from` with the same `--extended-obs --hide "platforms,progress"`;
  the final policy has 255 inputs against 657 and cannot be loaded that way
  without padding its first layer and normaliser), or change its scan,
  memory or critic (a critic that sees the course would be new). It needs no
  planner at all, but so far it is the weakest option (85.4% / 70.9% on
  test).
- **A learned candidate scorer.** Keep the planner's perception and limits,
  and learn the score that ranks candidates: a contextual bandit (or a
  one-step PPO) whose reward is whether the jump policy reaches the course's
  goal after taking that candidate. It is the smallest change, and it is
  still RL if trained from episode outcomes.

## 6. Already tried and ruled out

On the validation courses (`results/planner/validation_tuning.jsonl`, 52 rows):

- `cost=difficulty` (minus the log of the jump's forgiveness) with `align`
  from 0.5 to 4: at best 92.5% standard and 89.8% obstacles (100 courses
  per level), against 93.0% and 90.5% for `cost=distance, align=2.0` on the
  same courses.
- `value_pick` 2 to 6 (take the chain the policy's own critic values most;
  tried only with the earlier `cost=difficulty, align=3.0`): 84.7-87.7%
  standard against 92.5% without it, and 87.7-90.3% obstacles against 89.0%
  (100 courses per level). The critic was trained only on the course's own
  platforms and is a poor judge of other routes.
- Hard drop limits (`dy_min` -2 or -1): they break the planner, since
  courses have real 3-block drops (15.3% and 1.8% standard, 600 courses).
- The head-hitter rule (a slab 4 blocks above a platform it touches is a
  beam, not a target) is kept: it raised obstacle validation from 83.0% to
  87.3% with the first final policy (recorded in commit `71faf51`'s message;
  no results file).
- `drop` 0.25 to 2 (a cost per block a jump goes down): never better on both
  sets; with the current default it lowered standard validation to 92.28,
  91.56 and 89.17% (1,800 courses, against 92.78%).
- `lookahead` (rank by the jump's cost plus the cheapest onward jump): 93.33%
  standard but 88.44% obstacles, against 92.78% and 89.00% without it, an
  exact tie over both sets (3,272 of 3,600 each); left off.
- `cost=distance, align=2.0` (+1.5 and +1.6 points over the earlier
  default) is the default in Python and JavaScript, confirmed in real
  Minecraft (51/60 and 56/60). The radius, the height window, the gap
  limits, `slab_only`, `mark_passed` and `clear_path` were never varied.
- Hiding the platform positions in training (the `noroute` arm of the reward
  ablation: the 98-point block scan, the goal and the progress input, no
  obstacles, 1,000 iterations): 82.4% on test, level 5 43.0%
  (`results/README.md`). The scan reaches 5.5 blocks ahead and 1.5 to each
  side, so after a turn the next platform is often out of sight. The
  extended observation (`noroute-v2`) reached 85.4% / 70.9%.
- Nobody has built an RL router, a learned candidate scorer or a no-route
  model with a privileged critic yet (`noroute-v2` hides the inputs from the
  critic too).

## 7. Files to read first

1. `AGENTS.md` (rules for agents) and `HANDOFF.md`: the team split at the top,
   section 5 (do not touch), section 6 (machine and environment).
2. `parkour/planner.py`: the baseline end to end (`perceive`, `options`,
   `choose`, `GreedyPlanner`, `evaluate_planned`, `failure_modes`, the CLI).
3. `parkour/env.py`: the observation (`OBS_LAYOUT`: proprio 6, action 5,
   platforms 4 x 11, scan 196, goal 3, progress 1 = 255 numbers), the slot
   encoding (`Observer.slots`), `ParkourEnv.step`, `EnvConfig` (`fall_rule`,
   `stall_ticks`, `extended_obs`), the reward terms.
4. `parkour/train.py`: the PPO loop to adapt (rollout, GAE, update, KL stop,
   validation, `--test-once`), `held_out_courses` and the seed families,
   `CourseFactory`, `evaluate`.
5. `parkour/model.py`: `ActorCritic`, its running normaliser and the JSON
   export the bot loads.
6. `bridge/planner.js`, `bridge/policy.js`, `bridge/parkour_bot.js`,
   `bridge/offline_bot.js`: how the bot plans and acts.
7. `parkour/live.py` (offline and real-Minecraft runs, `--planner`) and
   `parkour/neo_assist.py` (`--serve`: a Python service the bot calls).
8. `tests/test_planner.py`, `tests/test_planner_service.py`,
   `tests/test_bridge_parity.py`, `tests/test_heldout_fingerprints.py`.
9. `results/README.md` (sections "Without the course list" and "Learned
   no-route model") and `docs/design.md`.

## 8. Setup and commands (Windows, from the repository root)

**Prerequisites:** Windows 10 or 11 (x64), Git, Python 3.12 with the `py`
launcher, an NVIDIA GPU of the Turing generation or newer (RTX 20xx / GTX
16xx and up: the torch build targets compute capability 7.5+) with a driver
for CUDA 12.8, internet access (pypi.org, download.pytorch.org, nodejs.org,
github.com, registry.npmjs.org, piston-data.mojang.com) and about 7 GB of
disk inside the clone (the torch wheel alone is about 3 GB; pip and pnpm
also cache in the user profile). Clone into a **short path outside
OneDrive** (for example `C:\dev\RL-Minecraft-project`): OneDrive locks files
during writes, and without Windows long-path support the Node archive's
deepest file needs the repository path under about 90 characters.

```text
git clone https://github.com/taraskHy/RL-Minecraft-project.git C:\dev\RL-Minecraft-project
cd C:\dev\RL-Minecraft-project
git checkout -b routing/<name>          for example routing/hrl
setup.cmd                               Python 3.12 venv, torch 2.11.0 + CUDA 12.8, then portable Node 24, Java 21,
                                        pnpm, the bot's packages and the official Minecraft 1.21.4 server jar, all
                                        checksum-verified, into .venv, .local and bridge\node_modules
run.cmd test                            all checks (122 on 29 Oct); must pass before you change anything
```

In PowerShell type `.\setup.cmd` and `.\run.cmd`. Without `setup.cmd`'s
portable Node, about 20 tests are skipped (physics and JavaScript parity),
so read the summary line. `run.cmd` passes arguments through cmd.exe, which
splits unquoted values at commas, semicolons and `=`: quote such values
(`--hide "platforms,progress"`) or call `.venv\Scripts\python.exe -m
parkour.<module>` directly. For a check that must stay on the CPU, set
`CUDA_VISIBLE_DEVICES=-1` (an empty value does not hide the GPU on Windows).
**Never run `scripts/training_queue.py` there**: it is the training PC's
launcher, reads that PC's plan (`configs/training_queue.json`, with
checkpoints that exist only there) and adds seeds by itself.

Keep these two running in their own windows whenever GPU work runs:

```text
.venv\Scripts\python.exe scripts\gpu_watchdog.py --hot 65 --cool 58 --every 1
.venv\Scripts\python.exe scripts\keep_awake.py --pattern parkour.train --pattern parkour.live --pattern parkour.route_train
```

The watchdog suspends this project's GPU jobs above 65 C and resumes them
below 58 C (no state is lost). It recognises jobs by module name
(`GPU_JOBS` in `scripts/gpu_watchdog.py`), and it already covers any module
named `parkour.route*`, `parkour.router*` or `parkour.routing*`: **name
your GPU modules that way** (for example `parkour.route_train`); any other
name must be added to `GPU_JOBS` with a test in
`tests/test_gpu_watchdog.py`, and the watchdog restarted. Keep-awake holds
Windows awake (no power setting changes) while a Python process whose
command line contains one of its patterns runs, and exits when none does:
start it after your job starts. The training PC sleeps after 15 idle
minutes without it; check the teammate's PC the same way.

Baselines on your PC (each takes a few minutes). The section 2.1 rows were
measured on the CPU; the physics is float64 and deterministic, but the
network runs in float32, so on CUDA a few courses may come out differently.
Your own planner run, on the same machine and device as your router's
evaluation, is the baseline you compare against; report it next to the
section 2.1 numbers:

```text
.venv\Scripts\python.exe -m parkour.planner --family 2 --per-level 300 --out results\routing\planner_validation_standard.json
.venv\Scripts\python.exe -m parkour.planner --family 2 --obstacles --per-level 300 --out results\routing\planner_validation_obstacles.json
```

(`--family 2 --obstacles` plays family 5; careful: `--family 4
--obstacles` also plays family 5, and only `--family 1 --obstacles` plays
the obstacle test family 4.) The test sets (`--family 1`, 200 per level) are
for the end only.

Training and evaluation of your router: write them as new modules (for
example `parkour/route_policy.py`, `parkour/route_train.py`,
`parkour/route_eval.py`) with argparse CLIs like `parkour/train.py`, so
that one command trains (`python -m parkour.route_train --run NAME ...`) and
one evaluates a checkpoint on a chosen family. Reuse
`train.held_out_courses` for the held-out sets and `train._save` for
checkpoints.

Offline bot and real Minecraft (after you add your flag to `parkour.live`):

```text
run.cmd offline --checkpoint results\final_policy.pt <your route flag> --courses-file results\games\v2-tuned-planner-live\courses.json --out runs\route-offline
run.cmd offline --checkpoint results\final_policy.pt <your route flag> --courses-file results\games\v2-tuned-planner-live-obstacles\courses.json --out runs\route-offline-obstacles
run.cmd accept-eula --agree             ONLY the PC's owner, once, if he accepts https://aka.ms/MinecraftEULA
run.cmd live --checkpoint results\final_policy.pt <your route flag> --courses-file results\games\v2-tuned-planner-live\courses.json --out runs\route-live
run.cmd live ... --courses-file results\games\v2-tuned-planner-live-obstacles\courses.json --out runs\route-live-obstacles
```

The planner's own versions of these runs used `--planner` (the current
defaults; `--planner-config` takes JSON overrides; `--planner` and
`--assist` cannot be combined). `--viewer NAME` lets a person watch in
Minecraft Java 1.21.4 at `127.0.0.1:25570`. A `--out` folder must not exist
yet. Without `--courses-file`, `parkour.live` plays test courses (the first
`--per-level` of each level of family 1, or 4 with `--obstacles`), so for
bot checks during development write a validation course file from
`train.held_out_courses(..., family=2)` (or `obstacle_prob=0.5, family=5`)
as `{"courses": [c.to_json() for c in courses]}`. Each run writes
`summary.json` and `summary.md` with the bot's result, the simulator's
result on the same courses, the outcome agreement, the replay error and the
server corrections.

## 9. Done criteria

1. **Validation:** with the final policy as the jumper, your router completes
   more of the family-2 standard courses and more of the family-5 obstacle
   courses (300 per level, 1,800 each, fall rule `local`) than the greedy
   planner measured by you on the same machine and device (the reference
   numbers are **92.78%** and **89.00%**).
2. **Test, once:** the validation-chosen router on families 1 and 4 (1,200
   each), reported against the planner's 92.33% / 90.50% and the given
   route's 98.42% / 95.25%. One test run per router version; never choose
   on test.
3. **Offline bot:** more than 51/60 and 56/60 on the two course files of
   section 2.3, with the same outcome as the simulator on every course.
4. **Real Minecraft:** the same courses, more than 51/60 and 56/60, the same
   outcome as the simulator, 0 server corrections.
5. `run.cmd test` passes, including your new tests (parity between the
   Python and JavaScript sides of your router, or of the service protocol).
6. `results/routing/README.md` and its JSON files hold every number, each
   saying where it was measured, and a pull request into `main` is open.

## 10. Rules

- Work on your own branch `routing/<name>`; push only that branch; open a
  pull request into `main` (a draft early is fine) and keep it updated. If
  you cannot push to the repository, push to a fork and open the pull
  request from there. Never force-push or rewrite history.
- Do not modify `parkour/physics.py`, the environment's reward
  (`RewardConfig` and the reward code in `parkour/env.py`),
  `results/final_policy.*`, or anything else in HANDOFF.md section 5. Add
  opt-in flags instead; the defaults must keep every existing test passing,
  including `tests/test_heldout_fingerprints.py`.
- Tune and choose only on validation (families 2 and 5). Test families 1 and
  4 are for the final numbers only. Do not use seed families already taken
  (0 training, 1 test, 2 validation, 3 stress, 4/5 obstacle test/validation,
  7 the neo DAgger courses, 11 analysis, 12-17 other sets); new sets get new
  families, from 18.
- Every number says where it was measured: simulator, offline bot
  (prismarine-physics without a server) or real Minecraft. Every number
  comes from a file under `runs/` or `results/`.
- Results go to `results/routing/`: one JSON file per evaluation and a short
  `README.md` (what was tried, what was chosen and why, every number with
  its file). Name the JSON files `planner_*.json`, `route_*.json` or
  `router_*.json`: `results/DATA.md` already lists those patterns, so
  `tests/test_handoff.py` keeps passing; give any other file its own entry
  there. Avoid `*.log` names (Git ignores them). Bot runs write to `runs/`,
  which Git ignores: copy their `summary.json`, `summary.md` and
  `courses.json` into `results/routing/` (for example as
  `route_live_summary.json`). Ship the chosen router's weights as
  `results/routing/router_policy.pt` (and its JSON export as
  `router_policy.json`, if the bot needs one), with a
  `router_policy_provenance.json` like `results/noroute_policy_provenance.json`
  (`runs/` never reaches the repository).
- **Do not run `python -m parkour.report` (`run.cmd report`) or `python -m
  parkour.handoff` on your PC.** The report rebuilds `results/README.md` from
  the training PC's `runs/` (it now stops when that folder is missing), and
  the handoff snapshot is regenerated by the main session after it merges
  your pull request. `AGENTS.md`'s "after each result: report" step is for
  the training PC.
- GPU: at most two training processes at once (a third spills out of GPU
  memory and slows everything 5 to 10 times); the watchdog runs whenever
  GPU work runs.
- Python/JS parity: whatever the bot runs must match the simulator (the
  physics port matches prismarine-physics to 1.4e-14 blocks; keep that kind
  of check for your code).
- Keep your own session log at the top of `results/routing/README.md`
  (date, who you are, what changed, what is next) instead of in
  `HANDOFF.md`, whose log and generated block change on `main` all the time;
  the main session adds your summary there when it merges. Leave
  `HANDOFF.md` and `TEAM_STATUS.md` to the main session; your status lives in
  `results/routing/README.md` and the pull request.

## 11. Asking questions

Ethan and the teammate want as few questions as possible. Decide technical
matters yourself and record the reasons in `results/routing/README.md`.
Ask the PC's owner only for things only he can do: accepting the Minecraft
EULA (`run.cmd accept-eula --agree`), and GitHub access if pushing fails.
Ask about direction only if the done criteria above turn out to be wrong
or unreachable, with the evidence.

## 12. What a fresh clone has and lacks

In Git: the code, configs, tests, `results/` (every run's logs in
`results/runs/`, the bot games in `results/games/`, the planner results in
`results/planner/`), and these policies: `results/final_policy.pt` (the
jumper you use), `results/final_policy_v1.pt`, `results/neo_policy.pt`,
`results/noroute_policy.pt`.

Not in Git (they exist only on the training PC, and you do not need them):
`runs/` (training checkpoints and full logs; every run's small logs are in
`results/runs/NAME/`) and `.local/` (Node, Java, the server jar, the server
folder and the EULA consent; `setup.cmd` recreates everything except the
consent, which only the PC's owner can give). To retrain a run, read its
settings from `results/runs/NAME/config.json` and pass the values that
differ from the `TrainConfig` defaults to `python -m parkour.train --run
NAME`; `HANDOFF.md` section 6 lists the local-only checkpoints, how to
regenerate each and how long it took. Some configs and scripts point into
`runs/` and fail on a fresh clone (`configs/training_queue.json`,
`configs/sweep_extra.json`, `scripts/neo_evidence.py`); routing needs none
of them.
