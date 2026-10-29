# Mission: RL routing (choose the next platform without the course list)

A self-contained brief for an autonomous agent working on a teammate's
Windows PC with an NVIDIA GPU, from a fresh clone of
<https://github.com/taraskHy/RL-Minecraft-project>. Written on 29 Oct 2026 by
the main session on the training PC and checked against the repository by
independent reviewers. Read this file to the end before you start; it is
meant to answer every question you would otherwise ask.

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

Levels 0-5 (`configs/jump_levels.json`). Held-out courses are generated with
10 jumps, but generation stops when the next jump does not fit, so a few are
shorter: all 10 jumps in 1,748 of the 1,800 family-2 courses, 1,665 of 1,800
family-5, 1,151 of 1,200 family-1 and 1,129 of 1,200 family-4 (computed by
regenerating the sets with `train.held_out_courses`). Use `pool.n` and never
assume 11 platforms. "Standard" courses have no obstacles; "obstacle" courses
have a head-hitter beam or a decoy pillar on 50% of the jumps. Fall rule
`local` (the default: an episode fails when the feet drop 1 block below every
nearby platform top) unless stated. The jumper is always the final policy,
unchanged.

### 2.1 Validation (tune on these; seed family 2 standard, family 5 obstacles)

300 courses per level (1,800 per set), simulator, from
`results/planner/validation_tuning.jsonl`. The rows were written on 28 Oct
by a tuning script of a cloud session, on the CPU; the script is not in the
repository and the rows record no fall rule, settle or tick limit
(`evaluate_planned`'s defaults are fall rule `local`, settle 0, 4,000
ticks).

| Router | Standard (family 2) | Obstacles (family 5) | Failures: wrong platform / nothing in view / jump |
| --- | --- | --- | --- |
| **greedy planner, current default** (`cost=distance`, `align=2.0`) | **92.78%** (L0-L5: 97.7, 95.3, 94.3, 94.0, 90.3, 85.0) | **89.00%** (97.0, 93.0, 94.7, 88.0, 85.0, 76.3) | standard 112 / 0 / 18; obstacles 145 / 8 / 45 |
| greedy planner, earlier setting (`cost=difficulty`, `align=3.0`) | 91.28% | 87.39% | standard 139 / 0 / 18; obstacles 178 / 8 / 41 |

**Done means beating the greedy planner on these two sets, measured by you
on your machine** (sections 8 and 9); 92.78% and 89.00% are the reference.
Your baseline run also gives the course list's success on the same courses
(`given_route`), the upper bound.

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

In each of these runs the outcome matched the simulator on every course,
and the real-Minecraft runs had 0 server corrections (each run's
`summary.json` in `results/games/NAME/`: `outcome_agreement` is 1.0, and
`server_corrections` is 0 in every entry of `episodes`). Hold your router to
the same standard.

### 2.4 The learned no-route model, in brief

`noroute-v2` (`results/runs/noroute-v2/config.json`): trained from scratch for
3,000 iterations with obstacles on 30% of jumps, discount 0.99, the platform
slots and the progress input hidden (`--hide platforms,progress`), and the
extended observation (`--extended-obs`: a wide coarse block scan and its last
three landing spots). The route-based progress reward is used in training
only. Its checkpoint, iteration 2849, was chosen by `train.py`'s selection
score, 83.2% (families 2 and 5 together, 100 courses per level each; not
comparable with section 2.1's per-family numbers). Its weights are in Git as
`results/noroute_policy.pt` (provenance in
`results/noroute_policy_provenance.json`). It is weaker than the planner on
the 1,200-course test sets, although it made 55/60 against the planner's
51/60 on the 60 standard bot courses, and obstacles fool it: 10 of its 23
real-Minecraft obstacle failures end standing on top of one, 6 on beams and
4 on decoy pillars (computed from `results/games/noroute-live-obstacles`).

## 3. How the greedy planner works and fails

`parkour/planner.py` (mirrored in `bridge/planner.js`; tests keep them equal):

- **Perception** (`perceive`): solid blocks within `radius` 12 blocks
  (horizontal square), 10 below and 4 above the feet. A standable cell is a
  solid block with two air blocks above it and (`slab_only`) air below it, so
  wall and pillar tops are not platforms. Standable cells at one height that
  touch side by side form a platform, used as its bounding rectangle
  `(x0, x1, z0, z1, y)`. Head-hitter beams are one-block slabs with air above
  and below, so `perceive` finds them too; the head-hitter rule then drops a
  slab as a target when it sits 4 blocks above a platform it touches.
- **Choice** (`options`, `choose`): from the platform it stands on, the next
  one is the unvisited platform that minimises `cost - align * cos(angle to
  the goal)`, with `cost` the horizontal air gap (`cost=distance`), among the
  jumps within limits (air gap up to `MAX_GAP[dy] + slack` from the
  feasibility study, height change -3 to +1, straight flight path clear of
  blocks). The rule repeats from the chosen platform to fill the slots after
  it and stops at the goal platform.
- **When:** the first plan comes from `GreedyPlanner.start`;
  `GreedyPlanner.update` re-plans after every landing on a new perceived
  platform (landing on a cell that is not a perceived platform, such as a
  decoy pillar's top, keeps the old plan; a beam's top is a perceived
  platform, but standing on any obstacle ends the episode as a fall before a
  re-plan, in `ParkourEnv.step` (`on_obstacle`) and in the bots
  (`onObstacle`)). Landing beyond the planned next platform marks the
  skipped ones as passed (`mark_passed`).
- **Slots** (`GreedyPlanner.slots`): the platform stood on plus up to three
  planned ones fill the policy's 4 platform slots; `progress` (the last
  observation number) becomes `1 - distance to goal / distance at start`,
  measured at each re-plan (section 4, *The jumper's progress input*).

**The route loses most courses.** With the course list, the same jumper
completes 1,181 of the 1,200 standard and 1,143 of the 1,200 obstacle test
courses; with the planner, 1,108 and 1,086 (`given_route` and
`planner_route` in `results/planner/sim_test_*_tuned.json`): 73 and 57 more
courses lost to the route.

**Failure labels** (`failure_modes()` in `parkour/planner.py`, decided at
the first irregular re-plan): *wrong platform*: the planner's next platform
was not the course's next one (a deviation, not necessarily the cause; dead
ends after a skip count here); *nothing in view*: the plan was empty (no
unvisited platform within the jump limits, with a clear path, that is not a
head hitter; platforms may still be visible); *jump* (`motor`): the plan
followed the course at every re-plan and the run still failed (a fall, a
stall or the time limit). On validation, wrong-platform failures are 112 of
130 failed standard courses and 145 of 198 failed obstacle courses
(section 2.1). What goes wrong, from the recorded plans and games:

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
  test run that ever had an empty plan failed (11 standard and 14 obstacle
  runs with the earlier setting; computed from the plans in
  `results/planner/sim_test_*.json`).
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
- **Hard jumps** (`motor`): 12 standard and 30 obstacle test courses. Jump
  failures concentrate at level 5: with the course list, 13 of the jumper's
  19 standard and 38 of its 57 obstacle test failures are there
  (`given_route.per_level` in `results/planner/sim_test_*_tuned.json`). The
  planner's own level-5 gap (82.5% against 93.5% with the course list,
  standard) is mostly route failures.
- **Obstacles.** The head-hitter rule keeps beams from being targets, but it
  also drops a real course platform that sits 4 blocks above a beam it
  touches: 2 family-5 validation courses (level 4, seeds 65 and 291) and 1
  family-4 test course (level 4, seed 196) start with an empty plan and
  fail. Decoy pillars, 4-block columns, never become platforms
  (`slab_only`), but they block `path_clear` and the jumper can land on top.
- **Clipping.** A platform that crosses the 12-block window is seen only up
  to the window's edge until the bot comes closer (visible in the recorded
  plans; usually in slots 2 and 3).

## 4. Recommended approach: hierarchical RL

A **high-level route policy** chooses the next target platform; the **frozen
final policy** jumps to it. Train the route policy with PPO in the batched
simulator.

**Decision points.** At the start of an episode (`GreedyPlanner.start`'s
moment) and every time the player lands on a new perceived platform
(`GreedyPlanner.update`'s rule). Between decisions the final policy acts
every tick, as now.

**Action.** An index into the top-K candidate platforms (K about 8 to 16),
masked where fewer exist. Fill the jumper's slots consistently: slot 0 is the
platform stood on, slot 1 the chosen target; fill slots 2 and 3 either by
asking the route policy again from the chosen platform (autoregressive) or,
to start, with the greedy rule from it. If no candidate passes the filters,
take no action and fill the slots as the planner does with an empty plan:
slot 0 the platform stood on, slots 1 to 3 empty (`valid` False), the event
recorded with an empty chain; never take a softmax over an all-masked set.
This is not rare and not only on obstacle courses: 8 obstacle validation
courses failed with an empty plan as their first deviation (*nothing in
view*, 2 of them at the start), and with the earlier setting 11 standard and
14 obstacle test runs reached an empty plan at some re-plan, 21 of them
after leaving the course (labelled *wrong platform*); every one failed.

**Candidates.** Use the planner's final candidate list, so that training sees
what the bot sees: `perceive`, then the head-hitter rule, then `options`'
limits (jump limits, `path_clear`; visited platforms and the current one
are excluded). Rank them (for example by the greedy cost) and keep the top
K.

**Observation of the route policy.** Per candidate: its four top-face corners
and top height in the player's frame (the same encoding as `Observer.slots`
in `parkour/env.py`), size, air gap and height change from the current
platform, the greedy cost, the cosine of the angle to the goal, distance from
the candidate to the goal, an is-goal flag. Global: goal direction and
distance, the current platform, the block scan around the player (the
observation's `scan` part) and a memory of the visited platforms (for example
the last 3 to 5 landing spots). Sort or permutation-encode the candidates; a
small shared MLP per candidate with a softmax over candidates (pointer-style
scoring) suits a variable set.

**The jumper's progress input.** In training, the jumper's last input was
the fraction of course platforms reached, which a router cannot know. Use
the planner's estimate, `GreedyPlanner.progress`, which the bot already
computes the same way: at each re-plan, 1 minus the 3D distance from the top
centre of the platform stood on to the goal's top centre, divided by that
distance from the start platform, clamped to 0..1 and held until the next
re-plan. Like the training input, it changes only at landings; do not
recompute it from the player's position every tick.

**Reward.** Per decision: course progress (decrease of the remaining path
length to the goal; in training you may use the true course for this, as
`noroute-v2` did, never as an input), a bonus when the goal is reached, a
penalty on a fall, a small cost per decision or per tick. Discount per
decision (semi-MDP; or `gamma ** ticks`). Episodes end on goal, fall, the
stall limit (`stall_ticks`, 200 ticks) or the time limit, as in `ParkourEnv`.

**Training.** PPO in `ParkourEnv` with many environments; the low-level
policy is `results/final_policy.pt`, frozen and deterministic (argmax), as on
the bot. Train on generated courses (seed family 0) with obstacles on 30% of
jumps, like the final policy. Build the course factory as `train.train`
does: `CourseFactory(TrainConfig(obstacle_prob=0.3, seed=S), courses.load_levels(),
{course_hash(c) for c in tests + obstacle_tests + vals})`, with the held-out
sets from `train.held_out_courses` and `train.obstacle_courses`; an empty
exclude set silently lets held-out courses into training. Select checkpoints
on validation (families 2 and 5). Start simple: K small, greedy fill of slots
2 and 3, then grow.

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

**Deployment.** The bot plays in real time from JavaScript (`bridge/`).
Prefer porting the router's candidate features and network to JavaScript,
next to `bridge/planner.js`: that is the path already proven in real
Minecraft (the greedy planner, 0 server corrections), with parity tests like
`tests/test_planner.py` and `tests/test_bridge_parity.py`. A Python service
is the other way, but it has only run offline so far: it must take its
inputs from the bot (its state and the blocks or candidates it sees), never
from `--courses-file` or a course index (the neo assist service,
`parkour.neo_assist --serve`, builds its world from the course file and
plans by course index, so copy its protocol, not its inputs), and live runs
must start it with both `--serve` and `--realtime`, or the watchdog may
pause it mid-game. Either way, add an opt-in flag to `parkour.live` (next to
`--planner`) that passes the router to the bot **and** adds a branch to its
simulator comparison (the block where `--planner` calls `evaluate_planned`
with `settle=SETTLE_TICKS`), so that `summary.json`'s simulator numbers and
`outcome_agreement` refer to your router; without it they compare the bot
with the course-list simulator.

### 4.1 Implementation notes from the code

- **Leave `parkour/env.py` alone; wrap it.** A new module (for example
  `parkour/route_env.py`) owns a `ParkourEnv` and, every tick, does what
  `evaluate_planned` does: `obs = env.observe()`, overwrite the slot part
  `obs[:, 11:55]` with `env.obs_fn.slots(env.state, R, V, G)` and the
  progress input `obs[:, 254]` with your progress estimate, then
  `jumper.act(obs[:, :255], deterministic=True)` and `env.step`. `R` (rects,
  N x 4 x 5), `V`, `G` (valid and goal flags) and `P` stay on the GPU and
  change only at decisions. Load the jumper with `parkour.live.load_model`,
  run it under `torch.no_grad()` and never update its normaliser.
- **Landing memory.** `ParkourEnv` keeps its landing memory (`env.mem`) only
  with `EnvConfig(extended_obs=True)`, and then `observe()` returns 657
  numbers: the jumper takes the first 255.
- **Decision trigger** (`GreedyPlanner.update`): a row that is on the
  ground, with its feet within 1e-6 of a whole block height, standing on a
  perceived platform other than its current one (landing anywhere new, not
  only on the target), plus every row that `env.step` just reset
  (`info["done"]`; the reset happens inside `step`, so copy what you need
  first), which needs a start decision.
- **Candidates on the GPU.** Courses are static, so you may precompute each
  course's platforms once when a pool is built (`pool.plat[:pool.n]` holds
  the course platforms; the beams are cells in `pool.extra`) and filter per
  decision as `perceive`, the head-hitter rule and `options` do: the 25 x 25
  window around the player's block, rows 10 below to 4 above the feet,
  rects clipped to the window, height change -3 to +1, air gap up to
  `MAX_GAP[dy] + 0.5`, `path_clear`, visited and current excluded, the goal
  flag. Test the result against the planner's final candidate list (not
  `perceive`'s raw output, which includes beams) on the standard and the
  obstacle validation courses, from many positions. Courses swapped in by a
  pool refresh need their tables rebuilt and their rows' route state reset.
- **Never feed the course list.** `env.cp`, `pool.rem` and the order of
  `pool.plat` are privileged: sort candidates by a geometric key with fixed
  tie-breaks and never use a course index as a feature. Your training
  reward may use the true course.
- **Keep the jumper in its training distribution.** In training, empty
  slots followed slot 1 only when slot 1 was the goal (`is_goal` 1); a target
  followed by empty slots is a state it never saw. Fill slots 2 and 3
  (`planner.choose(..., first=target)` does it greedily in Python). Its
  jumps are the course's own jumps; how it handles longer shortcuts is
  unmeasured.
- **Encoding details.** `Observer.slots` encodes exactly 4 slots per call
  (`PLATFORM_SLOTS`); with `extended_obs` the progress input is still index
  254; `evaluate_planned` cannot evaluate extended-observation checkpoints,
  `train.evaluate` can.
- **Semi-MDP PPO.** One transition per decision per environment, with the
  environment's own reward summed between decisions (or your own), discount
  `gamma ** ticks` or per decision; success and fall are terminal, stall and
  the time limit count as failures, as in evaluation. Reuse `train.py`'s
  pieces: GAE, the dual clip, the KL stop and KL-adaptive learning rate,
  `RunningNorm` with a shadow copy, `train._save`.
- **Memory.** A `CoursePool` keeps a 64 x 32 x 64 block grid per course
  (128 KiB): 1 GiB for 8,192 courses, 225 MiB for a 1,800-course validation
  set.
- **In the bots** (`bridge/policy.js`): `Episode` takes the planner's JSON
  config (its `planner` option) and builds `new GreedyPlanner(inGrid, goal,
  cfg)` itself, with `isSolid` limited to the 64 x 32 x 64 grid; it then
  calls `start(pos)`, `update(pos, onGround, tick)`, `slots()` (a list of
  `{p: rect, goal}`), `progress` and `events`. Add an opt-in `router` option
  that builds your router with the same interface, and pass-through flags in
  `offline_bot.js`, `parkour_bot.js` and `parkour.live`. The bots' `Policy`
  accepts only the jumper's JSON format (`parkour-policy-v1`, 255 or 657
  inputs), so give the router its own export.
- **Parity tests to copy.** `tests/test_planner.py::PlannerParity` (24
  generated courses, half with obstacles: the Python evaluation with settle
  10 and the offline bot make identical plans, outcomes and tick counts),
  `tests/test_bridge_parity.py` (observation within 1e-5, logits within
  1e-3) and `tests/test_planner_service.py` for a service. Do the geometry in
  float64 with the planner's own operation order.

## 5. Alternatives (if hierarchical RL stalls)

- **Extend the end-to-end no-route model.** Fine-tune `results/noroute_policy.pt`
  (`--init-from` with the same `--extended-obs --hide platforms,progress`;
  the final policy has 255 inputs against 657 and cannot be loaded that way
  without padding its first layer and normaliser), or change its scan,
  memory or critic (a critic that sees the course would be new). It needs no
  planner at all, but so far it is the weakest option (85.4% / 70.9% on
  test).
- **A learned candidate scorer.** Keep the planner's perception and limits,
  and learn the score that ranks candidates: a contextual bandit (or a
  one-step PPO) whose reward is whether the jump policy reaches the course's
  goal after taking that candidate, with the greedy rule as a fallback. It is
  the smallest change, and it is still RL if trained from episode outcomes.

## 6. Already tried and ruled out

On the validation courses (`results/planner/validation_tuning.jsonl`, 52 rows):

- `cost=difficulty` (minus the log of the jump's forgiveness) with `align`
  from 0.5 to 4: at best 92.5% standard and 89.8% obstacles (100 courses
  per level), against 93.0% and 90.5% for `cost=distance, align=2.0` on the
  same courses.
- `value_pick` 2 to 6 (take the chain the policy's own critic values most;
  tried only with the earlier `cost=difficulty, align=3.0`): much worse on
  standard courses (84.7-87.7% against 92.5%), mixed on obstacles
  (87.7-90.3% against 89.0%, within the noise of 600 courses). A likely
  reason: the critic only ever saw the course's own platforms.
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
  limits, `slab_only`, `mark_passed` and `clear_path` were not varied in the
  52 tuning rows.
- Hiding the platform positions in training (the `noroute` arm of the reward
  ablation: the 98-point block scan, the goal and the progress input, no
  obstacles, 1,000 iterations): 82.4% on test, level 5 43.0% (mean of 3 seeds
  at their last checkpoint, `results/README.md`). `noroute-v2` (one seed,
  3,000 iterations, validation-selected, 30% obstacles, progress also hidden,
  the extended observation) reached 85.4% / 70.9%; the two differ in more
  than the observation.
- Nobody has built an RL router, a learned candidate scorer or a no-route
  model with a privileged critic yet (`noroute-v2` hides the inputs from the
  critic too).

## 7. Files to read first

1. `AGENTS.md` (rules for agents; section 10 below says which of them are
   for the training PC only) and `HANDOFF.md`: the team split at the top,
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
7. `parkour/live.py` (offline and real-Minecraft runs, `--planner`, the
   simulator comparison) and `parkour/neo_assist.py` (`--serve`: a Python
   service the bot calls).
8. `tests/test_planner.py`, `tests/test_planner_service.py`,
   `tests/test_bridge_parity.py`, `tests/test_heldout_fingerprints.py`.
9. `results/README.md` (sections "Without the course list" and "Learned
   no-route model") and `docs/design.md`.

## 8. Setup and commands

**Prerequisites:** Windows 10 or 11 (x64), Git, Python 3.12 with the `py`
launcher, an NVIDIA GPU of the Turing generation or newer (RTX 20xx / GTX
16xx and up: the torch build targets compute capability 7.5+) with a driver
for CUDA 12.8, and about 7 GB of disk inside the clone (the torch wheel
alone is about 3 GB; pip and pnpm also cache in the user profile). Check with
`py -3.12 --version` and `nvidia-smi`. If something is missing, ask the PC's
owner to install it; do not install system software yourself. `setup.cmd`
downloads from pypi.org and files.pythonhosted.org, download.pytorch.org,
nodejs.org, github.com (whose release files redirect to
*.githubusercontent.com), registry.npmjs.org and piston-data.mojang.com.

**Where:** a **short path outside OneDrive**, for example
`C:/dev/RL-Minecraft-project`. OneDrive locks files during writes, and
without Windows long-path support the Node archive's deepest file needs the
repository path under about 90 characters. You are probably reading this in
a clone already: then skip `git clone` and `cd` below, and check that
`git rev-parse --show-toplevel` prints such a path and `git status` shows
`main` with nothing changed. If the clone is in OneDrive or on a long path,
ask the owner to clone into `C:/dev/RL-Minecraft-project` and to start your
session in that folder.

**Shells:** the commands below use forward slashes and `./run.cmd`, which
work in PowerShell and in Git Bash (Claude Code's Bash tool on Windows); in
cmd.exe use backslashes and drop the `./`. `run.cmd` passes arguments
through cmd.exe, which splits values at commas, semicolons and `=`, and
quotes typed in PowerShell or Git Bash do not survive that: for any value
with one of those characters, or JSON, call the module directly
(`.venv/Scripts/python.exe -m parkour.<module> ...`): `./run.cmd offline ARGS`
runs `.venv/Scripts/python.exe -m parkour.live --offline ARGS` (without
`--offline` it plays in real Minecraft), `./run.cmd live ARGS` the same
without `--offline`, and `./run.cmd test` runs
`.venv/Scripts/python.exe -m pytest tests -q`. In Windows PowerShell
5.1 also escape JSON's inner quotes (`--planner-config '{\"align\": 3.0}'`),
or make a JSON-valued flag of your own take a file path.

**Long commands:** Claude Code's Bash and PowerShell tools stop a
foreground command after 2 minutes unless you give a longer timeout (10
minutes at most). These take longer: `./setup.cmd` (about 3 GB of
downloads), `./run.cmd test`, each planner baseline (four evaluations; one
planned evaluation of 1,800 courses took about 85 s on the CPU) and each bot
run (the planner's live runs played for 459 s and 524 s, plus the server
start and the simulator comparison). Run them as background tasks and read
their output when they finish.

```text
git clone https://github.com/taraskHy/RL-Minecraft-project.git C:/dev/RL-Minecraft-project
cd C:/dev/RL-Minecraft-project
git checkout -b routing/hrl
./setup.cmd
./run.cmd test
```

`setup.cmd` creates the Python 3.12 venv with torch 2.11.0 for CUDA 12.8,
then installs portable Node 24, Java 21, pnpm, the bot's packages and the
official Minecraft 1.21.4 server jar, all checksum-verified, into `.venv`,
`.local` and `bridge/node_modules`. `run.cmd test` runs every check (123 on
29 Oct, about 7 minutes); without `setup.cmd`'s portable Node about 20 of them are skipped
(physics and JavaScript parity), so read the summary line. Then check that
torch sees the GPU; this must print `True`:

```text
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, the planner and the bot runs quietly use the CPU while
the tests still pass, so ask the owner about the driver. If a test fails
on the untouched clone, rerun it alone once
(`.venv/Scripts/python.exe -m pytest tests/<file>::<test> -q`); if it still
fails, do not change `main`'s code to fix it: record the output in
`results/routing/README.md` and the pull request, and carry on unless it
involves the planner, the bridge or the simulator, in which case ask.

**Never run `scripts/training_queue.py` there**: it is the training PC's
launcher, reads that PC's plan (`configs/training_queue.json`, with
checkpoints that exist only there) and adds seeds by itself.

**While GPU work runs**, keep these going (each in its own window or as a
background process):

```text
.venv/Scripts/python.exe scripts/gpu_watchdog.py --hot 65 --cool 58 --every 1
.venv/Scripts/python.exe scripts/keep_awake.py --pattern parkour.rout --pattern parkour.planner --pattern parkour.train --pattern parkour.live
```

The watchdog suspends this project's GPU jobs above 65 C and resumes them
below 58 C (no state is lost). It recognises jobs by module name (`GPU_JOBS`
in `scripts/gpu_watchdog.py`) and already covers any module named
`parkour.route*`, `parkour.router*` or `parkour.routing*`: **name your GPU
modules that way** (for example `parkour.route_train`); any other name must
be added to `GPU_JOBS` with a test in `tests/test_gpu_watchdog.py`, and the
watchdog restarted. After starting it, check that `runs/gpu_watchdog.log`
shows `sample: NN C ...` lines about every 10 s. `monitor error (work held
after stale telemetry)` lines mean a sample failed (nvidia-smi or the
PowerShell process list): after 5 s without a good sample the watchdog keeps
the jobs it already knew about suspended until sampling works again (a
training seems to hang), and jobs started since run unprotected; if it never
logged a `sample` line, nothing is protected. Fix the cause and restart it.
Ctrl+C stops it and resumes the jobs it paused; if it is killed (a stopped
background task, a closed window, a restart) while it holds jobs, they stay
suspended: run `.venv/Scripts/python.exe scripts/gpu_watchdog.py --resume-all`
(once more if a job still hangs: suspensions add up). Keep-awake holds Windows awake
(no power setting changes) while a Python process whose command line
contains one of its patterns runs, and exits when none does, so start it
again after each job starts.

**Baselines on your machine** (each takes a few minutes). The section 2.1
rows were measured on the CPU; the physics is float64 and deterministic, but
the networks run in float32, so on CUDA a few courses may come out
differently. Your own planner run, on the same machine and device as your
router's evaluation, is the baseline you compare against. `parkour.planner`
and `parkour.live` have no device flag and use CUDA whenever it is
available; to run one on the CPU (for example while two trainings hold the
GPU), hide the GPU from that process with `CUDA_VISIBLE_DEVICES=-1`
(PowerShell: `$env:CUDA_VISIBLE_DEVICES='-1'` in that window first; Git
Bash: as a prefix of the command; an empty value does not hide it):

```text
.venv/Scripts/python.exe -m parkour.planner --family 2 --per-level 300 --record --out results/routing/planner_validation_standard.json
.venv/Scripts/python.exe -m parkour.planner --family 2 --obstacles --per-level 300 --record --out results/routing/planner_validation_obstacles.json
```

`--family 2 --obstacles` plays family 5. Careful: `--family 4 --obstacles`
also plays family 5, and only `--family 1 --obstacles` plays the obstacle
test family 4. The output holds `local` and `remaining` blocks (the two fall
rules), each with `given_route` (the course list) and `planner_route`; the
baseline is `local.planner_route.success`, the upper bound
`local.given_route.success`. `--record` adds, in each block, `per_course`
(the planner's outcome on every course: `success`, `fell`, `ticks`,
`progress`, `end_pos`, in `held_out_courses` order) and `plans` (its
re-plan events), about 9 MB per file. Section 9's paired comparison needs
these per-course outcomes, so keep `--record` for the planner's test runs
too (`--family 1 --per-level 200`, with and without `--obstacles`) and
write your router's per-course outcomes in the same order.

**Your router's code:** new modules (for example `parkour/route_policy.py`,
`parkour/route_train.py`, `parkour/route_eval.py`) with argparse CLIs like
`parkour/train.py`, so that one command trains
(`python -m parkour.route_train --run NAME ...`) and one evaluates a
checkpoint on a chosen family. Reuse `train.held_out_courses`,
`train.obstacle_courses` and `train._save`.

**Evaluation protocol** (as `planner.evaluate_planned`, so that your numbers
compare with the planner's): one episode per course (the first); the jumper
by argmax; the router by argmax with ties broken by the fixed candidate
order (sample only in training); `stall_ticks` 200, `max_ticks` 4000;
settle 0 on the held-out sets (10 only when comparing with the bots); both
fall rules, `local` as the headline. Write the planner CLI's JSON layout with
a `router_route` entry next to `given_route` and `planner_route`, including
`failure_modes` computed by `planner.failure_modes` from your router's
`[tick, here, chain]` events, and its `per_course` outcomes in the order
`--record` writes the planner's.

**Offline bot and real Minecraft** (after you add your flag to
`parkour.live`; section 9 says when to run them):

```text
./run.cmd offline --checkpoint results/final_policy.pt <your route flag> --courses-file results/games/v2-tuned-planner-live/courses.json --out runs/route-offline
./run.cmd offline --checkpoint results/final_policy.pt <your route flag> --courses-file results/games/v2-tuned-planner-live-obstacles/courses.json --out runs/route-offline-obstacles
./run.cmd live --checkpoint results/final_policy.pt <your route flag> --courses-file results/games/v2-tuned-planner-live/courses.json --out runs/route-live
./run.cmd live --checkpoint results/final_policy.pt <your route flag> --courses-file results/games/v2-tuned-planner-live-obstacles/courses.json --out runs/route-live-obstacles
```

Real Minecraft needs the owner's acceptance of the Minecraft EULA once,
which only he may give, after reading <https://aka.ms/MinecraftEULA>:
`./run.cmd accept-eula --agree`. The planner's own versions of these runs
used `--planner` (the current defaults; `--planner-config` takes JSON
overrides; `--planner` and `--assist` cannot be combined). `--viewer NAME`
lets a person watch in Minecraft Java 1.21.4 at `127.0.0.1:25570`. A `--out`
folder must not exist yet. Without `--courses-file`, `parkour.live` plays
test courses (the first `--per-level` of each level of family 1, or 4 with
`--obstacles`), so for bot checks during development write a validation
course file from `train.held_out_courses(..., family=2)` (or
`obstacle_prob=0.5, family=5`) as `{"courses": [c.to_json() for c in courses]}`.
Each run writes `summary.json` and `summary.md`. `summary.json` holds the
bot's `success`, `sim_success_same_courses`, `outcome_agreement` and
`replay_max_pos_err`, and for each course in `episodes` its `result`,
`sim_success` and `server_corrections` (null offline); nothing totals the
server corrections, so add them up. `summary.md` shows the same results
without the server corrections.

## 9. Done criteria

1. **Validation:** with the final policy as the jumper, your router completes
   more of the family-2 standard courses and more of the family-5 obstacle
   courses (300 per level, 1,800 each, fall rule `local`) than the greedy
   planner measured by you on the same machine and device. Report the paired
   difference on the same courses (the planner's per-course outcomes come
   from `--record`, section 8) with a 95% interval (a bootstrap over
   courses, or McNemar's test on the courses where exactly one of the two
   succeeds). Call it a win only if both sets improve and the interval of
   the combined difference excludes 0; otherwise write "no clear
   difference", which is still a valid result to report. (One standard error
   is about 0.6 points on 1,800 standard courses and 0.7 on obstacles; on the
   same 1,800-course sets the planner variants in
   `results/planner/validation_tuning.jsonl` came 0.2 to 3.6 points from the
   current default, the earlier default 1.5 and 1.6 points below it.)
2. **Test, once:** freeze the validation-chosen router, then play it and the
   greedy planner once each on families 1 and 4 (1,200 each, 200 per level),
   on the same device; report both, the paired interval, and the reference
   numbers (planner 92.33% / 90.50%, course list 98.42% / 95.25%). The first
   tested version is the headline; a later tested version is reported as
   additional, never chosen because of its test result.
3. **Offline bot**, only for the frozen, tested router: each course file of
   section 2.3 once. It passes if the outcome matches the simulator (settle
   10) on every course. Report the counts next to the planner's 51/60 and
   56/60 and the course list's 59/60 and 57/60; sixty courses are too few to
   decide "better" (one standard deviation is about 2 courses), so that claim
   rests on criteria 1 and 2. Never tune on these courses; debug on
   validation course files.
4. **Real Minecraft**, the same courses, once: the same outcome as the
   simulator on every course and 0 server corrections; counts reported as in
   criterion 3.
5. `run.cmd test` passes, including your new tests (parity between the
   Python and JavaScript sides of your router, or of the service protocol).
6. `results/routing/README.md` and its files hold every number, each saying
   where it was measured, and a pull request into `main` is open.

## 10. Rules

**Git and GitHub.**

- Work on your own branch `routing/<name>`; push only that branch; never
  force-push or rewrite history. If `git config user.email` is empty, set a
  repository-local identity (`git config user.name "Claude (routing agent)"`
  and the owner's GitHub noreply address, or ask him for it).
- Push with `git push -u origin routing/<name>` after every result. Open a
  draft pull request early, after your first commit, if the GitHub CLI is
  installed and logged in (gh cannot prompt in Claude Code's tools, so give
  it `--title` and `--body`):

  ```text
  gh pr create --draft --base main --head routing/<name> --title "RL routing (draft)" --body "Work in progress; results in results/routing/README.md"
  ```

  Later, `gh pr edit routing/<name> --body-file <file>` updates the
  description and `gh pr ready routing/<name>` marks it ready. Otherwise
  give the owner
  `https://github.com/taraskHy/RL-Minecraft-project/compare/main...routing/<name>?expand=1`.
  If you cannot push to the repository, push to a fork and open the pull
  request from there.
- To pick up `main`, `git fetch origin` then `git merge origin/main` (never
  rebase), then run the tests. While two GPU jobs run, run the tests on the
  CPU so that they do not become a third GPU process:
  `CUDA_VISIBLE_DEVICES=-1 ./run.cmd test` in Git Bash, or
  `$env:CUDA_VISIBLE_DEVICES='-1'; ./run.cmd test` in PowerShell (the CUDA
  physics test is then skipped). The pull request's description holds the
  results table (validation, test, offline, live, each with its file), how
  to reproduce, and what is still open; mark it ready when section 9 is
  met, or when you stop.

**What you may change.**

- Add, opt-in and additive only: new `parkour/rout*` modules and tests; new
  `bridge/rout*.js` files (the JavaScript router, or the spawner of a router
  service, like `bridge/assist.js`); a `router` option in `bridge/policy.js`'s
  `Episode` next to `planner`;
  pass-through flags in `bridge/offline_bot.js`, `bridge/parkour_bot.js` and
  `parkour/live.py` (with the simulator branch of section 4); new helpers in
  `parkour/planner.py` and `bridge/planner.js` that leave `perceive`,
  `options`, `choose` and `GreedyPlanner` unchanged. With the options off,
  behaviour must be identical and every existing test must pass (among them
  `tests/test_bridge_parity.py`, `test_planner.py`, `test_extended_obs.py`,
  `test_physics_parity.py` and `test_heldout_fingerprints.py`).
- Add no Python or npm packages: `requirements.txt` and
  `bridge/package.json` are pinned and shared with the training PC; numpy is
  enough for a bootstrap or McNemar's test.
- Never change: `parkour/physics.py`, `bridge/physics_patch.js`, the
  environment's reward and termination code (`RewardConfig` and the reward
  code in `parkour/env.py`), the 255-number observation, the
  `parkour-policy-v1` format, default course generation and the seed
  families, `results/final_policy.*`, `results/runs/`, `results/games/`,
  `results/README.md`, or anything else in HANDOFF.md section 5.

**Evaluation discipline.** Tune and choose only on validation (families 2
and 5). Test families 1 and 4 are for the final numbers only. Do not use
seed families already taken (0 training, 1 test, 2 validation, 3 stress, 4/5
obstacle test/validation, 7 the neo DAgger courses, 11 analysis, 12-17 other
sets); new sets get new families from 18 to 39 (40 and up are the main
session's: 40 and 41 hold its single-neo sets). Every number says where it was
measured (simulator, offline bot on prismarine-physics without a server, or
real Minecraft) and comes from a file you commit under `results/routing/`.

**Results.**

- `results/routing/` holds a short `README.md` (what was tried, what was
  chosen and why, every number with its file, and your session log at the
  top: date, who you are, what changed, what is next) and one file per
  evaluation. Name files `planner_*.json`, `route_*` or `router_*` (any
  extension): `results/DATA.md` already lists those patterns, so
  `tests/test_handoff.py` keeps passing. A file named otherwise needs its own
  entry in the `results/routing/` row of `results/DATA.md`; if
  `tests/test_handoff.py` fails, that is the reason. Avoid `*.log` names
  (Git ignores them).
- Bot runs write to `runs/`, which Git ignores: copy their `summary.json`,
  `summary.md` and `courses.json` into `results/routing/` as
  `route_<run>_summary.json`, `route_<run>_summary.md` and
  `route_<run>_courses.json` (and `episodes.jsonl` gzipped as
  `route_<run>_episodes.jsonl.gz` for replays).
- Copy each router training run's small files (config, train log,
  validation log, summary; never checkpoints) to
  `results/routing/runs/<run>/`; subfolders of `results/routing/` need no
  entry. Never write into `results/runs/` or `results/games/`, which feed the
  handoff snapshot.
- Ship the chosen router's weights as `results/routing/router_policy.pt`
  (and its JSON export as `router_policy.json`, if the bot needs one), with a
  `router_policy_provenance.json` like `results/noroute_policy_provenance.json`.

**Training-PC steps that are not yours.** Do not run `python -m
parkour.report` (`run.cmd report`) or `python -m parkour.handoff` on your
PC; the read-only `python -m parkour.handoff --check` is fine: it exits 1
and asks for `HANDOFF.md`'s hand-written parts to be updated whenever there
are commits after HANDOFF.md last changed, your own included, which is
expected; leave `HANDOFF.md` alone. The report
rebuilds `results/README.md` from the training PC's `runs/`; if you run it
by mistake it stops and suggests `parkour.handoff` or `--force`: do neither.
AGENTS.md's "after each result: report" step and its "Keep the handoff
current" section are for the main session, which refreshes the handoff
snapshot and adds your summary to `HANDOFF.md` when it merges your pull
request. Leave `HANDOFF.md` and `TEAM_STATUS.md` to it; your status lives in
`results/routing/README.md` and the pull request.

**GPU.** At most two GPU processes at once, evaluations included (two
trainings, or one training and one evaluation), and fewer if they do not
fit: the training PC's 16 GB card fits two, and a third spills into system
RAM and makes everything 5 to 10 times slower. Windows spills instead of
failing, so when a job starts check both the card's memory and the shared
GPU memory (the figure Task Manager shows). The second command prints bytes
per adapter from either shell; the NVIDIA card is the one with a large
`dedicated usage`, and a rise of gigabytes in its `shared usage` means the
job spilled:

```text
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
powershell -NoProfile -Command "(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage').CounterSamples | Select-Object Path,CookedValue"
```

Size `--num-envs` to the card. Unless the owner says otherwise, full GPU use within the
temperature rule is fine; if he plays games on the PC, start the watchdog
with `--pause-while <game.exe>` as well.

**Parity.** Whatever the bot runs must match the simulator (the physics port
matches prismarine-physics to 1.4e-14 blocks; keep that kind of check for
your code).

## 11. Questions and the time box

Ethan and the teammate want as few questions as possible. Decide technical
matters yourself and record the reasons in `results/routing/README.md`.
At the start, ask the PC's owner once for the deadline (the presentation
date) and how long the PC is yours, and plan backwards from it: first day,
setup, `run.cmd test`, the two planner baselines and a draft pull request;
then validation results for the hierarchical router; leave time to freeze
the best router, run its one test, the offline and live runs, and write the
README. If no router beats the planner on validation in time, still freeze
the best one, test it once and report honestly. Otherwise ask only for what
only the owner can do (accepting the Minecraft EULA, GitHub access,
installing a missing prerequisite) or about direction if the done criteria
turn out to be wrong or unreachable, with the evidence.

## 12. What a fresh clone has and lacks

In Git: the code, configs, tests, `results/` (every run's logs in
`results/runs/`, the bot games in `results/games/`, the planner results in
`results/planner/`), and these policies: `results/final_policy.pt` (the
jumper you use), `results/final_policy_v1.pt`, `results/neo_policy.pt`,
`results/noroute_policy.pt`.

Not in Git (they exist only on the training PC, and you do not need them):
`runs/` (training checkpoints and full logs; every run's small logs are in
`results/runs/NAME/`) and `.local/` (Node, Java, the server jar, the local
server folder and the EULA consent). `setup.cmd` recreates the runtimes and
the server jar; the server folder appears on the first `run.cmd live`; the
consent only the PC's owner can give. To retrain a run, read its settings
from `results/runs/NAME/config.json` and pass the values that differ from
the `TrainConfig` defaults to `python -m parkour.train --run NAME`;
`HANDOFF.md` section 6 lists the local-only checkpoints, how to regenerate
each and how long it took. Some configs and scripts point into `runs/` and
fail on a fresh clone (`configs/training_queue.json`,
`configs/sweep_extra.json`, `scripts/neo_evidence.py`); routing needs none
of them.
