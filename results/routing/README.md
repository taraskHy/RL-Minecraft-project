# RL routing: a learned route policy instead of the greedy planner

Mission: [docs/MISSION_ROUTING.md](../../docs/MISSION_ROUTING.md). Branch `routing/hrl`.

## Headline

The frozen router (`router_policy.pt`, run `route-full-f` iteration 29, chosen on validation by the
rule below) with the unchanged final policy as the jumper, against the greedy planner on the same
courses. Paired difference with a 95% bootstrap interval over courses and McNemar's exact test.

| Where | Courses | Router | Greedy planner | Course list given | Router minus planner | File |
| --- | --- | --- | --- | --- | --- | --- |
| Validation, simulator | family 2, 1,800 | **97.89%** | 92.78% | 97.89% | **+5.1 [+3.9, +6.3]**, p = 3e-18 | `router_validation_route-full-f_it29_*.json` |
| Validation, simulator | family 5 (obstacles), 1,800 | **95.56%** | 89.00% | 95.56% | **+6.6 [+5.2, +7.9]**, p = 3e-21 | same |
| **Test (once), simulator** | family 1, 1,200 | **97.83%** | 92.33% | 98.42% | **+5.5 [+4.0, +7.0]**, p = 1e-12 | `router_test_*.json` |
| **Test (once), simulator** | family 4 (obstacles), 1,200 | **95.00%** | 90.50% | 95.25% | **+4.5 [+2.9, +6.1]**, p = 4e-8 | same |
| Test, both sets | 2,400 | 96.42% | 91.42% | 96.83% | **+5.0 [+3.9, +6.1]**, p = 3e-19 | `router_test_paired.json` |
| Offline bot (prismarine-physics) | 60 standard test courses | **57/60** | 51/60 | 59/60 | outcome = simulator on 60/60 | `route_offline_summary.json` |
| Offline bot | 60 obstacle test courses | **56/60** | 56/60 | 57/60 | outcome = simulator on 60/60 | `route_offline_obstacles_summary.json` |
| **Real Minecraft 1.21.4** | 60 standard test courses | **57/60** | 51/60 | 59/60 | outcome = simulator on 60/60, 0 server corrections | `route_live_summary.json` |
| **Real Minecraft 1.21.4** | 60 obstacle test courses | **56/60** | 56/60 | 57/60 | outcome = simulator on 60/60, 0 server corrections | `route_live_obstacles_summary.json` |

Fall rule `local` (the bots' rule); the `remaining` rule is in the same files (test: router 98.00% /
95.17%, planner 94.67% / 92.42%). The planner's test numbers on this PC equal the reference
(`planner_test_*.json`, recorded). Sixty bot courses are too few to call a winner (one standard
deviation is about 2 courses); the claim rests on the 1,800- and 1,200-course sets.

What it shows: the router closes about 90% of the gap between the greedy planner and the course
list on test (5.0 of 5.4 points), so on these courses the bot needs only the goal's position.
Reinforcement learning alone (pure PPO from a random router, run E) already beats the planner by
+3.8 / +4.4 points on validation; adding imitation of the course's next platform inside the RL loop
(a training signal only, never an input) brings it to the course list's level.

## Reproduce

```text
# the frozen router's recipe (run route-full-f; about 20 s per iteration on an RTX 4060 Ti, 16,384 games)
.venv/Scripts/python.exe -m parkour.route_train --run route-full-f --iterations 250 --num-envs 16384 --rollout-ticks 64 --pool-size 4096 --minibatches 8 --bc-coef 1.0 --bc-decay --bc-until 100 --bc-floor 0.3 --fill policy --fill-bc 1.0 --ent-coef 0.001 --lr 0.0003 --eval-every 15 --val-per-level 100 --duty 0.75 --seed 4
# evaluation against the planner (writes <out>_standard.json, <out>_obstacles.json, <out>_paired.json)
.venv/Scripts/python.exe -m parkour.route_eval --router results/routing/router_policy.pt --set validation --out results/routing/router_validation_X
# the bots (offline; drop --offline for real Minecraft)
.venv/Scripts/python.exe -m parkour.live --offline --checkpoint results/final_policy.pt --router results/routing/router_policy.pt --courses-file results/games/v2-tuned-planner-live/courses.json --out runs/route-offline
```

Each run's configuration is in `runs/<run>/config.json`. Tests: `tests/test_route.py` (the GPU
perception against `planner.perceive`/`options` on validation courses; the offline bot against the
Python evaluation, same plans, outcomes and ticks, for a random router with both fill modes and for
the shipped router).

## Open points

- The remaining failures are mostly the jumper's own (level 5, and obstacle courses): on test the
  router is 0.6 points below the course list on standard courses and 0.25 on obstacle courses.
- 7 validation and 5 test obstacle courses fail with an empty plan: the head-hitter rule drops a
  real course platform 4 blocks above a beam it touches (section 3 of the brief); the router uses the
  planner's perception unchanged, so it inherits this.
- PPO's own contribution after the imitation warm start is not measurable (run D); pure PPO from
  scratch (run E) beats the planner but is less stable and 1-2 points below.

## Session log

- **29 Oct 2026, Claude (Opus 5.5) on the teammate's PC (RTX 4060 Ti 16 GB, i9-10900K):** fresh
  clone, `setup.cmd`, all tests pass on the CPU (116 + the physics parity checks). Built the GPU
  perception, the route policy, the PPO trainer, the bot port and their tests; measured the
  baselines and oracle ceilings; first training runs (below). GPU rules on this PC: the watchdog
  (`--hot 65 --cool 58 --every 1`) and keep-awake run whenever GPU work runs; one training process
  at a time; the trainer paces itself to 75% of wall time (`--duty 0.75`).

- **29 Oct 2026, evening (same session, after the brief's update `948a365`/`2fb95b2`):** merged
  `main`; repository-local Git identity; draft pull request #2. Re-ran the planner baselines with
  `--record` (same 92.78% / 89.00%). `parkour.route_eval` now writes the planner CLI's layout
  (`given_route`, `planner_route`, `router_route` with `failure_modes`, both fall rules, every
  course) and the paired comparison. Owner: a few days are available. Notes against the updated
  brief: Python 3.12 was installed by me (`winget`, user scope) before the brief said to ask the
  owner; the offline bot was run once with an early router (run A) before the brief restricted bot
  runs to the frozen, tested router: it is reported below as an early deployment check and was not
  used for any choice.

- **30 Oct 2026, night (same session):** controls C (PPO without imitation), D (imitation without
  PPO), E (pure PPO from scratch) and F (the whole recipe in one run, new seed). Applied the freeze
  rule (`route-full-f` it 29), shipped `router_policy.*`, then played the test sets once (router and
  planner, same device), the offline bot once per course file and real Minecraft once per course
  file. Next for the main session: review and merge the pull request.

Early deployment check (not part of the protocol, run A it 99, before the brief's update): offline
bot 57/60 and 55/60, outcome = simulator on all 120 (`route_a_offline_*_summary.json`).

## Freeze rule (written before the candidates' full validation results were compared)

The router to test is the checkpoint with the highest mean of the two full validation sets
(families 2 and 5, 1,800 courses each, fall rule `local`) among those that beat the greedy planner
on both sets; if two are within 0.1 points, the earlier (simpler) one. Training runs' own screening
uses 100 courses per level; candidates are re-evaluated on the full sets before the rule is applied.
The test sets are then played once with that checkpoint and the greedy planner.

## Validation results (simulator, this PC's GPU)

Full validation sets, 1,800 courses each, the protocol of `planner.evaluate_planned` (section 8 of
the brief). Router minus greedy planner on the same courses, 95% bootstrap interval over courses,
McNemar's exact test on the courses only one of the two completes.

| Router | Standard (family 2) | Obstacles (family 5) | Paired difference, standard | obstacles | both sets | File |
| --- | --- | --- | --- | --- | --- | --- |
| greedy planner | 92.78% | 89.00% | | | | `planner_validation_*.json` |
| course list given (upper bound) | 97.89% | 95.56% | | | | same files |
| `route-bc-a` it 99 (greedy fill) | 97.33% | 94.50% | | | | `router_validation_route-bc-a_it99.json` (router only) |
| **`route-fill-b` it 134** | **97.56%** | **95.56%** | **+4.8 [+3.6, +6.0]**, 106 vs 20 courses, p = 2e-15 | **+6.6 [+5.2, +7.9]**, 142 vs 24, p = 1e-21 | **+5.7 [+4.8, +6.6]**, p = 1e-35 | `router_validation_route-fill-b_it134_*.json` |

Fall rule `remaining` (drops onto a later, lower platform count as landings): router 97.72% / 95.61%,
planner 95.17% / 90.89%, paired +2.6 [+1.6, +3.6] and +4.7 [+3.4, +6.0] (same files).

Failure modes (`planner.failure_modes`, local rule): the planner's 112 standard and 145 obstacle
*wrong platform* failures fall to 13 and 14; the router's remaining failures are mostly *jump*
failures of the jumper (31 standard, 58 obstacles; 8 obstacle courses have an empty plan, as with
the planner: a course platform 4 blocks above a beam it touches is dropped by the head-hitter
rule, section 3 of the brief). The router equals the course list on the obstacle set and is 0.3
points below it on the standard set.

Other `route-fill-b` checkpoints on the full sets (router only, local): it 89 97.39% / 95.67%,
it 119 97.50% / 95.61%, it 149 97.17% / 95.56%.

## The frozen router

Every checkpoint evaluated on the full validation sets (local rule, simulator, router only unless
noted; mean of the two sets):

| Checkpoint | Standard | Obstacles | Mean |
| --- | --- | --- | --- |
| **`route-full-f` it 29** | **97.89%** | **95.56%** | **96.73%** |
| `route-fill-b` it 134 | 97.56% | 95.56% | 96.56% |
| `route-fill-b` it 119 | 97.50% | 95.61% | 96.56% |
| `route-full-f` it 104 | 97.28% | 95.78% | 96.53% |
| `route-fill-b` it 89 | 97.39% | 95.67% | 96.53% |
| `route-imit-d` it 104 | 97.28% | 95.50% | 96.39% |
| `route-imit-d` it 149 | 97.39% | 95.33% | 96.36% |
| `route-full-f` it 119 | 97.17% | 95.56% | 96.36% |
| `route-imit-d` it 89 / 119 | 97.28% | 95.44% | 96.36% |
| `route-fill-b` it 149 | 97.17% | 95.56% | 96.36% |
| `route-full-f` it 89 | 97.11% | 95.50% | 96.31% |
| `route-bc-a` it 99 | 97.33% | 94.50% | 95.92% |
| `route-ppo-e` it 39 | 96.61% | 93.39% | 95.00% |

The freeze rule takes **`route-full-f` iteration 29** (no other checkpoint within 0.1 points). It is
shipped as `router_policy.pt` (weights, network sizes, training config), `router_policy.json` (the
bot's copy) and `router_policy_provenance.json` (source and hashes). Its slots 2 and 3 are filled by
the router itself (`fill: policy`). Being the maximum of 13 checkpoints, its validation score is
slightly optimistic; the test below is the unbiased number.

Its paired validation against the greedy planner (`router_validation_route-full-f_it29_*.json`,
local rule): standard +5.1 points [+3.9, +6.3] (107 courses only the router completes, 15 only
the planner, McNemar p = 3e-18); obstacles +6.6 [+5.2, +7.9] (143 against 25, p = 3e-21); both sets
+5.8 [+4.9, +6.8] (p = 3e-38). Fall rule `remaining`: +2.9 [+1.9, +3.9], +4.7 [+3.4, +5.9], +3.8
[+3.0, +4.7]. Failures (local): wrong platform 13 and 15 (planner 112 and 145), nothing in view 0 and 7,
jump 25 and 58.

## Training runs

| Run | Start | Recipe | Logs |
| --- | --- | --- | --- |
| `route-bc-a` | scratch | PPO + imitation of the course's next platform (weight 1.0, falling to 0.03 by iteration 150), entropy 0.01, greedy fill; stopped at 161 | `runs/route-bc-a/` |
| `route-fill-b` | `route-bc-a` it 99 | slots 2-3 by the route policy (fill imitation 1.0), slot-1 imitation 0.3, PPO, entropy 0.001, lr 1e-4, 150 iterations | `runs/route-fill-b/` |
| `route-ppo-c` | `route-fill-b` it 134 | as B with slot-1 imitation 0 (slot 1 by PPO only): does RL alone hold or improve the router? Stopped at 60 | `runs/route-ppo-c/` |
| `route-imit-d` | `route-bc-a` it 99 | as B (same seed) with the PPO policy loss off (`--pg-coef 0`, imitation only): what PPO adds | `runs/route-imit-d/` |
| `route-ppo-e` | scratch | pure PPO: no imitation at all, greedy fill, entropy 0.003, lr 3e-4; stopped at 120 of 300 (validation peaked at 39 and declined) | `runs/route-ppo-e/` |
| `route-full-f` | scratch | the whole recipe in one run, new seed: fill by the policy (imitation 1.0), slot-1 imitation 1.0 falling to 0.3 by 100, entropy 0.001, lr 3e-4; stopped at 120 of 250 (flat at 97.6-97.8% on the screen since iteration 14) | `runs/route-full-f/` |

**Run E, result: RL alone beats the greedy planner.** Pure PPO from a random router, without any
imitation, reached 96.61% / 93.39% on the full validation sets at its best checkpoint (it 39),
against the planner's 92.78% / 89.00%: paired +3.8 points [+2.8, +4.9] standard and +4.4
[+3.3, +5.6] obstacles, +4.1 [+3.3, +4.9] together (179 courses only the router completes, 31 only
the planner; McNemar p = 2e-26; `router_validation_route-ppo-e_it39_*.json`). It learned its own
route style (on multi-option decisions it picks the course's next platform 79-82% of the time,
against 99% for the imitation runs), was less stable (a dip to 90% rollout success at iteration
70; the KL stop often ended updates after a few minibatches) and declined after iteration 39 (95.3%
/ 94.0% at 119 on the 100-per-level screen). Imitation of the course's next platform inside the RL
loop adds about 1 point standard and 2 points obstacles on top (runs A, B, D).

**Run D, result:** imitation only is as good as B within the noise. Full validation, local rule:
D it 89 97.28% / 95.44%, it 104 97.28% / 95.50%, it 119 97.28% / 95.44%, it 149 97.39% / 95.33%,
against B's 97.39-97.56% / 95.56-95.67%. Paired, B it 134 minus D it 104: +0.2 points
[-0.1, +0.5] over both sets (18 courses only B completes, 12 only D, McNemar p = 0.36); against
D it 149: +0.2 [-0.1, +0.5], p = 0.23 (`route_paired_fill-b-it134_vs_imit-d-*_validation.json`).
No clear difference: after the warm start, the router's quality comes from on-policy imitation of
the course's next platform (with the learner's own states, as in DAgger), and PPO's policy loss adds
at most a few tenths of a point. Run C (PPO without the imitation) drifted down.

**Run C, result:** PPO alone did not improve the router. Its rollouts' success (sampled actions,
training courses) stayed where run B ended (97.0% over iterations 40-60 against 96.9% over B's last
30 iterations), its agreement with the course's next platform fell from 0.98 to 0.94, and
validation (100 courses per level per set, deterministic) fell from B's 97.83% mean to
96.5-97.3% (iterations 14 to 59). From a router already at the course list's level, the
remaining improvements are below what PPO's noisy, mostly forced decisions can find.

Observations so far: when the imitation weight decayed in run A, the policy's entropy rose (0.003 to
0.022), its agreement with the course's next platform fell (0.997 to 0.94) and validation slipped;
with entropy 0.01 and mostly forced decisions, PPO's signal is weak. Run B used a 10 times smaller
entropy bonus and kept some imitation.

## Method (hierarchical RL)

The frozen final policy (`results/final_policy.pt`, deterministic) jumps; a route policy chooses
the platform it jumps to. Code: `parkour/route_env.py` (perception and decisions on the GPU),
`parkour/route_policy.py` (network), `parkour/route_train.py` (training), `parkour/route_eval.py`
(evaluation), `bridge/router.js` (the bots), `parkour.live --router` (offline and real games).

- **Perception = the planner's own.** Every course's standable platforms (the course's and any
  obstacle top the planner would also take for a platform) are found once per course; each decision
  filters them by the planner's 25 x 25 window (10 rows below to 4 above the feet), clips them,
  drops head hitters and applies the planner's jump limits, path check, visited and passed marks.
  Proved equal to `planner.perceive`/`options` on 7,778 validation views (`tests/test_route.py`),
  and with the greedy rule plugged in, `RouteRunner` reproduces `planner.evaluate_planned` plan
  for plan, with the same outcome and tick count on every course (checked on 96 courses on the GPU
  and 48 on the CPU, `runs/scratch/chk_greedy.py`).
- **Decisions** when the planner re-plans: at the start and after every landing on a new platform.
- **Observation:** one token per platform in view, sorted by a geometric key (never the course's
  order): position, corners, size, gap and height change in a frame at the platform planned from
  turned towards the goal; the greedy planner's cost, rank and choice; distance to the goal; goal
  flag; whether it reaches the window's edge; jumps on from it; whether a block hangs over the
  flight path at head height; role flags (selectable, planned from, stood on, visited, in the
  chain); and global numbers (goal distance and height, plan depth, counts, progress, the player's
  position and velocity at the decision, ticks since the last new platform and since the last
  decision).
- **Network:** a set transformer (2 layers, width 128, 4 heads, no positional encoding) scores the
  selectable tokens (pointer network); a separate encoder pools the tokens into the value.
- **Action:** the next platform (slot 1). Slots 2 and 3: the greedy rule from it (`--fill greedy`)
  or the route policy's own argmax from it (`--fill policy`).
- **Training:** PPO, one transition per decision (semi-MDP): reward = the environment's own reward
  summed over the ticks to the next decision (discount 0.997 per tick), with its time cost added
  back and -5 when the episode ends by the stall or time limit; GAE over decisions (lambda 0.95).
  An imitation loss pulls the policy towards the course's next platform (the selectable platform
  with the smallest course index above the player's progress): the course is used for the reward
  and this training signal only, never as an input. Training courses: generator family 0, levels
  0-5, obstacles on 30% of jumps, 5 to 12 jumps; 16,384 parallel games.
- **Selection:** validation families 2 and 5 only; the test families are run once at the end.

## Baselines and ceilings (validation, simulator, this PC's GPU)

| Router | Standard (family 2, 1,800) | Obstacles (family 5, 1,800) | File |
| --- | --- | --- | --- |
| greedy planner (`parkour.planner`, current default) | 92.78% | 89.00% | `planner_validation_standard.json`, `planner_validation_obstacles.json` |
| course list given to the jumper | 97.89% | 95.56% | same files, `given_route` |
| oracle: the course's next platform, greedy slots 2-3 (diagnostic) | 97.50% | 94.83% | `route_oracle_diagnostic_validation.json` |
| oracle: the course's next platform, oracle slots 2-3 (diagnostic) | 98.11% | 96.17% | same file |

The greedy numbers equal the mission's reference (section 2.1) exactly.
