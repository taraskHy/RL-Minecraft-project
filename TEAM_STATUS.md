# Team status

One page for everyone on the team: what works, what is running, what is open.
Updated and pushed regularly; the newest results can reach
[results/README.md](results/README.md) first. The main results are in that
README (generated from the run files); the rest come from the files in
`runs/`, `results/` and `configs/` (one GPU note comes from
`docs/overnight_plan.md`). [results/DATA.md](results/DATA.md) lists most of
the data: training runs, policies, the route planner and neo files, the game
folders (including the v2, planner, noroute, neo and film games) and the
videos.

**Last updated:** 29 Oct 2026, 05:05, training PC: active-run validation refreshed; initial advanced/ice comparison already delivered.

## The project

**A bot that learns Minecraft parkour and completes courses it has never
seen.** We changed direction to this on 27 October 2026; the survival work
(wooden pickaxe, stone design) is archived on the `archive/survival` tag and
the `codex/*` branches.

How it works, in one paragraph: a Mineflayer bot moves with a physics library
called prismarine-physics. We ported that library to the GPU (it matches the
real one to 1.4e-14 blocks over 99,000 ticks, `results/physics_parity.json`),
generate endless random parkour courses at six difficulty levels, and train a
neural network with PPO on 16,384 parallel games that share a pool of 8,192
courses (about 170,000 game ticks per second, network updates included). The
trained network then drives the Mineflayer bot on a real Minecraft server.
Details: `docs/design.md`.

## Results

**Final policy since 28 October 10:41:** `obstacles-g995`, iteration 699:
the main recipe with discount 0.995 (`main-g995`, 3,000 iterations) plus the
1,000-iteration obstacle fine-tune. It beat the first final policy
(`obstacles-ft`, discount 0.99) on the validation courses, 97.6% to 96.6%.

**Real Minecraft (1.21.4 server), courses never seen in training:**

| Courses | Final policy | First policy | Same as the simulator? |
| --- | --- | --- | --- |
| standard, up to 10 jumps, all levels | **59 / 60** | 59 / 60 | yes, every course and tick count |
| with obstacles (head-hitter beams, decoy pillars) | **57 / 60** | 58 / 60 | yes |
| hand-designed (tour, stairs, sprint, neo) | **3 / 4** | 3 / 4 | yes |

Both policies fail the neo course; neither trained on neos. The offline bot
(prismarine-physics, no server) also agrees with the simulator on every
course. The server corrected the bot's position 0 times in 124 episodes per
policy; replaying the recorded moves in the simulator reproduces every real
trajectory to 2.1e-14 blocks. On these 60-course samples the two policies are
within noise; the 1,200-course simulator test sets show the new one ahead.

**Simulator, 1,200 held-out courses per test set:**

- Final policy: **98.4%** of standard courses (hardest level 93.5%) and
  **95.2%** with obstacles (hardest level 81.0%). First policy: 97.1%
  (90.5%) and 93.6% (75.5%).
- Training from scratch (first main run `main-s0`, 2.6 hours): success in
  training episodes goes from 0% (mean reward -5.4) to 95.4% (mean reward
  +53), and falls drop from 100% to about 5%; the last checkpoint completes
  95.5% of the held-out test courses. Every iteration:
  `results/figures/training_main-s0.png`.
- Every jump type of levels 0-5 is made from a standing start, all but one at
  all 16 tested angles (that one at 14 of 16); hardest level 99.1% (first
  policy 99.6%). Of the 49 jump types no scripted strategy makes, the bot
  makes 3 at some angles (first policy 4).
- Course styles outside training (100 courses per level each): 15-jump
  courses 96.5%, a turn at every jump 94.2%, any start direction 96.7%,
  single blocks 93.0%.

**Hyperparameters** (9 settings, one at a time, 0.52 B ticks each; report
7.3): only discount 0.995 beat the defaults by more than 3 points on
validation, in all three seeds (95.2%, 95.8% and 95.2% against 91.8%).

**Our reward ideas, tested (3 seeds each, `results/figures/ablation_curves.png`):**

- *Height / fall penalty*: matters most. Without it the hardest level drops to
  54-57% from 69-77%, in every seed.
- *Distance to the goal* (progress reward): learning about 3 times faster;
  same final score.
- *Number of jumps* (landing bonus) and the *curriculum*: no reliable effect
  at this budget.
- Not told where the platforms are (only the blocks it sees and the goal
  direction): 82% instead of 93%.
- *Hyperparameters* (one at a time, chosen on validation): only the
  discount 0.995 beats the defaults by more than 3 points (95.4% against
  91.8% on validation, 3 seeds); it is in the final policy now.

**Problems we found and fixed** (good presentation material):

- PPO collapsed three times in the first runs (success 75-80% to about 15%
  within two updates): fixed with a KL stop, dual clipping and a KL-adaptive
  learning rate (`results/figures/training__v1_unstable_main-s0.png`).
- Minecraft centres whole-number `/tp` coordinates on the block, which broke
  the first live run: the bot now sends decimals.
- The double-neo search started the player inside the wall (Minecraft lets
  you walk out of a block you overlap), so its first landings were invalid;
  starts inside blocks are now rejected and tested.

## Harder jumps (neo, double neo, 4-block)

- **Single neos are possible** but tight: a tick-by-tick search lands 16 of 18
  layouts, and only 1 in 2,000 to 1 in 16,000 sampled action sequences land
  one. The one-block-wide takeoff our generator puts before a neo is our
  simplification; parkour has no such rule. From a 2x2 takeoff the
  demonstration search (below) lands neos when the landing is next to the
  wall's open end, never when it is two lanes away (simulator,
  `results/neo/wide_takeoff_search.json`). `courses._place(wide_neo=True)`
  allows wide takeoffs in hand-designed and DAgger training courses; the
  default generator and every test set are unchanged.
- **No double or triple neo was found** (0 of 64 layouts: 36 from standing,
  28 with a 3-4 block run-up). We treat them as impossible; this is not a
  proof.
- **4-block jumps with a run-up:** the level 0-8 policy (`advanced-ft3`)
  makes some that no scripted strategy makes (12-38% of start angles).
- **Refusal:** the level 0-8 policy (`advanced-ft3`) made none of the 13
  single-neo types (standing start, 16 angles, simulator,
  `results/neo/jump_eval_advanced-ft3.json`). On 11 it stood still until the
  stall limit every time, because standing still cost less reward than a
  likely fall (-5). A stall penalty (`runs/neo-stall`) still gave 0 of 13:
  random tries almost never find the curved run a neo needs. A good
  presentation point about reward design.
- **Demonstrations:** a cross-entropy search in the bot's own action space
  (`parkour/neo_demos.py`) collected 50,706 verified landing sequences on all
  16 landable layouts, both wall sides; training used 12,800. Imitation plus
  PPO (`runs/neo-bc`) then made all 13 single-neo types, 54.8% on average
  (standing start, 16 angles, simulator).
- **DAgger** (`parkour/neo_dagger.py`): at each neo takeoff the policy
  reaches on a training course, a search looks for a landing from its exact
  state (it found one from 84 to 93% of them, simulator,
  `results/neo/neo_dagger_r*.txt`), and those steps become new imitation
  pairs. Two fixes on the way:
  noisy search labels collapsed a first attempt before round 1 (held-out
  success on levels 0-8 dropped to 59%, from 86.5% for `neo-bc`, simulator),
  so the search now starts from the policy's own action probabilities and
  keeps the bot's choice wherever the neo still lands, and the imitation
  weight went from 1 to 0.3; and round 1 counted one-block-up jumps as neo
  takeoffs (rounds 2 and 3 count only wall blocks).

Simulator, 210 held-out test courses of levels 6-8 that contain a neo (337
neo jumps; `results/neo/`):

| Neo policy | Courses completed | Neo jumps passed, of those reached |
| --- | --- | --- |
| stall penalty only (`neo-stall`) | 0.0% | 0.0% |
| + imitation (`neo-bc`) | 23.3% | 36.1% |
| + DAgger round 1 | 29.5% | 43.4% |
| + DAgger round 2 (best checkpoint) | 30.0% | 46.9% |
| + DAgger round 3 (**chosen** on validation) | 29.0% | 43.3% |

- Rounds 2 and 3 level off at about 30%. The neo policy is round 3's
  checkpoint (`runs/neo-dagger3/checkpoints/best.pt`), which won on the
  validation neo courses (34.4% against 33.0% for round 2). From a standing
  start it lands single neos 60.6% of the time, all 13 types (round 1: 70.2%).
  By number of neos in the course: one 56/129, two 3/47, three 2/25, four or
  more 0/9 (test courses, simulator, `results/neo/by_neo_count.json`).
- **Real Minecraft:** on the neo course set (`configs/neo_live_courses.json`:
  30 held-out test courses with a neo, plus the showcase neo course) the neo
  policy completed 11/31 offline and 11/31 in real Minecraft, the same
  courses as the simulator, 0 server corrections. Every course it completed
  has one neo. It fails the showcase neo course: it passes the first neo
  (from a 2x2 block) and falls at the second neo, after 3 of the 11 jumps,
  the same in the simulator, offline and in real Minecraft
  (`results/games/neo-live-final`, `neo-offline-final`).
- The neo policy and the final policy are separate networks: the neo policy
  is a fine-tune of the discount-0.99 line on levels 0-8; the final policy
  is trained on levels 0-5 with obstacles.

## New abilities (28 Oct evening, cloud session; simulator unless said)

The cloud session cannot use the PC's GPU, so it built and checked these on
its CPU; the training for them is queued for the PC
(`configs/sweep_new_skills.json`). Plan and status: `docs/abilities_plan.md`.

- **The bot plans its hardest jumps.** When it first stands on a neo's
  takeoff (or a wide or run-up jump's), a search with the exact game physics
  finds an action sequence from its exact state, and the bot plays it.
  On the 218 validation neo courses: the final policy alone completes 0;
  with planning at neos 49; planning every advanced jump 124 (57%), passing
  221 of the 289 neos it reaches (measured with the search the bots
  reproduce; a first run with batched searches made 119). Every one of the
  900 plans played landed exactly as searched; a search prior from the
  policy made it worse (111). Planning every jump, and stopping on the
  takeoff to search again when a search finds nothing: 147 (67%), 246 of
  311 neos. On random courses of ordinary jumps where half the jumps are
  neos, planning every jump completes 106 of 200 (the policy alone 27),
  with the stop-and-search-again 119, and with one plan over the preceding jump
  and the neo as well 135. On official validation courses, approach planning
  completes 157/218 (72.02%), versus 147/218 (67.43%) without it. The
  cloud new-abilities test task is on hold; the PC training queue continues.
  Archived policy-alone test results are 0/210 official neo courses and
  35/200 mixed courses (`results/neo/assist_test.jsonl`), reported only. The offline bot (prismarine-physics, the live bot's
  engine) reproduces the simulator exactly with this planner: on the 31
  neo-set courses the neo policy made 11 of in real Minecraft, the final
  policy with the planner makes 25 offline, the same courses as in the
  simulator. The live bot has it too (it stands still on the takeoff while
  the planner thinks), not yet run on a real server.
- **Head hitters and ice run-ups**: new jump kinds (a run-up under a
  2-high roof, a tunnel, an ice runway), checked possible by a scripted
  study. The final policy, never trained on them, makes 68% of the head
  hitter kinds alone, 22% of the hard ones and 49% of the ice ones; whole
  courses of those kinds 25.5%, 1.5% and 7.0%, and with the planner 31/40,
  38/40, 32/40.
- **New blocks in the physics**, each matching prismarine-physics to
  1.4e-14 blocks over 60,000 ticks: slabs, stairs, soul sand, carpets,
  trapdoors, fence and wall posts, ladders (block shapes), slime (bounces),
  ladders and vines (climbing). Courses can use them as jump kinds that
  land on a shaped block, bounce off a slime pad, or climb a ladder onto a
  wall; both bots build the real blocks and the offline bot plays them
  exactly like the simulator.

## No course list: the bot finds its own route

Requested from the side chat: run the trained bot on maps where nobody gives
it the list of platforms. A **greedy route planner** (`parkour/planner.py`,
same code in `bridge/planner.js`) looks at the blocks around the bot after
every landing, finds the platforms, and picks the next three; the final
policy jumps to them, no retraining. Only the goal position is given.

| Unseen courses | Course list given | Planner | Planner, first policy |
| --- | --- | --- | --- |
| 1,200 standard, simulator | 98.4% | **92.3%** | 89.6% |
| 1,200 with obstacles, simulator | 95.2% | **90.5%** | 86.2% |
| 60 standard, **real Minecraft** | 59/60 | **51/60** | 52/60 |
| 60 with obstacles, **real Minecraft** | 57/60 | **56/60** | 53/60 |

For both policies, real Minecraft and the offline bot match the simulator on
every course, 0 server corrections. If a drop onto any later, lower platform
counts as a landing (the fall rule named `remaining`, evaluation only), the
planner makes 94.5% and 91.4% in the simulator. With the first policy it also
completed the three hand-designed courses without a neo (tour, stairs,
sprint) in real Minecraft, 0 server corrections; it was not run on the neo
course. Most failures (first policy, simulator, report 7.11: 104 of 125
standard, 116 of 165 obstacles) come after a shortcut: the planner skips
ahead to a later platform in reach. Some of those jumps are harder, and 40
of the 125 standard failures come from the fall rule: it calls a fall while
the bot is above the lower platform it chose
(`results/planner/sim_test_*_v1.json`). Videos, first
policy: `real_minecraft_showcase_no_course_list.mp4` (first person) and
`replay_233_*` / `replay_240_*` (the plan drawn as numbered cyan platforms).

**Learned model that finds its route itself** (`runs/noroute-v2`, 3.2 hours
of training; wide block scan, memory of its last three landings, platform
list and progress hidden): 85.4% of standard and 70.9% of obstacle courses
in the simulator; in real Minecraft 55/60 and 37/60, matching the simulator,
0 corrections. It beats the earlier no-route arm (82.4%) and stays below the
planner on every large test set. Obstacles fool it: 10 of its 23 falls on the
60 obstacle courses in real Minecraft end on top of one (6 on a beam, 4 on a
decoy pillar). So the planner is the model for new maps; the learned one is
the research comparison (report 7.12). Video: `replay_295_learned_*` (level 5,
a course the planner failed).

**Retuned on 28 Oct** (chosen on the
validation courses): ranking platforms by gap length with a weaker pull to
the goal gives **92.3%** standard and **90.5%** obstacle courses in the
simulator (from 90.9% and 88.1%), and 51/60 and 56/60 with the offline bot;
real Minecraft still to be run on the PC. Videos: `real_minecraft_showcase_no_course_list.mp4`
(first person) and `replay_233_*` / `replay_240_*` (the plan drawn as
numbered cyan platforms). Report section 7.11.

The retuned planner is confirmed in real Minecraft: 51/60 standard and 56/60 obstacles, all outcomes identical to the simulator, zero server corrections. It is now the default in Python and JavaScript (`cost=distance`, `align=2.0`). Evidence: `results/games/v2-tuned-planner-live*`.

## Last known state

The cloud work is merged through `8c8bdfa`, including resumable assist parts and per-kind previews, hardened on the PC against duplicate workers, mixed cached evidence and repeated result rows. The full CPU suite passed 109 tests and 26 subtests (one CUDA test skipped); 13 focused restart/cache checks pass after review. `g995-ice` has finished: selected iteration 449, simulator validation 97.33%, final standard test 97.50%, obstacle test 94.50%. Each test was run once. `g995-adv` has also finished: selected iteration 99, nine-level validation 81.78%, final tests 82.28% standard/78.72% obstacles; on shared levels 0-5, tests 97.42%/94.50%. `g995-ice-s1` has finished: selected iteration 949, simulator validation 97.67%, final standard test 97.25%, obstacle test 95.42%; each test was run once. Against the existing final policy on the same six levels, this is -1.17 percentage points standard and +0.17 obstacles (`results/operations/g995-ice-s1-comparison.json`). `g995-extra` also completed: selected iteration 849, validation 90.71% and final tests 91.21%/89.33% across 12 levels; common levels 0-5 tests 96.75%/93.75% (`results/operations/g995-extra-comparison.json`). The queue now trains ice seeds 2 and 3 using validation only; inspect its state and live processes for current jobs. The user permits 100% GPU use with no memory, duty or thread caps; thermal pause above 65 C/resume below 58 C remains active. The combined completion comparison has been delivered; see `results/operations/g995-initial-comparison.json`. The existing final policy remains unchanged.

Overwatch was observed running at 22:04, with no trainers active and about 40% card memory already in use; this previously required a 13% allocator cap per job. The new unrestricted mode retains 1,024-course pools, 8,192 environments, rollout 64, and 32 minibatches for continuity; game ticks per iteration remain 524,288, while rollout shape and minibatch changes affect optimization. The earlier ice run exceeded its 28% allocator cap during PPO; original logs/config are retained under its `recovery/20261028-220658/` directory, and it resumed the saved checkpoint. Thermal protection and keep-awake remain active. The local queue fills slots; a five-minute native heartbeat follows the latest full-GPU instruction and publishes results. Later recipes are selected from common validation levels 0-5; test scores do not select recipes.

- **GPU temperature watchdog** (user rule, since 28 Oct 07:47):
  `scripts/gpu_watchdog.py` checks the GPU about every second, suspends the
  project's GPU jobs above 65 C (no state lost) and resumes them below 58 C.
  Its log is `runs/gpu_watchdog.log`. The merged watchdog also covers `parkour.neo_dagger` and `parkour.neo_assist`; the live bot and its planner service are excluded.
- Only two training runs fit on the GPU at once (a third spills out of GPU
  memory and makes all of them about 8 times slower, as noted in
  `docs/overnight_plan.md`).

## For the presentation

- **Videos** (`results/videos/`, 1080p MP4):
  - `real_minecraft_neo_level7.mp4` (new): first person, real Minecraft; the
    neo policy completes a held-out level-7 course with one neo (164 ticks,
    0 server corrections).
  - `ghost_race_00_tour.mp4` and `ghost_race_01..06`: ten training stages of
    one agent (demo run `progress-demo`, untrained to 400 updates) run one
    unseen course at once, seen from above. Later stages get further; on
    level 5 no stage gets past the first jump.
  - `real_minecraft_showcase.mp4` (35 s, first final policy): the bot's own
    view in **real Minecraft 1.21.4** on three hand-designed courses it never
    saw (`real_minecraft_tour/stairs/sprint.mp4` separately). All completed,
    0 server corrections.
  - No course list: see the section above.
- **Policy files:** `results/final_policy.pt` (final, discount 0.995) and
  `results/final_policy.json` (the exported network the bot loads); the first
  final policy is `results/final_policy_v1.pt` / `.json`. The neo policy is
  `results/neo_policy.pt` / `.json`, exported from the validation-selected `runs/neo-dagger3/checkpoints/best.pt`; original final policies are retained.
- Pages (open in a browser): `results/theatre.html` (Minecraft-style 3D
  replays and the ghost races), `results/replay_viewer.html` (real-game
  replays), `results/progress.html` (learning progress),
  `results/neo_preview.html` (early neo attempts).
- Figures: `results/figures/`. `python -m parkour.report` copies every run's
  training and evaluation logs to `results/runs/` (in Git); the last copy was
  at 11:34.
- Make new videos: `scripts/render_videos.py` (ghost races) and
  `scripts/film_pov.py` (real Minecraft, first person).
- Watch it live: `run.cmd live --checkpoint results\final_policy.pt --out
  runs\demo --viewer YourMinecraftName`, then join `127.0.0.1:25570` in
  Minecraft Java 1.21.4.

## Open problems anyone can pick up

The full list, with what not to change, is in [HANDOFF.md](HANDOFF.md)
(sections 3 to 5).

1. Neos: the learned neo policy completes 29% of neo courses; with the
   new planning assist (simulator) the final policy completes 55% of the
   validation neo courses. Training the policy itself to do it is still open.
2. One policy for both normal and neo courses (now there are two).
3. New blocks are in the physics and in course pieces (shapes, slime,
   ladders); the policy has not been trained on them yet (PC queue).
4. Hand-built maps: extract the platform chain from a region of a real map.
5. The report discussion and conclusion are drafted. The cloud DOCX/PDF exports predate this merged source; rebuild and check pagination before submission.

## Update log

- **29 Oct 2026, 05:05, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 949, validation 96.17%, selected best 449 (97.00%), 6 levels; g995-ice-s5: iteration 849, validation 93.83%, selected best 99 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 05:00, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 849, validation 95.17%, selected best 449 (97.00%), 6 levels; g995-ice-s5: iteration 799, validation 94.83%, selected best 99 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 04:55, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 799, validation 96.83%, selected best 449 (97.00%), 6 levels; g995-ice-s5: iteration 699, validation 96.33%, selected best 99 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 04:50, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 699, validation 94.08%, selected best 449 (97.00%), 6 levels; g995-ice-s5: iteration 649, validation 95.33%, selected best 99 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 04:45, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 649, validation 94.50%, selected best 449 (97.00%), 6 levels; g995-ice-s5: iteration 599, validation 95.92%, selected best 99 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 04:40, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 599, validation 95.83%, selected best 449 (97.00%), 6 levels; g995-ice-s5: iteration 499, validation 96.67%, selected best 99 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 04:35, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 499, validation 95.75%, selected best 449 (97.00%), 6 levels; g995-ice-s5: iteration 449, validation 94.92%, selected best 99 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 04:30, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 449, validation 97.00%, selected best 449 (97.00%), 6 levels; g995-ice-s5: iteration 349, validation 88.50%, selected best 99 (97.00%), 6 levels. Seed 5's validation drop is concentrated in level 5 (39% success, 8% falls, 53% unfinished); training continues with finite metrics and no runtime errors, and its retained validation best remains unchanged. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 04:23, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 349, validation 95.33%, selected best 199 (95.92%), 6 levels; g995-ice-s5: iteration 299, validation 95.75%, selected best 99 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 04:15, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 199, validation 95.92%, selected best 199 (95.92%), 6 levels; g995-ice-s5: iteration 149, validation 95.75%, selected best 99 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 04:10, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 149, validation 93.17%, selected best 49 (95.25%), 6 levels; g995-ice-s5: iteration 99, validation 97.00%, selected best 99 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 04:05, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s4: iteration 99, validation 95.17%, selected best 49 (95.25%), 6 levels; g995-ice-s5: iteration 49, validation 93.58%, selected best 49 (93.58%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 04:02, Codex heartbeat on training PC:** completed and archived both 1,000-iteration ice seeds in the simulator: g995-ice-s2: validation-selected checkpoint 749, validation 96.9167%, final standard 97.4167%, obstacles 94.9167% (vs existing final policy: -0.6667/-1.0000/-0.3333 percentage points, validation/standard/obstacles); g995-ice-s3: validation-selected checkpoint 249, validation 96.7500%, final standard 97.4167%, obstacles 94.0000% (vs existing final policy: -0.8333/-1.0000/-1.2500 percentage points, validation/standard/obstacles). Comparison uses identical common levels 0-5, 200 courses per level and final test families 1/4. Exactly one final standard and obstacle test row per run, at the validation-selected checkpoint; no evaluation rerun. Sources: `results/operations/g995-ice-s2-comparison.json` and `results/operations/g995-ice-s3-comparison.json`. This combined completion notification is delivered in this chat now; do not repeat. Existing final policy files retained. The queue automatically filled both slots with ice seeds 4/5 using the recipe selected from common validation only; seed 4 validation iteration 49 is 95.25%. Full-GPU settings, thermal watchdog (65/58 C) and keep-awake remain active. Cloud evaluation hold remains in effect. Report/handoff refreshed; independent result audit and CSV checks pass. No executable changes; the 109-test CPU suite from 03:16 remains applicable.

- **29 Oct 2026, 04:00, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s3: iteration 949, validation 95.75%, selected best 249 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 03:53, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 949, validation 95.58%, selected best 749 (96.92%), 6 levels; g995-ice-s3: iteration 899, validation 94.25%, selected best 249 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 03:48, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 849, validation 95.33%, selected best 749 (96.92%), 6 levels; g995-ice-s3: iteration 799, validation 95.50%, selected best 249 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 03:43, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 799, validation 95.67%, selected best 749 (96.92%), 6 levels; g995-ice-s3: iteration 749, validation 95.67%, selected best 249 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 03:38, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 749, validation 96.92%, selected best 749 (96.92%), 6 levels; g995-ice-s3: iteration 649, validation 96.50%, selected best 249 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 03:33, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 649, validation 94.67%, selected best 249 (96.75%), 6 levels; g995-ice-s3: iteration 599, validation 95.75%, selected best 249 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 03:28, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 599, validation 96.08%, selected best 249 (96.75%), 6 levels; g995-ice-s3: iteration 499, validation 95.67%, selected best 249 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 03:23, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 499, validation 94.83%, selected best 249 (96.75%), 6 levels; g995-ice-s3: iteration 449, validation 95.67%, selected best 249 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 03:18, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 449, validation 95.33%, selected best 249 (96.75%), 6 levels; g995-ice-s3: iteration 399, validation 93.75%, selected best 249 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 03:13, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 349, validation 94.67%, selected best 249 (96.75%), 6 levels; g995-ice-s3: iteration 299, validation 93.08%, selected best 249 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. Integrated cloud replay page `8c8bdfa` and fixed orbit switching, disposal of replaced graphics, paused/end-of-recording redraws, and a readable fallback when the 3D CDN is unavailable. The builder now checks the verification-file hash as well. Rebuilt from pinned data `6994797`: all 45 offline showcase episodes and 7,023 frames are unchanged. JavaScript syntax, isolated fallback/playback checks, and rejection of a corrupt verification file pass. Browser visual inspection was blocked by the browser tool local-file URL policy; no alternate route was attempted. Full CPU merge suite passes: 109 tests, 1 CUDA skip, 26 subtests (`runs/heartbeat-0302-merge-tests.txt`); training/physics code is unchanged. Replay-page availability is reported in this heartbeat; prior run-completion notifications are not repeated. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 03:04, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 249, validation 96.75%, selected best 249 (96.75%), 6 levels; g995-ice-s3: iteration 199, validation 96.08%, selected best 149 (96.25%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. Merged cloud replay artifact `8c8bdfa`, preserving both session histories. Rebuilt the 45-episode/7,023-frame page from pinned data commit `6994797`; manifest hashes and exact page content match, JavaScript syntax check passes. No recordings or held-out evaluations rerun. Training code is unchanged; prior CPU/fingerprint checks remain applicable. Cloud evaluation hold remains in effect. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 02:57, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 149, validation 95.83%, selected best 149 (95.83%), 6 levels; g995-ice-s3: iteration 99, validation 95.92%, selected best 99 (95.92%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 02:53, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s2: iteration 99, validation 94.50%, selected best 49 (95.25%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 02:50, Codex heartbeat on training PC:** completed `g995-extra`; simulator validation-selected checkpoint 849, 90.71% across 12 levels. Final tests reported once: 91.21% standard and 89.33% obstacles across all 12; on identical common levels 0-5, 96.75%/93.75%, versus existing final policy 98.42%/95.25% (-1.67/-1.50 percentage points). One test row per set at the selected validation maximum verified. Evidence: `results/operations/g995-extra-comparison.json`. Completion notification delivered together with ice seed 1 in this heartbeat; do not repeat either. Queue automatically started `g995-ice-s3`; ice seeds 2 and 3 now occupy both slots. Ice remains the recipe selected by common validation (initial ice 97.33%, advanced 96.17%, extra 95.33%); tests were not used. Existing final policy retained. Both trainers are unrestricted; thermal-only watchdog and keep-awake verified. No executable changes; recent full CPU/fingerprint checks remain applicable.

- **29 Oct 2026, 02:49, Codex heartbeat on training PC:** completed `g995-ice-s1`; selected checkpoint 949 on simulator validation (families 2/5), 97.67% over six levels. Final tests reported once: 97.25% standard, 95.42% obstacles; archived comparison against `results/final_policy.pt`: -1.17/+0.17 percentage points on the identical held-out level 0-5 courses. CSV maxima, selected iteration and exactly one row per test verified; comparison in `results/operations/g995-ice-s1-comparison.json`. Completion notification delivered in this heartbeat; do not repeat. Extra archived through iteration 999, best 849 at 90.71% over 12 levels; completed=True. Existing queue filled the freed slot automatically; active jobs: g995-ice-s2, g995-ice-s3. No executable changes; full CPU and fingerprint checks remain applicable. Reports, status docs and the monitor prompt updated; cloud hold remains separate from PC training.

- **29 Oct 2026, 02:41, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 899, validation 89.62%, selected best 849 (90.71%), 12 levels; g995-ice-s1: iteration 949, validation 97.67%, selected best 949 (97.67%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. Merged documentation-only cloud `2abdc47`, which records frozen cloud evaluation jobs and leaves PC training untouched. No executable changes; the full CPU suite (109 passed) and held-out fingerprint checks (3 passed, 26 subtests) from this session remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 02:38, Codex heartbeat on training PC:** merged cloud `cd81960`/`d2b81dd`/`39882fb` (data/docs only). CPU simulator official neo validation family 2 with approach: 157/218 (72.02%) versus brake/retry 147/218 (67.43%), +4.59 percentage points; archived detail/JSONL aggregates and shipped-checkpoint hashes verified. Newly archived policy-alone tests are official 0/210 and mixed 35/200; the mixed row and bins agree on 35 despite the commit message saying 30. No tests rerun for these results, and no test-based selection. The cloud evaluation hold is preserved and explicitly leaves PC training running; no cloud resume command was executed. Both complete session logs preserved. Current status docs reconciled. `run.cmd test -k HeldOutFingerprints`: 3 passed, 107 deselected, 26 subtests passed (`runs/heartbeat-0236-fingerprint-tests.txt`); the unchanged executable code also passed the full 109-check CPU suite minutes earlier. Existing two trainers, queue, thermal watchdog and keep-awake continue.

- **29 Oct 2026, 02:37, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 899, validation 89.62%, selected best 849 (90.71%), 12 levels; g995-ice-s1: iteration 899, validation 95.67%, selected best 599 (96.17%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 02:35, Codex heartbeat on training PC:** merged cloud commits `711846a`/`c87068d`, data and docs only. CPU simulator random neo validation family 13: all-jump planning with brake/retry plus approach completes 135/200 courses (67.5%), versus 119/200 (59.5%) without approach: +8.0 percentage points; 332/392 reached neos passed. Detail and JSONL aggregates, checkpoint and dataset identities reconciled (`results/neo/assist_details/random13_all_brake_approach.json`). Approach jointly plans the preceding jump and the neo; corrected that wording and the abilities-plan status. Official-family approach validation remains pending. Both complete session logs retained; generated conflicts refreshed by report. Full CPU `run.cmd test`: 109 passed, 1 CUDA test skipped, 26 subtests passed (`runs/heartbeat-0228-merge-tests.txt`). No held-out tests rerun; two trainers and all existing helpers continue.

- **29 Oct 2026, 02:35, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 849, validation 90.71%, selected best 849 (90.71%), 12 levels; g995-ice-s1: iteration 849, validation 94.33%, selected best 599 (96.17%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 02:29, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 799, validation 90.00%, selected best 799 (90.00%), 12 levels; g995-ice-s1: iteration 799, validation 87.33%, selected best 599 (96.17%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 02:23, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 749, validation 89.17%, selected best 699 (89.79%), 12 levels; g995-ice-s1: iteration 749, validation 96.08%, selected best 599 (96.17%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 02:18, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 699, validation 89.79%, selected best 699 (89.79%), 12 levels; g995-ice-s1: iteration 649, validation 95.58%, selected best 599 (96.17%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 02:13, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 649, validation 88.71%, selected best 649 (88.71%), 12 levels; g995-ice-s1: iteration 599, validation 96.17%, selected best 599 (96.17%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 02:08, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 599, validation 87.42%, selected best 299 (88.12%), 12 levels; g995-ice-s1: iteration 549, validation 95.25%, selected best 99 (95.83%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 02:03, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 549, validation 87.96%, selected best 299 (88.12%), 12 levels; g995-ice-s1: iteration 499, validation 95.83%, selected best 99 (95.83%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 01:58, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 499, validation 83.17%, selected best 299 (88.12%), 12 levels; g995-ice-s1: iteration 399, validation 95.42%, selected best 99 (95.83%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 01:53, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 449, validation 87.17%, selected best 299 (88.12%), 12 levels; g995-ice-s1: iteration 349, validation 95.42%, selected best 99 (95.83%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 01:48, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 399, validation 87.58%, selected best 299 (88.12%), 12 levels; g995-ice-s1: iteration 299, validation 93.83%, selected best 99 (95.83%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 01:46, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 349, validation 86.58%, selected best 299 (88.12%), 12 levels; g995-ice-s1: iteration 249, validation 95.50%, selected best 99 (95.83%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 01:38, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 299, validation 88.12%, selected best 299 (88.12%), 12 levels; g995-ice-s1: iteration 149, validation 94.33%, selected best 99 (95.83%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 01:33, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 249, validation 83.12%, selected best 249 (83.12%), 12 levels; g995-ice-s1: iteration 99, validation 95.83%, selected best 99 (95.83%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 01:29, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-extra: iteration 199, validation 81.75%, selected best 199 (81.75%), 12 levels; g995-ice-s1: iteration 49, validation 94.83%, selected best 49 (94.83%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 01:25, Codex heartbeat on training PC:** both requested initial runs completed 1,000 iterations. Delivered the combined validation/test comparison in this chat; do not repeat that notification. Advanced selected checkpoint 99 on nine-level validation 81.78% (common levels 0-5: 96.17%); final tests across nine levels: 82.28% standard / 78.72% obstacles. On the identical common levels 0-5: 97.42% standard / 94.50% obstacles. Ice selected checkpoint 449 on six-level validation 97.33%; final tests 97.50% / 94.50%. Existing final policy (checkpoint 699): validation 97.58%, tests 98.42% / 95.25%. Thus advanced trails the baseline on common tests by 1.00 / 0.75 percentage points and ice by 0.92 / 0.75 points. All numbers are simulator results, validation families 2/5 and test families 1/4; each new run has exactly one test row per split, and selected iterations match validation maxima. Evidence: `results/operations/g995-initial-comparison.json` with source files. No tests were repeated and the existing final-policy files remain unchanged. The queue automatically started g995-ice-s1 beside g995-extra, selected from common validation (ice 97.33% versus advanced 96.17%), never test results. Extra new validation best 149: 76.71% across twelve levels. Exactly two unrestricted trainers, one queue, one thermal watchdog (65/58 C) and one keep-awake are healthy; current telemetry 83% busy, 5791/16303 MiB, 41.58 C. Native monitor updated to record delivered notification and continue extra/seeds. Report/handoff refreshed; no executable changes, so prior checks remain applicable.

- **29 Oct 2026, 01:19, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-adv iteration 949: 81.22%, selected best 99 (81.78%); g995-extra iteration 99: 73.29%, selected best 99 (73.29%). Advanced covers nine levels and extra twelve; overall scores are not directly comparable. Summaries and selected validation maxima match their CSVs; both active test logs remain empty, while ice retains exactly one final standard/obstacle test row. Two unrestricted trainers and the existing queue, thermal watchdog (65/58 C), and keep-awake are healthy. GPU sample: 71% busy, 5350/16303 MiB, 40.23 C. No new cloud commits or executable changes; prior checks remain applicable. Report/handoff refreshed and results published. Combined advanced/ice completion notification remains pending until advanced finishes.

- **29 Oct 2026, 01:15, Codex heartbeat on training PC:** merged cloud documentation commit `5f22b61`, retaining its cloud CPU session entry and the PC session history. Archived simulator validation (families 2/5): g995-adv iteration 899: 81.17%, best remains iteration 99 at 81.78%; g995-extra first checkpoint 49: 59.625%, selected best 49. Advanced covers nine levels and extra twelve, so these overall scores are not directly comparable. Verified archived summaries against their validation CSVs and validation-selected maxima; both active runs have no test rows. Ice remains completed and reported once. Two unrestricted trainers, one queue, one thermal watchdog (65/58 C) and one keep-awake remain healthy; current GPU sample was 76% busy, 5349/16303 MiB, 38.38 C. No executable changes; the prior full CPU suite and 13 focused restart checks remain applicable. Report/handoff refreshed. Combined advanced/ice completion notification remains pending until advanced finishes.

- **29 Oct 2026, 01:11, Codex on training PC:** merged cloud `c38de17` without rewriting either history. Hardened its opt-in resumable assist/preview evaluators: canonical manager/part locks, cleanup of sibling workers, checkpoint/device/course identity checks, atomic cache/detail/marker writes, and idempotent result rows after interrupted publication; existing test results can be republished from cache without reevaluation. The full CPU suite passes 107 tests and 26 subtests, with one CUDA test skipped to avoid extra GPU work (`runs/heartbeat-resume-merge-tests.txt`); after review, 13 focused cache/lock/publication tests pass (`runs/heartbeat-resume-cache-focused.txt`). No training/physics or held-out course changes. Ice completed 1,000 iterations: validation-selected checkpoint 449 scored 97.33% on families 2/5, then its only final test scored 97.50% standard and 94.50% obstacles (families 1/4, simulator, six levels). The existing queue immediately started g995-extra beside g995-adv. Advanced archived iteration 849 validation is 81.00% across nine levels; best remains iteration 99, tests remain unrun. Exactly two unrestricted trainers, one thermal watchdog (65/58 C), one queue and one keep-awake remain active. The user requests one combined notification after both initial runs finish; the native monitor now retains this request. `results/operations/g995-initial-comparison.json` contains the existing final-policy baseline and completed ice comparison, with ready=false until advanced finishes. Final-policy obstacle baseline is 95.25% from its selected iteration-699 CSV (the published one-decimal headline is 95.2%). No completion notification has been delivered yet. Report refreshed; next: finish advanced and send the combined validation/test comparison while extra continues.

- **29 Oct 2026, 00:55, Codex on training PC:** applied the user's new permission for 100% GPU use. Replaced the shared throttle with thermal-only watchdog (65/58 C, one-second telemetry); kept the existing queue and keep-awake. Advanced and ice saved checkpoints 699/799 and resumed from their own latest.pt without memory, duty or thread caps; rollout dimensions and training recipe remain unchanged. Verified exactly two progressing trainers, one queue, one watchdog and one keep-awake. The 30-sample card-wide measurement averaged 75.2% GPU busy time and peaked at 42.3 C; full availability does not imply constant utilization. Evidence: `results/operations/full-gpu-transition-20261029.json`. Updated the native five-minute monitor to follow the latest full-GPU instruction. Five focused CPU queue/watchdog tests pass (`runs/full-gpu-mode-checks.txt`). Refreshed report and checked simulator validation summaries against CSVs (families 2/5): g995-adv iteration 699: 81.44% validation, selected best 99; g995-ice iteration 799: 93.75% validation, selected best 449. Advanced has nine levels and ice six; these overall scores are not directly comparable. Test logs remain header-only; extra remains next.

- **29 Oct 2026, 00:47, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 649, validation 80.61%, selected best iteration 99; g995-ice: iteration 749, validation 94.50%, selected best iteration 449. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **29 Oct 2026, 00:43, Codex heartbeat on training PC:** merged cloud baseline `1752c13`: the shipped final policy without assistance completes 37/200 mixed simulator validation courses (family 15, ordinary and special jump kinds). Verified checkpoint hash and aggregate count consistency; no per-course detail file accompanies this result. The mixed dataset matches its pinned fingerprint. Added the mixed family definitions to the report; three integrity tests and 26 subtests pass (93 deselected; `runs/heartbeat-mixed-baseline-tests.txt`). Archived training validation (families 2/5): g995-adv iteration 599: 80.17%, selected best 99; g995-ice iteration 699: 95.00%, selected best 449. Advanced uses nine levels and ice six; checkpoints remain selected only on validation, and training test logs are header-only. Report/handoff refreshed; two trainers and the existing queue/watchdog/keep-awake remain active.

- **29 Oct 2026, 00:34, Codex heartbeat on training PC:** merged cloud diagnostic result `ed31daf`. Verified all 826 single-jump kinds at 16 starting angles and their aggregate summaries: the shipped final policy succeeds on shaped-block kinds at 99.06%, slime bounces at 39.94%, ladder climbs at 37.50% (simulator; `results/extra/jump_eval_final_policy.json`). These are single-jump diagnostics, separate from validation/test course families and not used for checkpoint or recipe selection. Retained original result fields; clarified the rendered label as "no scripted landing" because the policy does pass one such kind, and expanded the data index to cover all measured kinds. Regenerated report/handoff; three fingerprint tests and 26 subtests passed (93 deselected; `runs/heartbeat-extra-jumps-tests.txt`), and rendered labels/data totals were checked. Existing training, queue and safety helpers continue.

- **29 Oct 2026, 00:34, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 599, validation 80.17%, selected best iteration 99; g995-ice: iteration 699, validation 95.00%, selected best iteration 449. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. Training code is unchanged; reporting changes and checks are recorded in the accompanying merge entry.

- **29 Oct 2026, 00:29, Codex heartbeat on training PC:** merged cloud test/documentation commits through `9f7a9a9`. Fingerprints now pin mixed families 14/15 and the six extra-kind preview sets (test/validation); all match the preserved PC level definitions. Three integrity tests and 26 subtests pass (93 deselected; `runs/heartbeat-extra-fingerprints-tests.txt`). No runtime code changed. Kept the actual two-slot PC queue and its limits in the abilities plan; the cloud-reported `extra-ft` smoke check uses uniform curriculum, while pending `g995-extra` keeps the requested adaptive recipe. That smoke output was not archived, so it is not presented as a verified result of the PC queue. Refreshed stale planner-status cells from previously verified evidence and regenerated report/handoff. Both trainers and safety helpers remain healthy; training test sets remain unreported.

- **29 Oct 2026, 00:24, Codex heartbeat on training PC:** merged cloud generated-data commits through `d5943ca`. The extra-level file is byte-identical to the existing PC output (six levels, 286/40/194/450/41/4 specs). Retained the PC-generated feasibility artifact without hand-editing values: seven shape fractions differ by 0.00001 between the PC and cloud artifacts and the latest reuse duration differs; all 826 candidates and level membership agree. Both source commits and differences are archived in `results/operations/extra-feasibility-reconciliation.json`. Pending extra training and its evaluation courses are unchanged. Two held-out-integrity tests and 12 subtests pass (93 deselected; `runs/heartbeat-extra-reconciliation-tests.txt`). No trainer, queue or watchdog restart was needed.

- **29 Oct 2026, 00:24, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 549, validation 80.56%, selected best iteration 99; g995-ice: iteration 649, validation 96.08%, selected best iteration 449. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **29 Oct 2026, 00:19, Codex heartbeat on training PC:** merged cloud report/data commits through `4c6761d`, retaining both histories. Verified the compressed offline-bot evidence: final policy plus all-jump planning/braking/retry completes 25/31 neo-set courses, with 350 verified plans from 366 requests, zero planner errors, and simulator outcomes/tick counts identical on all 31. This fixed set contains 30 held-out test examples plus the showcase; it is reported evidence only, not validation or a tuning signal, and was not re-evaluated on the PC. Real-Minecraft verification of this assist remains pending. Merged the assist table and marked offline server corrections as not applicable. Three focused report/course-integrity tests and 12 subtests pass (92 deselected; `runs/heartbeat-offline-neo-tests.txt`); training/physics code is unchanged. Archived simulator training validation (families 2/5): g995-adv iteration 499: 81.33%, best iteration 99; g995-ice iteration 599: 94.67%, best iteration 449. Advanced uses nine levels and ice six; both trainers still select on validation only and their test logs remain header-only. The existing queue, two trainers, watchdog and keep-awake continue under the 70% target; extra remains next.

- **29 Oct 2026, 00:12, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 499, validation 81.33%, selected best iteration 99; g995-ice: iteration 599, validation 94.67%, selected best iteration 449. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **29 Oct 2026, 00:07, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 499, validation 81.33%, selected best iteration 99; g995-ice: iteration 549, validation 95.83%, selected best iteration 449. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **29 Oct 2026, 00:02, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 449, validation 80.83%, selected best iteration 99; g995-ice: iteration 549, validation 95.83%, selected best iteration 449. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **28 Oct 2026, 23:52, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 449, validation 80.83%, selected best iteration 99; g995-ice: iteration 499, validation 96.58%, selected best iteration 449. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **28 Oct 2026, 23:48, Codex heartbeat on training PC:** merged data/documentation commits through `8b918e2`, preserving both histories and regenerating conflicting report/snapshot tables. Verified cloud CPU simulator validation evidence against all per-course records: planning all jumps with braking/retry completes 119/200 random neo courses (family 13), passing 320/396 reached neos. No executable changes since the tested assist integration. `run.cmd test -k HeldOutFingerprints` passes 2 tests and 12 subtests (93 deselected; `runs/heartbeat-random-brake-tests.txt`). Archived simulator training validation (families 2/5): g995-adv iteration 399: 81.78%, best iteration 99; g995-ice iteration 499: 96.58%, best iteration 449. Advanced covers nine levels, ice six; their overall scores are not directly comparable. Validation alone selects checkpoints; test logs remain header-only. The existing two trainers, queue, thermal/duty watchdog and keep-awake remain healthy; extra stays next.

- **28 Oct 2026, 23:42, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 399, validation 81.78%, selected best iteration 99; g995-ice: iteration 449, validation 97.33%, selected best iteration 449. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **28 Oct 2026, 23:37, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 349, validation 77.28%, selected best iteration 99; g995-ice: iteration 449, validation 97.33%, selected best iteration 449. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **28 Oct 2026, 23:26, Codex heartbeat on training PC:** merged cloud commits through `7208ebe` with both histories retained. The watchdog now includes offline assist services/runs while excluding real-time services and keep-awake; retained the strict module match and all PC pacing/thermal controls, then restarted only the watchdog with the same limits. Planner-service failures resolve outstanding requests and are preserved in episode/summary JSON; follow-up fixes keep normal shutdown quiet and mark degraded assist runs in saved Markdown/report tables. Full CPU suite: 92 passed, 1 CUDA test skipped, 12 subtests passed (`runs/heartbeat-offline-assist-merge-tests.txt`); three focused safety/failure checks also pass (`runs/heartbeat-assist-failure-tests.txt`). Cloud simulator validation evidence verified: planning every jump with braking/retry completes 147/218 official neo courses (family 2); no test selection. Archived simulator training validation: g995-adv iteration 349: 77.28%; g995-ice iteration 399: 96.17%. Advanced uses nine levels and ice six; selected best remains iteration 99 for both, and test logs remain header-only. Two trainers, one queue, one watchdog and keep-awake continue; extra stays next.

- **28 Oct 2026, 23:13, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 299, validation 79.72%, selected best iteration 99; g995-ice: iteration 349, validation 95.83%, selected best iteration 99. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **28 Oct 2026, 23:03, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 249, validation 80.72%, selected best iteration 99; g995-ice: iteration 299, validation 95.33%, selected best iteration 99. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **28 Oct 2026, 22:56, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 199, validation 78.50%, selected best iteration 99; g995-ice: iteration 249, validation 96.08%, selected best iteration 99. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **28 Oct 2026, 22:49, Codex heartbeat on training PC:** archived new simulator validation progress (families 2/5): g995-adv: iteration 199, validation 78.50%, selected best iteration 99; g995-ice: iteration 199, validation 96.67%, selected best iteration 99. Advanced covers nine levels and ice six, so overall scores are not directly comparable. Verified each summary against its validation CSV and the selected validation maximum; test logs remain header-only and no completion/test result exists. Report and handoff refreshed. Exactly two trainers and the existing queue, thermal/duty watchdog and keep-awake helper remain healthy. No executable changes; prior tests remain applicable.

- **28 Oct 2026, 22:41, Codex heartbeat on training PC:** merged documentation-only cloud commit `45e5e09`; its four validation counts match the archived CPU simulator evidence. Updated the abilities-plan queue row to the actual PC queue and corrected stale neo-export wording. No executable code changed since the 91-passing full CPU suite; focused `run.cmd test -k HeldOutFingerprints` passed 2 checks and 12 subtests (`runs/heartbeat-plan-docs-tests.txt`). Archived ice iteration 199 validation 96.67% (simulator, families 2/5); its best remains iteration 99, with no test evaluation. Exactly two trainers, one repaired queue, one watchdog and one keep-awake helper remain active.

- **28 Oct 2026, 22:37, Codex heartbeat on training PC:** repaired queue persistence after Windows denied an atomic state-file replacement (`runs/queue-20261028-220659.stderr.log`). Both trainers continued uninterrupted; restored only the manager at 22:29, retaining their process IDs and existing state. Transient replace locks retry; persistent write errors preserve the prior file and live inventory for the next loop; unreadable config prevents new launches. Two regression checks cover transient/persistent locks. Merged cloud approach-planning refinement `758eadd`; full CPU suite: 91 passed, 1 CUDA test skipped, 12 subtests passed (`runs/heartbeat-queue-recovery-tests.txt`). Archived iteration-149 simulator validation: advanced 79.39% on nine levels, ice 96.25% on six; validation-selected best remains iteration 99 for both. Test sets remain unreported until completion.

- **28 Oct 2026, 22:26, Codex heartbeat on training PC:** merged cloud result-only commit `37ede6e`: exact planning completes 106/200 random neo validation courses (family 13, CPU simulator; detail/summary consistency checked). `run.cmd test` passed on CPU with one-thread test workers: 87 passed, 1 CUDA test skipped, 12 subtests passed (`runs/heartbeat-random-neo-tests.txt`). Archived simulator validation at iteration 99: advanced 81.78% across nine levels and ice 96.83% across six levels, both families 2/5; these different overall sets are not directly comparable and neither test set has been evaluated. Both trainers and all safety helpers remain running under the current limits.

- **28 Oct 2026, 22:18, Codex heartbeat on training PC:** merged seven cloud commits through `3ecc266`, preserving both session logs and current PC limits. The new assist course-file/retry fixes and held-out fingerprints pass `run.cmd test` on CPU: 87 passed, 1 CUDA test skipped to avoid extra GPU load, 12 subtests passed (`runs/heartbeat-cloud-merge-tests.txt`). The earlier full CUDA-enabled suite passed before these assist-only changes. Both trainers are progressing with the same 13% caps; ice iteration 99 validation is 96.83% (families 2/5, simulator), with no test evaluation yet. Refreshed report and archived logs. Keep the existing two-slot queue; do not start neo_pipeline until it inherits the PC controls.

- **28 Oct 2026, 22:10, Codex on training PC:** merged cloud through `21188a6`; 85 tests and 8 subtests passed, then 6 focused checks passed after releasing unused validation cache. Regenerated extra feasibility/levels through the study command. Retuned planner live verification and neo export complete. Replaced rejected long-burst duty with 50-ms shared feedback pacing. Overwatch was found running; resumed both saved iteration-49 checkpoints with 13% allocator caps and smaller tensor working sets, preserving the ice OOM attempt and 41 uncheckpointed updates in recovery logs. Card-wide telemetry is archived in `results/operations/gpu-limit-verification.json`; a five-minute heartbeat monitors results and publication. Next: finish requested three recipes, select by validation only, report each test once.

- **28 Oct 2026, 21:38, Codex on the training PC:** retuned planner confirmed in real Minecraft (51/60 standard, 56/60 obstacles, 0 corrections, exact outcome parity) and adopted as default in both implementations. Cloud work merged again through `21188a6`. User changed GPU budget to 70% compute and memory. Compact friction and evaluation batches preserve physics and fixed evaluation courses; extra training uses a 2,048-course pool to fit the budget (16,384 environments unchanged).
- **28 Oct 2026, evening, Codex on the training PC:** merged cloud work, retained the PC audit corrections and both histories, exported the neo policy, fixed relative-path checkpoint resume and preserved validation best, added opt-in final-test-only training and a two-slot queue. 56 tests and 8 subtests pass. Next: GPU training and real-Minecraft confirmation of the retuned planner.
- **29 Oct 2026, night (cloud session):** a 3D replay page of the 45
  showcase recordings made after training (the advanced and ice policies,
  offline physics, failures included): `results/showcase_replays.html`.
  The new-abilities test runs are on hold until the user resumes them.

- **28 Oct 2026, evening (cloud session):** new abilities: planning the
  hardest jumps with the exact physics (neo assist; simulator and both
  bots), head-hitter and ice run-up jump kinds, block shapes, slime and
  ladders in the physics and in courses (exact against prismarine-physics),
  a gaming-friendly training mode for the PC and its queue. Details above.

- **28 Oct 2026, afternoon (cloud session):** this page brought up to date:
  final policy with discount 0.995 (98.4% / 95.2% in the simulator; 59/60,
  57/60, 3/4 in real Minecraft), neos learned from demonstrations and DAgger
  (0% to 29% of neo courses; 11/31 in real Minecraft), hyperparameter study
  finished. Then, on the CPU: the route planner retuned (92.3% / 90.5%
  without the course list, from 90.9% / 88.1%), the neo policy shown to be
  only 2-3 points behind the final policy on ordinary courses, a failure
  analysis by jump type, ice platforms ready to train (physics checked
  against the real library; the current policy already makes 95.6% of
  all-ice test courses without having seen ice), and the report
  completed (20 pages in Word/PDF). `HANDOFF.md` and `AGENTS.md` tell any AI
  model how to continue.
- **28 Oct 2026, 12:40:** hyperparameter study finished. Discount 0.995,
  trained at full length, became the final policy at 10:41 (simulator test
  98.4% and 95.2% against 97.1% and 93.6%) and was re-run offline, in real
  Minecraft (59/60, 57/60, 3/4) and with the planner. Neos learned from
  demonstrations and DAgger: 0% to 29.0% of held-out neo courses in the
  simulator, 11/31 neo courses in real Minecraft, a first-person neo video. An
  audit checked 829 claims in the documents against the data (60 errors fixed,
  14 disputed claims resolved; `results/audit_claims.json`); physics parity
  re-run. GPU temperature watchdog running since 07:47.
- **28 Oct 2026, 02:00:** ghost-race and real-Minecraft first-person videos;
  double and triple neos searched (none found); neo refusal diagnosed and a
  stall penalty tried; run logs pushed to GitHub.
- **28 Oct 2026, 00:10:** all training data, evaluation logs and game
  recordings (real and offline, every tick) archived in `results/`; training
  curve figures; GitHub push working. Neo training running.
- **27 Oct 2026, 23:45:** EULA accepted; first real-Minecraft runs (59/60,
  58/60 with obstacles, 3/3), identical to the simulator. Neo, double-neo,
  momentum and wide jumps added.
- **27 Oct 2026, 23:00:** ablation (15 runs), obstacle fine-tune and no-route
  runs finished; the obstacle-trained policy became the first final policy.
- **27 Oct 2026, evening:** main run finished (95.5% held-out); PPO
  stabilised after early collapses.
- **27 Oct 2026, afternoon:** new direction (parkour). Simulator, physics
  parity, feasibility study, environment, PPO, JS bot and offline check built.
