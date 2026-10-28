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

**Last updated:** 28 October 2026, evening, training PC: cloud work merged through `21188a6`; validation-only training queue being prepared.

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
  policy made it worse (111). Planning every jump and braking before a
  second search are being measured. The offline bot (prismarine-physics, the live bot's
  engine) reproduces the simulator exactly with this planner; the live bot
  has it too (it stands still on the takeoff while the planner thinks), not
  yet run on a real server.
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

The cloud work is merged through `45e5e09`. The latest CPU merge checks pass (91 tests, 12 subtests; one CUDA test skipped to avoid extra GPU load). The earlier CUDA-enabled suite passed 85 tests and 8 subtests, with 6 focused checks after the cache-release change. Both `g995-adv` and `g995-ice` resumed their own iteration-49 checkpoint at 22:07, with `g995-extra` next. The latest user instruction limits both GPU compute and memory to 70%. The initial 10-second duty cycle produced unacceptable long 98% bursts; it has been replaced by shared 50-ms pacing with feedback from measured GPU usage, targeting headroom below 70%. This software control cannot guarantee every transient sample or the user's game stays below 70%.

Overwatch was observed running at 22:04, with no trainers active and about 40% card memory already in use. Training now uses a 13% allocator cap per job, 1,024-course pools, 8,192 environments, rollout 64, and 32 minibatches; game ticks per iteration remain 524,288, while rollout shape and minibatch changes affect optimization. The earlier ice run exceeded its 28% allocator cap during PPO; original logs/config are retained under its `recovery/20261028-220658/` directory, and it resumed the saved checkpoint. Shared 70% compute feedback, thermal protection and keep-awake remain active. The local queue fills slots; a five-minute native heartbeat monitors progress and publishes results. Later recipes are selected from common validation levels 0-5; test scores do not select recipes.

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
