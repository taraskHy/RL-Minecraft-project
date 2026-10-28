# Handoff

The living handoff for whoever works on this project next: any model (Claude,
Codex, Gemini, Cursor, ...) or a person. Rules for working here are in
[AGENTS.md](AGENTS.md); this file says where the project stands, what is left,
what not to touch and which problems are still open.

**Keep it current.** Every session updates the hand-written sections it made
untrue, adds a line to the [session log](#session-log), and refreshes the
generated [snapshot](#9-snapshot-generated) at the end (section 10 says how).
`python -m parkour.handoff --check` lists what changed since the last update.

Read next: [TEAM_STATUS.md](TEAM_STATUS.md) (the team's page),
[docs/design.md](docs/design.md) (method), [results/README.md](results/README.md)
(every number, generated) and [results/DATA.md](results/DATA.md) (where the data is).

## 1. Goal and how the user works

- Goal (since 27 Oct 2026): an RL bot that completes **unseen** Minecraft
  parkour courses and reaches the target. Requested on 27 Oct: harder jumps
  (neo, double neo, 4-block jumps). The earlier survival project is archived
  (`archive/survival` tag; `codex/*` branches untouched).
- The user manages at a high level and wants as few questions as possible.
  Ask only about direction or the user's own decisions.
- **One session changes the project.** On 28 Oct the user decided that only
  the main session (on the training PC) edits files, commits and starts or
  stops runs; other sessions are for questions only. A side chat once
  committed and restarted a run in parallel; check `git log` and running
  processes before acting. Sessions elsewhere (a cloud container, another
  model's sandbox) push only to the branch they were given, never `main`.
- Team of three students; course-level methods (PPO, actor-critic, GAE are
  in the course material). They need data for a presentation: keep
  `results/` complete (`python -m parkour.report` refreshes it).
- Repository: <https://github.com/taraskHy/RL-Minecraft-project>, branch
  `main`; pushing works from the PC (the user signed in there). Never
  force-push. Push and update TEAM_STATUS after each result.

## 2. Current state (28 Oct 2026)

Training results up to the PC's last commit (`a7a426c`), plus the CPU
results of the cloud session on branch `claude/pensive-knuth-biwj11`
(planner retuning). Every number below is in `results/README.md`,
`results/neo/` or `results/planner/`, and says where it was measured.

| What | File / run | Simulator (1,200 held-out courses) | Real Minecraft 1.21.4 |
| --- | --- | --- | --- |
| **Final policy** (discount 0.995) | `results/final_policy.pt` / `.json` = `runs/obstacles-g995` best, iteration 699 (validation 97.6) | 98.4% standard (level 5: 93.5), 95.2% obstacles (level 5: 81.0) | 59/60 standard, 57/60 obstacles, 3/4 hand-designed (the miss is the neo course, not in its training); 0 server corrections; same outcome as the simulator on every course |
| First final policy (discount 0.99) | `results/final_policy_v1.*` = `runs/obstacles-ft` | 97.1%, 93.6% | 59/60, 58/60, 3/3 |
| **New map, no course list**: greedy route planner + final policy | `parkour/planner.py` = `bridge/planner.js` | 90.9% standard, 88.1% obstacles; **retuned** (`cost=distance, align=2.0`, chosen on validation): 92.3%, 90.5% | 50/60, 55/60 (retuned: not run yet; offline bot 51/60, 56/60) |
| Learned no-route model (research comparison) | `runs/noroute-v2` | 85.4%, 70.9% | 55/60, 37/60 |
| **Neo policy** (imitation + 3 DAgger rounds) | `runs/neo-dagger3` best, **only on the PC** (not in Git) | 61/210 (29.0%) held-out level 6-8 courses with a neo; 97/224 neo jumps passed | 11/31 neo course set (`configs/neo_live_courses.json`), same as offline and the simulator |

More facts that hold now:

- Every level 0-5 jump type is made from a standing start (level 5 mean
  99.1%); course styles outside training 93.0-98.3% (`results/stress.json`).
- Neo results by neos per course (`results/neo/by_neo_count.json`): one neo
  56/129, two 3/47, three 2/25, four or more 0/9.
- Neo history on the 210 courses (`results/neo/course_eval.json`,
  `round3_eval.json`): stall penalty 0 -> imitation 49 -> DAgger 1: 62 ->
  DAgger 2: 63 (best) -> DAgger 3: 61. Round 3 is the reported one because
  it was better on the validation neo courses (75 vs 72 of 218). DAgger 2's
  *last* checkpoint made 74/210, but picking it would be selecting on the
  test set.
- Double and triple neos: no landing in 64 searched layouts; treated as
  impossible (search evidence, not a proof).
- Physics port vs. the real library: 1.4e-14 blocks over 99,000 ticks
  (`results/physics_parity.json`).
- `final-v2` (`results/runs/final-v2`) is an abandoned run: from scratch
  with obstacles on 30% of jumps and a learning-rate cap of 3e-4 (the main
  run used 1e-3), started 27 Oct 22:53, stopped after 534 of 3,000
  iterations (44 minutes; last evaluation at 499: 88.5% standard, 88.1%
  obstacles). No result uses it; why it stopped is not recorded (it ended
  about when the neo levels were added, 23:25).
- Two policies, not one: the final policy for levels 0-5 and obstacles, the
  neo policy for levels 6-8. The neo policy is close to being both
  (`results/neo/one_policy_comparison.json`, simulator, identical courses:
  levels 0-5 of the advanced test set are the standard test courses): 96.5%
  standard and 92.5% obstacles at levels 0-5 (final policy: 98.4, 95.2),
  while the final policy completes 0 of the 210 neo courses (levels 6/7/8:
  71.0/3.5/1.5% against the neo policy's 80.5/46.5/71.5%).
- What is running on the PC right now cannot be known from outside it: the
  last commits record finished results only. Check the processes there.

## 3. What is left to do

Concrete work, most important first. Nothing here needs new research.

1. **Merge `claude/pensive-knuth-biwj11` into `main`** (PC session; the
   cloud session's work: docs, the handoff tools, planner options, analyses,
   new files in `results/`). `git fetch origin && git merge
   origin/claude/pensive-knuth-biwj11`, no rebase. If `main` has moved:
   `results/README.md` is generated, so after the merge run `run.cmd report`
   instead of resolving it by hand (it also rewrites this file's snapshot);
   for `HANDOFF.md` keep both sides' hand-written text and add both session
   log entries; then `run.cmd test` and push. The branch adds opt-in
   features to the physics, courses, training, planner and both bots (ice,
   planner options), all off by default and covered by tests (42 pass);
   the only default behaviour changes are the GPU watchdog also pausing
   `parkour.neo_dagger` and `run.cmd report` also refreshing this file's
   snapshot and writing the new tables in `results/README.md`.
2. **Confirm the retuned planner in real Minecraft** (PC; section 4,
   item 4): `run.cmd live --checkpoint results\final_policy.pt --planner
   --planner-config "{\"cost\": \"distance\", \"align\": 2.0}"
   --courses-file results\games\v2-planner-live\courses.json --out
   runs\v2-tuned-planner-live` (and the same with
   `v2-planner-live-obstacles`). If it holds, make it the default in both
   `parkour/planner.py` (`PlannerConfig`) and `bridge/planner.js`
   (`DEFAULTS`), keep `tests/test_planner.py` passing, and update report
   7.11 and 9, `README.md` and `TEAM_STATUS.md`.
3. **Report** (`docs/report_draft.md`, for the course): section 7 is
   complete (7.3-7.9 report the final policy with the first one in
   brackets; 7.10 has the imitation + DAgger results; 7.11-7.12 give both
   policies' planner numbers); sections 8 (discussion) and 9 (conclusion)
   are written; figures 1-4 are embedded (from `results/figures/`).
   `scripts/report_docx.py --pdf` writes `results/report_draft.docx` / `.pdf`
   in the submission format (needs pandoc, python-docx, LibreOffice; not on
   the PC by default): **20 pages before the references, exactly the
   limit**, so any addition needs a cut (figure 3 of the main run's
   training curves was already cut for space). Rules for it: every number
   comes from `results/` or the run files; Arial 12, 1.5 spacing,
   one-inch margins, 8 to 20 pages without the references. Still to do: a
   read-through by the team.
4. **Ship the neo policy in Git** (it is nearly a one-policy candidate,
   section 4, item 2): it exists
   only as `runs/neo-dagger3/checkpoints/best.pt` on the PC. Export it next
   to the final policy (for example `results/neo_policy.pt` / `.json`)
   without replacing `results/final_policy.*`.
5. **Restart the GPU watchdog on the PC** after merging this branch: it
   now also pauses `parkour.neo_dagger`, but the copy already running uses
   the old job list.
6. Keep `TEAM_STATUS.md` and this file in step with each new result.

## 4. Open problems (unsolved, up for grabs)

1. **Neo jumps are still mostly failed.** 29% of neo courses; courses with
   two or more neos almost never (5/81). DAgger round 3 beat round 2 on
   validation but not on the test courses (61 vs 63 of 210). Alone from a
   standing start (`results/neo/jump_eval_neo-dagger3.json`, 16 angles)
   the chosen policy no longer refuses (stalls 0-12% per type); it now
   tries and falls. Weakest types: gap 2 down 1 with the wall at gap cell
   2 (19%), then gap 2 up 1 (wall cell 2), gap 3 down 1 (wall cell 1 or 2)
   at 44%; best 81%. So more DAgger states on those layouts is the obvious
   next step.
   Untried directions: more DAgger rounds with more arrival states, a
   separate neo skill the main policy hands over to, a curriculum of neo
   tightness. Keep picking checkpoints on the validation family.
2. **One policy for everything:** the neo policy is 1.9 (standard) and 2.7
   (obstacles) points behind the final policy on levels 0-5, mostly at level
   5 (85.5 vs 93.5%, 70.5 vs 81.0%), and far ahead on levels 6-8 (section 2).
   It is built on the discount-0.99 runs; the obvious try is the same neo
   recipe (imitation + DAgger, `parkour/neo_dagger.py`) starting from the
   discount-0.995 final policy (GPU). Not yet done either: the neo policy on
   the 60 + 60 real-Minecraft courses and the stress tests (PC).
3. **Hardest level:** level 5 is 93.5% standard and 81.0% with obstacles;
   most remaining failures of the final policy are there. Where exactly
   (`scripts/jump_failures.py` -> `results/jump_failures.json`, simulator,
   test sets): the 19 standard failures are all falls, 13 at level 5,
   spread over many jump types (none fails more than 3 times); 12 of them
   are at the 7th to 9th jump, but on 3,000 extra level-5 courses (seed
   family 11, analysis only, `scripts/level5_positions.py` ->
   `results/level5_by_position.json`) the failure rate is flat along the
   course (0.8-1.4% per jump), so that was noise. What the larger sample
   shows: 94% of the failures are on level-5 jumps, and those fail in
   0.7% of tries as the first jump (from a standing start) but 1.4-2.6%
   later in the course: the hardest jumps are harder when the bot arrives
   moving. Making it stop does not help: pausing 3-10 ticks after each
   landing lowers level-5 success from 90.2% to 70-80%
   (`scripts/level5_pause_test.py`, `results/level5_pause_test.jsonl`). Of the 57 obstacle failures,
   17 are on a jump under a head-hitter beam (12 of the 38 at level 5).
   Worst case: a 5-block gap down 3 with a 2-block sideways offset onto
   one block fails in 5 of its 10 attempts under a beam, 3 of 27 without.
   The beam rule (`configs/jump_beam.json`) only asks a jump to keep its
   level, and level 5 has no lower bound, so a beam can make a level-5
   jump several times less forgiving (this one: 1.2% to 0.27% of scripted
   strategies). Do not change the generator (the test sets are fixed); a
   training-side fix (more such beams in training) is the natural try.
4. **Planner shortcuts:** most planner failures take a shortcut to a later
   platform that is in reach, and that jump is harder
   (`results/planner/`, report 7.11). With the final policy 98 of the 109
   failed standard courses are route failures (jump failures fell from 21
   to 11, route failures only from 104 to 98), so the planner, not the
   policy, is what limits the "no course list" score. Note that leaving
   the course's order is usually fine: on validation 327 of the 367 runs
   that left it still succeeded. **Tried on 28 Oct (cloud session, CPU,
   validation only, `results/planner/validation_tuning.jsonl`):**
   critic-chosen routes (`value_pick`, opt-in, simulator only) are worse
   (84.7-87.7% vs 92.5%); a drop penalty (`drop`), a one-step lookahead
   (`lookahead`: +0.5 standard, -0.6 obstacles) and retuning `align` alone
   are within noise; hard limits on drop height break the planner
   (courses have real 3-block drops). **What worked:** `cost=distance,
   align=2.0`: validation 92.8 vs 91.3% and 89.0 vs 87.4% (1,800 courses
   each), test 92.3% and 90.5% (from 90.9 and 88.1), offline bot 51/60 and
   56/60 (from 50 and 55), identical to the simulator. Still open: about 80
   route failures per 1,200 courses remain, and the fall rule's artefact
   on long drops (the "remaining" rule scores about 2 points higher).
5. **Learned no-route model and obstacles:** 10 of its 23 obstacle-course
   falls end on top of an obstacle (6 on a beam, 4 on a decoy pillar).
6. **More block types:** slabs, ladders, ice, fences and water need physics
   ports (prismarine-physics has them), parity tests and course pieces.
   **Ice is ported (28 Oct, cloud session):** `physics.World(solid, slip)`
   takes an optional per-cell slipperiness grid (ice 0.98), read under the
   feet centre as prismarine-physics does; without it nothing changes. Parity
   with the real library on fields where half the blocks are ice: 7.1e-15
   blocks over 60,000 ticks, every flag equal (`results/physics_parity_ice.json`,
   `tests/test_physics_parity.py::test_ice_cpu`); ice changes 297 of 300 of
   those trajectories. Still needed before a policy can use ice: ice course
   pieces and training are now in place too, all opt-in:
   `generate(ice_prob=...)` / `Course.ice` (platform indices; the layout and
   course files are unchanged without it), `CoursePool` keeps the ice grid
   through refreshes, `run.cmd train --ice-prob P`, the live bot builds
   `minecraft:ice` (its physics then uses ice friction by itself), and the
   offline bot plays ice courses exactly like the simulator
   (`tests/test_ice.py`). Next (GPU): fine-tune the final policy with
   `--ice-prob 0.3` (and the final policy's other settings: `--gamma 0.995
   --obstacle-prob 0.3`), evaluate with `scripts/ice_preview.py
   --checkpoint ...` and on the normal test sets; later an observation of
   the block type (the 255-number observation cannot see ice; a new
   observation needs `bridge/policy.js` changed in step). A first measurement (`scripts/ice_preview.py` ->
   `results/ice_preview.json`, simulator): the final policy, blind to ice,
   still completes 95.6% of the standard test courses when every platform
   is ice (98.4% on stone); level 5 drops from 93.5% to 86.0%, falls there
   from 6.5% to 13%. So ice is a cheap next block type: most of the skill
   carries over, and the gap is at the hardest jumps.
7. **Hand-built maps:** extract the platform chain from a region of a real
   map instead of generating courses.
8. Double and triple neos: reopen only with a new search method; the
   current search found none (section 2).

## 5. Do not touch: use as a black box

These parts were verified against an outside reference (the real physics
library, the real game, fixed test sets). Changing them silently invalidates
results that are already reported. Use them through their commands; if a
change is unavoidable, make it an opt-in flag, run the listed tests, and
re-run the offline and real-Minecraft checks.

| Part | Why | Guarded by |
| --- | --- | --- |
| `parkour/physics.py` | exact float64 port of prismarine-physics 1.11.1 (operand order kept) plus vanilla's 1e-7 `COLLISION_EPSILON`; the optional ice grid (`World(slip=...)`) is off unless passed | `tests/test_physics_parity.py` (also with ice), `test_corner_slip.py`, `test_inside_block.py` |
| `bridge/physics_patch.js` | the same tolerance in the bot; must change together with `physics.py` | `tests/test_physics_parity.py` |
| `bridge/policy.js` <-> `parkour/env.py` (observation, fall and stall rules), `bridge/planner.js` <-> `parkour/planner.py` | the bot mirrors the Python code; the 255-number observation is what every shipped policy expects | `tests/test_bridge_parity.py`, `test_extended_obs.py`, `test_planner.py` |
| Seed families in `parkour/train.py` (training 0, test 1, validation 2, stress 3, obstacle test 4, obstacle validation 5) and default course generation in `parkour/courses.py` | every test number is on exactly these courses (`course_hash`); new course features are opt-in (like `wide_neo`) | `tests/test_courses.py` |
| `configs/jump_levels*.json`, `configs/jump_feasibility*.json` | outputs of the feasibility study; they define the difficulty levels | regenerate with `run.cmd feasibility`, never edit by hand |
| `results/final_policy.*`, `results/final_policy_v1.*` | shipped policies quoted everywhere | a new policy gets a new file; the old one is kept (`_v1` pattern) |
| `results/runs/`, `results/games/`, `results/README.md` | archived evidence and generated tables | written by `parkour.report` and `parkour.live`; change `parkour/report.py`, not the output |
| Policy JSON format `parkour-policy-v1` (`parkour/model.py`) | what `bridge/policy.js` loads | `tests/test_bridge_parity.py` |
| `RewardConfig` and PPO defaults in `parkour/env.py` / `parkour/train.py` | the ablation and hyperparameter tables are relative to these defaults | add flags (`--reward-stall`, `--gamma`) instead of changing defaults |
| `bridge/package.json` / `pnpm-lock.yaml` (Mineflayer 4.39.0, minecraft-data 3.117.0) | the bot's sprint speed depends on the installed minecraft-data (`docs/design.md` section 2) | the live replays (2e-14 blocks) |
| Live bot details in `bridge/parkour_bot.js` | `/tp` with decimals (whole numbers are centred), deleting the movement-speed attribute each tick, setting `bot.entity.yaw` directly | the live runs (0 server corrections) |
| `.local/` runtime (`setup.cmd`, `parkour/runtime.py`) | portable Node, Java and server jar with verified checksums; ignored by Git | `setup.cmd` |

How to use the black boxes without reading them (on the PC):

```text
run.cmd test                                             all checks, physics parity included
run.cmd train --run NAME [--init-from CKPT] [--gamma 0.995] ...   PPO training; flags = TrainConfig fields
run.cmd offline --checkpoint CKPT --out runs\NAME        the bot's JavaScript on prismarine-physics, no server
run.cmd live --checkpoint CKPT --out runs\NAME [--viewer NAME]   real Minecraft (EULA accepted on this PC)
run.cmd live ... --planner [--planner-config JSON]       no course list: the route planner picks the platforms
python -m parkour.planner --checkpoint results/final_policy.pt [--obstacles] [--family 2]   tune on 2, report 1
python -m parkour.jumpeval --checkpoint CKPT --out FILE [--advanced]   every jump type alone
python -m parkour.stress --checkpoint CKPT --out FILE    course styles outside training
run.cmd report                                           refresh results/ (and this file's snapshot)
python scripts/report_docx.py --pdf                      the report as .docx/.pdf, with its page count
```

## 6. Machine and environment

**Training PC:** Intel Core Ultra 7 270K Plus (24 cores), 128 GB RAM, RTX
5080 16 GB, Windows 11, clock in Israel time (UTC+2; Git Bash labels it JST).
Checkout: `C:\Users\ethan\OneDrive\Desktop\etgar\RL\RL-Minecraft-parkour`
(OneDrive folder). Python 3.12 venv in `.venv` (torch 2.11.0+cu128).
Portable Node, Java and the 1.21.4 server jar in `.local` (ignored by Git).
The Minecraft EULA was accepted by the user on 27 Oct 2026
(`.local/minecraft-eula-consent.json`). `runs/` (checkpoints, full logs)
exists only there.

**GPU memory is the limit:** two training processes fit (since 28 Oct, with
the desktop apps open); a third spills into system RAM and everything slows
5-10 times. Windows sleeps after 15 idle minutes: `scripts/keep_awake.py`
holds it awake while training runs.

**GPU temperature rule (user, 28 Oct):** above 65 C pause the GPU work,
resume when cool. `scripts/gpu_watchdog.py` runs in the background: every
10 s it reads nvidia-smi; above 65 C it suspends the project's GPU jobs by
name (train, neo_demos, neo_dagger, cem, planner, jumpeval, stress,
feasibility, cem_momentum_neo; suspended, so no state is lost) and resumes
them below 58 C. `neo_dagger` was added on 28 Oct afternoon: a watchdog
started before that does not pause it until restarted. The live bot is left
alone on purpose (it plays in real time). Log: `runs/gpu_watchdog.log`. If the watchdog is ever killed while jobs are
paused: `python scripts/gpu_watchdog.py --resume-all`.

**Anywhere else** (a cloud container, another model's sandbox): no GPU,
no `runs/`, no Minecraft server, but the test suite and CPU evaluations
(planner, jump and stress checks with the checkpoints in `results/`) do
run. Setup used on 28 Oct in a Linux cloud container (Python 3.13, Node 22):

```text
python3 -m venv VENV && VENV/bin/pip install torch==2.11.0 numpy==2.5.4 matplotlib==3.11.2 pytest==9.1.1
cd bridge && CI=true npx -y pnpm@11.25.0 install --frozen-lockfile --ignore-scripts --store-dir ../.local/pnpm-store
mkdir -p .local/runtime/node && ln -s "$(which node)" .local/runtime/node/node.exe   # tests look for node.exe
VENV/bin/python -m pytest tests -q        # 36 passed, 1 skipped (CUDA) on 28 Oct
```

PyPI's torch wheel is the CUDA build (about 3 GB with its dependencies)
but runs on the CPU; `download.pytorch.org` was blocked there. Training
needs the PC's GPU.

## 7. Key facts learned

- Physics: prismarine-physics 1.11.1 ported exactly, **plus vanilla's 1e-7
  collision tolerance** in both the simulator (`COLLISION_EPSILON`) and the
  bot (`bridge/physics_patch.js`). Without it a player resting against a face
  can slip into a block corner through rounding.
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
- Discount 0.995 beat the defaults on validation in all three seeds (95.4%
  vs 91.8%); the only hyperparameter that cleared the 3-point bar.
- Neos: a single neo (wall flush with a 1-wide takeoff) is possible in 16 of
  the 18 layouts tried (none for gap 3, dy +1 with the wall at gap cell 1
  or 3). The default course generator (all test and validation sets) places
  a neo only after a 1-wide takeoff; with `wide_neo` (hand-made courses,
  `train --wide-neo`) a wider takeoff is allowed if the landing reaches the
  wall's open end. Standing on any obstacle ends an episode as a fall.
- Neos are not found by exploration: with the fall penalty the bot refuses
  (stands until the stall limit); a stall penalty turned refusals into falls,
  not landings. What worked: imitation of 12,800 of 50,706 search
  demonstrations (`parkour/neo_demos.py`), then DAgger with a search expert
  from the bot's own arrival states (`parkour/neo_dagger.py`).
- DAgger labels must be clean: the search starts from the policy's own
  action probabilities (`--prior`) and is simplified to the corrections that
  matter (`--simplify`); a round without this collapsed
  (`runs/neo-dagger-collapsed`). Neo takeoffs are detected by wall blocks
  only (the first detector counted ordinary up-jumps: 1,090 of 2,526 were
  real neos).
- `train.py`: `bc_decay` defaults to False, as in every neo run (the
  imitation weight was constant).

## 8. Where things are

- Final policy: `results/final_policy.pt` / `.json`; the first one:
  `results/final_policy_v1.*`.
- Runs: `runs/<name>/` on the PC (ignored by Git; logs copied to
  `results/runs/` by `parkour.report`).
- Neo work: `runs/advanced-ft3` (levels 0-8 fine-tune), `runs/neo-stall`,
  `runs/neo-bc`, `runs/neo-dagger`, `-dagger2`, `-dagger3`; evaluations and
  evidence in `results/neo/` (written by `scripts/neo_evidence.py`).
- Route planner results: `results/planner/` (`*_tuned.json`: the retuned
  setting; `validation_tuning.jsonl`: every variant tried on validation)
  and the `planner-*`, `v2-planner-*`, `v2-tuned-planner-*` games.
- Report in the submission format: `results/report_draft.docx` / `.pdf`,
  written by `scripts/report_docx.py --pdf` from `docs/report_draft.md`.
- Failure analysis of the final policy by jump type: `results/jump_failures.json`
  (`scripts/jump_failures.py`).
- Claim audit: `results/audit_claims.json` (829 claims checked against the
  run files on 28 Oct).
- Sweeps: `scripts/sweep.py configs/<sweep>.json --slots N` (skips finished
  runs).
- Pages: `results/theatre.html`, `results/replay_viewer.html`,
  `results/progress.html`, `results/neo_preview.html`,
  `results/showcase.html`, `results/overnight_plan.html`. Videos:
  `results/videos/` (`scripts/render_videos.py`, `scripts/film_pov.py`).

## 9. Snapshot (generated)

Do not edit between the markers: `python -m parkour.handoff` rewrites it from
Git and `results/` (the report step runs it too).

<!-- BEGIN AUTO-SNAPSHOT: written by `python -m parkour.handoff`; edits inside are overwritten -->

Generated on top of commit `40df460` (2026-10-28 14:49, branch `claude/pensive-knuth-biwj11`): Ice courses, opt-in end to end: generation, pool, training flag, both bots

| Document | Last change | Commits since |
| --- | --- | --- |
| `AGENTS.md` | `85b8f50` 2026-10-28 14:27 | 7 |
| `HANDOFF.md` | updated together with this snapshot | 0 |
| `TEAM_STATUS.md` | `6228744` 2026-10-28 14:37 | 3 |
| `README.md` | `5b0af09` 2026-10-28 13:05 | 23 |
| `docs/design.md` | `8da16f5` 2026-10-28 14:31 | 6 |
| `docs/report_draft.md` | `12c736a` 2026-10-28 14:26 | 8 |
| `results/README.md` | `8c5bdfa` 2026-10-28 14:36 | 5 |
| `results/DATA.md` | `f405271` 2026-10-28 14:45 | 1 |

<!-- snapshot data: compared by `python -m parkour.handoff --check` -->

### Shipped policies

Simulator, held-out test sets (1,200 courses each), checkpoint picked on the validation family.

| File | Checkpoint | Iteration | Validation | Test, standard | Test, obstacles |
| --- | --- | --- | --- | --- | --- |
| `results/final_policy.pt` / `.json` | `runs/obstacles-g995/checkpoints/best.pt` | 699 | 97.6 | 98.4 | 95.2 |
| `results/final_policy_v1.pt` / `.json` | `runs/obstacles-ft/checkpoints/best.pt` | 999 | 96.6 | 97.1 | 93.6 |

### Training runs (`results/runs/*/summary.json`)

Simulator, held-out test set. "Picked" is the validation-selected checkpoint (`best.pt`); runs without one predate validation selection. Runs trained on levels 0-8 (`advanced-*`, `neo-*`) are evaluated on levels 0-8, the others on levels 0-5, so their overall numbers differ in kind; per level they compare (levels 0-5 are the same courses in both).

| Run | Last iteration | Game ticks (B) | Picked: iteration | validation | test | Last: test | Last: obstacles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `advanced-ft` | 799 | 0.42 | 699 | 69.2 | 71.2 | 70.9 | 67.6 |
| `advanced-ft2` | 299 | 0.16 | 249 | 81.6 | 83.0 | 82.8 | 78.6 |
| `advanced-ft3` | 599 | 0.31 | 499 | 82.7 | 83.6 | 83.5 | 80.6 |
| `final-v2` | 499 | 0.26 | 499 | 89.6 | 88.5 | 88.5 | 88.1 |
| `full-s1` | 999 | 0.52 | - | - | - | 94.2 | - |
| `full-s2` | 999 | 0.52 | 949 | 93.0 | 93.1 | 91.9 | - |
| `hp-ent0.001` | 999 | 0.52 | 999 | 94.3 | 94.1 | 94.1 | - |
| `hp-ent0.02` | 999 | 0.52 | 549 | 91.7 | 92.2 | 91.5 | - |
| `hp-envs4096` | 3999 | 0.52 | 3599 | 93.7 | 94.2 | 92.0 | - |
| `hp-gamma0.98` | 999 | 0.52 | 449 | 90.5 | 90.4 | 90.4 | - |
| `hp-gamma0.995` | 999 | 0.52 | 749 | 95.7 | 94.3 | 95.0 | - |
| `hp-gamma0.995-s1` | 999 | 0.52 | 949 | 95.8 | 95.9 | 94.9 | - |
| `hp-gamma0.995-s2` | 999 | 0.52 | 799 | 95.7 | 96.0 | 95.6 | - |
| `hp-kl0.005` | 999 | 0.52 | 849 | 91.7 | 91.7 | 90.8 | - |
| `hp-kl0.02` | 999 | 0.52 | 899 | 92.7 | 92.2 | 91.8 | - |
| `hp-net-large` | 999 | 0.52 | 649 | 94.5 | 95.3 | 95.4 | - |
| `hp-net-small` | 999 | 0.52 | 999 | 92.5 | 91.3 | 91.3 | - |
| `main-g995` | 2999 | 1.57 | 2799 | 98.7 | 97.9 | 96.2 | - |
| `main-s0` | 2999 | 1.57 | - | - | - | 95.5 | - |
| `neo-bc` | 799 | 0.42 | 699 | 86.1 | 85.7 | 86.5 | 83.1 |
| `neo-dagger` | 499 | 0.26 | 149 | 86.1 | 85.9 | 85.1 | 82.3 |
| `neo-dagger-collapsed` | 499 | 0.26 | 499 | 57.3 | 58.9 | 58.9 | 56.8 |
| `neo-dagger2` | 499 | 0.26 | 299 | 86.2 | 86.7 | 87.3 | 83.3 |
| `neo-dagger3` | 499 | 0.26 | 449 | 85.7 | 86.4 | 87.1 | 84.4 |
| `neo-stall` | 999 | 0.52 | 749 | 84.7 | 84.7 | 84.8 | 82.3 |
| `nofall-s0` | 999 | 0.52 | 949 | 93.7 | 91.8 | 91.2 | - |
| `nofall-s1` | 999 | 0.52 | 649 | 90.5 | 90.8 | 90.9 | - |
| `nofall-s2` | 999 | 0.52 | 799 | 91.2 | 90.2 | 90.5 | - |
| `nolanding-s0` | 999 | 0.52 | 599 | 91.7 | 89.8 | 90.0 | - |
| `nolanding-s1` | 999 | 0.52 | 799 | 95.3 | 94.6 | 94.2 | - |
| `nolanding-s2` | 999 | 0.52 | 849 | 92.5 | 90.4 | 91.0 | - |
| `noroute-s0` | 999 | 0.52 | 949 | 86.2 | 82.5 | 83.6 | - |
| `noroute-s1` | 999 | 0.52 | 799 | 83.3 | 80.2 | 83.2 | - |
| `noroute-s2` | 999 | 0.52 | 999 | 83.3 | 80.6 | 80.6 | - |
| `noroute-v2` | 2999 | 1.57 | 2849 | 83.2 | 85.4 | 85.0 | 73.4 |
| `obstacles-ft` | 999 | 0.52 | 999 | 96.6 | 97.1 | 97.1 | 93.6 |
| `obstacles-g995` | 999 | 0.52 | 699 | 97.6 | 98.4 | 97.5 | 94.7 |
| `progress-demo` | 399 | 0.21 | 324 | 89.5 | 89.8 | 91.0 | - |
| `sparse-s0` | 999 | 0.52 | - | - | - | 95.1 | - |
| `sparse-s1` | 999 | 0.52 | 749 | 93.5 | 93.2 | 91.5 | - |
| `sparse-s2` | 999 | 0.52 | 799 | 94.7 | 94.4 | 94.1 | - |
| `uniform-s0` | 999 | 0.52 | 599 | 95.3 | 94.8 | 95.0 | - |
| `uniform-s1` | 999 | 0.52 | 899 | 93.5 | 92.9 | 93.0 | - |
| `uniform-s2` | 999 | 0.52 | 949 | 94.2 | 92.8 | 93.9 | - |

### Games played by the bot (`results/games/*/summary.json`)

| Folder | Where | Checkpoint | Completed | Simulator, same courses | Same outcome |
| --- | --- | --- | --- | --- | --- |
| `cheat-check` | offline (prismarine-physics, no server) | `runs/neo-ck.pt` | 0/1 | 0.0 | 100.0 |
| `film-neo_level7-1028-112702` | live (Minecraft 1.21.4 server) | `runs/neo-dagger3/checkpoints/best.pt` | 1/1 | 100.0 | 100.0 |
| `film-sprint-1028-014252` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 1/1 | 100.0 | 100.0 |
| `film-sprint-1028-024939` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 1/1 | 100.0 | 100.0 |
| `film-stairs-1028-014205` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 1/1 | 100.0 | 100.0 |
| `film-stairs-1028-024852` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 1/1 | 100.0 | 100.0 |
| `film-tour-1028-014014` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 1/1 | 100.0 | 100.0 |
| `film-tour-1028-014119` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 1/1 | 100.0 | 100.0 |
| `film-tour-1028-024805` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 1/1 | 100.0 | 100.0 |
| `final-offline` | offline (prismarine-physics, no server) | `runs/obstacles-ft/checkpoints/best.pt` | 59/60 | 98.3 | 100.0 |
| `final-offline-obstacles` | offline (prismarine-physics, no server) | `runs/obstacles-ft/checkpoints/best.pt` | 58/60 | 96.7 | 100.0 |
| `final-showcase` | offline (prismarine-physics, no server) | `runs/obstacles-ft/checkpoints/best.pt` | 3/3 | 100.0 | 100.0 |
| `live-final` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 59/60 | 98.3 | 100.0 |
| `live-final-obstacles` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 58/60 | 96.7 | 100.0 |
| `live-rainbow-test` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 3/4 | 75.0 | 100.0 |
| `live-showcase` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 3/3 | 100.0 | 100.0 |
| `live-smoke` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 6/6 | 100.0 | 100.0 |
| `live2-final` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 59/60 | 98.3 | 100.0 |
| `live2-obstacles` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 58/60 | 96.7 | 100.0 |
| `live2-showcase` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 3/4 | 75.0 | 100.0 |
| `neo-full` | offline (prismarine-physics, no server) | `runs/neo-ck.pt` | 1/31 | 3.2 | 100.0 |
| `neo-live-final` | live (Minecraft 1.21.4 server) | `runs/neo-dagger3/checkpoints/best.pt` | 11/31 | 35.5 | 100.0 |
| `neo-offline-final` | offline (prismarine-physics, no server) | `runs/neo-dagger3/checkpoints/best.pt` | 11/31 | 35.5 | 100.0 |
| `neo-preview__bot` | offline (prismarine-physics, no server) | `runs/advanced-preview.pt` | 0/13 | 0.0 | 100.0 |
| `noroute-live` | live (Minecraft 1.21.4 server) | `runs/noroute-v2/checkpoints/best.pt` | 55/60 | 91.7 | 100.0 |
| `noroute-live-obstacles` | live (Minecraft 1.21.4 server) | `runs/noroute-v2/checkpoints/best.pt` | 37/60 | 61.7 | 100.0 |
| `noroute-offline` | offline (prismarine-physics, no server) | `runs/noroute-v2/checkpoints/best.pt` | 55/60 | 91.7 | 100.0 |
| `noroute-offline-obstacles` | offline (prismarine-physics, no server) | `runs/noroute-v2/checkpoints/best.pt` | 37/60 | 61.7 | 100.0 |
| `offline-final` | offline (prismarine-physics, no server) | `runs/main-s0/checkpoints/latest.pt` | 58/60 | 96.7 | 100.0 |
| `offline-showcase` | offline (prismarine-physics, no server) | `runs/main-s0/checkpoints/best.pt` | 3/3 | 100.0 | 100.0 |
| `offline-showcase-final` | offline (prismarine-physics, no server) | `runs/main-s0/checkpoints/latest.pt` | 3/3 | 100.0 | 100.0 |
| `offline-test` | offline (prismarine-physics, no server) | `runs/main-s0/checkpoints/best.pt` | 28/30 | 93.3 | 100.0 |
| `planner-live` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 52/60 | 86.7 | 100.0 |
| `planner-live-obstacles` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 53/60 | 88.3 | 100.0 |
| `planner-offline` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 52/60 | 86.7 | 100.0 |
| `planner-offline-obstacles` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 53/60 | 88.3 | 100.0 |
| `progress-demo-games__live_00000` | live (Minecraft 1.21.4 server) | `runs/progress-demo/checkpoints/updates_00000.pt` | 0/3 | 0.0 | 100.0 |
| `progress-demo-games__live_00010` | live (Minecraft 1.21.4 server) | `runs/progress-demo/checkpoints/updates_00010.pt` | 0/3 | 0.0 | 100.0 |
| `progress-demo-games__live_00050` | live (Minecraft 1.21.4 server) | `runs/progress-demo/checkpoints/updates_00050.pt` | 3/3 | 100.0 | 100.0 |
| `progress-demo-games__live_00400` | live (Minecraft 1.21.4 server) | `runs/progress-demo/checkpoints/updates_00400.pt` | 3/3 | 100.0 | 100.0 |
| `progress-demo-games__offline_00000` | offline (prismarine-physics, no server) | `runs/progress-demo/checkpoints/updates_00000.pt` | 0/7 | 0.0 | 100.0 |
| `progress-demo-games__offline_00002` | offline (prismarine-physics, no server) | `runs/progress-demo/checkpoints/updates_00002.pt` | 0/7 | 0.0 | 100.0 |
| `progress-demo-games__offline_00005` | offline (prismarine-physics, no server) | `runs/progress-demo/checkpoints/updates_00005.pt` | 0/7 | 0.0 | 100.0 |
| `progress-demo-games__offline_00010` | offline (prismarine-physics, no server) | `runs/progress-demo/checkpoints/updates_00010.pt` | 0/7 | 0.0 | 100.0 |
| `progress-demo-games__offline_00020` | offline (prismarine-physics, no server) | `runs/progress-demo/checkpoints/updates_00020.pt` | 1/7 | 14.3 | 100.0 |
| `progress-demo-games__offline_00035` | offline (prismarine-physics, no server) | `runs/progress-demo/checkpoints/updates_00035.pt` | 6/7 | 85.7 | 100.0 |
| `progress-demo-games__offline_00050` | offline (prismarine-physics, no server) | `runs/progress-demo/checkpoints/updates_00050.pt` | 6/7 | 85.7 | 100.0 |
| `progress-demo-games__offline_00100` | offline (prismarine-physics, no server) | `runs/progress-demo/checkpoints/updates_00100.pt` | 6/7 | 85.7 | 100.0 |
| `progress-demo-games__offline_00200` | offline (prismarine-physics, no server) | `runs/progress-demo/checkpoints/updates_00200.pt` | 6/7 | 85.7 | 100.0 |
| `progress-demo-games__offline_00400` | offline (prismarine-physics, no server) | `runs/progress-demo/checkpoints/updates_00400.pt` | 6/7 | 85.7 | 100.0 |
| `recheck-obs` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 58/60 | 96.7 | 100.0 |
| `recheck-show` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 3/4 | 75.0 | 100.0 |
| `recheck-std` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 59/60 | 98.3 | 100.0 |
| `v2-live` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 59/60 | 98.3 | 100.0 |
| `v2-live-obstacles` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 57/60 | 95.0 | 100.0 |
| `v2-live-showcase` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 3/4 | 75.0 | 100.0 |
| `v2-offline` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 59/60 | 98.3 | 100.0 |
| `v2-offline-obstacles` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 57/60 | 95.0 | 100.0 |
| `v2-offline-showcase` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 3/4 | 75.0 | 100.0 |
| `v2-planner-live` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 50/60 | 83.3 | 100.0 |
| `v2-planner-live-obstacles` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 55/60 | 91.7 | 100.0 |
| `v2-planner-offline` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 50/60 | 83.3 | 100.0 |
| `v2-planner-offline-obstacles` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 55/60 | 91.7 | 100.0 |
| `v2-tuned-planner-offline` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 51/60 | 85.0 | 100.0 |
| `v2-tuned-planner-offline-obstacles` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 56/60 | 93.3 | 100.0 |
| `viewer-test` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 1/1 | 100.0 | 100.0 |

### In `results/` but not yet in `results/DATA.md`

Nothing: the data index names everything.

<!-- END AUTO-SNAPSHOT -->

## 10. How to update this file

At the end of a session, or after each result:

1. Fix the hand-written sections that are no longer true (usually 2, 3 and
   4); keep every number traceable to a file.
2. Add one line at the top of the session log: date and time, who (model or
   person, and where), what changed, what is next.
3. Run `python -m parkour.handoff` (`run.cmd handoff`; `run.cmd report`
   does it too), then commit this file with the work it describes.
4. `python -m parkour.handoff --check` should then report it up to date.

## Session log

Newest first. One entry per session; earlier history is in the
[TEAM_STATUS update log](TEAM_STATUS.md#update-log) and `git log`.

- **28 Oct 2026, afternoon, Claude Code (cloud session, continued, branch
  `claude/pensive-knuth-biwj11`, CPU only):** the user asked this session
  to keep working on its own (no questions; they were away from the PC)
  and to push after every step. Setup: PyTorch from PyPI and the bridge
  packages in the container (section 6); the CPU simulator reproduces the
  published scores exactly (98.4% / 95.2%). Done:
  - Docs: `results/DATA.md` indexes everything; report section 7 on the
    final policy, 7.10 with imitation + DAgger, 7.11-7.12 with both
    policies, sections 8-9 written, figures embedded, introduction updated;
    `scripts/report_docx.py` builds it as .docx/.pdf (20 pages, the limit);
    `final-v2` documented as abandoned; README and TEAM_STATUS updated.
  - Route planner: retuned on validation (`cost=distance, align=2.0`):
    test 92.3% / 90.5% (from 90.9 / 88.1), offline bot 51/60, 56/60;
    critic-chosen routes (`value_pick`), a drop penalty and `align` alone
    did not help (`results/planner/validation_tuning.jsonl`). Defaults
    unchanged until the PC confirms it in real Minecraft.
  - Analyses: one policy for everything (the neo policy is 2-3 points
    behind on levels 0-5, `results/neo/one_policy_comparison.json`);
    failures by jump type (beams over level-5 jumps, `results/jump_failures.json`);
    level-5 failures by position (flat; `results/level5_by_position.json`);
    pausing after landings hurts (`results/level5_pause_test.jsonl`); the
    neo policy's weakest neo types (section 4, item 1).
  - Code: opt-in ice physics with parity 7.1e-15 blocks
    (`results/physics_parity_ice.json`), and ice courses end to end
    (generation, pool, `--ice-prob`, both bots; `tests/test_ice.py`); the
    final policy, blind to ice, completes 95.6% of all-ice test courses
    (`results/ice_preview.json`); `results/README.md` gains the
    retuned planner and one-policy tables (generator checked to reproduce
    the old page exactly); GPU watchdog also pauses `parkour.neo_dagger`.
  - Decisions: no Minecraft server here (the EULA is the user's to accept
    on each machine); no training (no GPU); nothing changes a default.
  - Next: section 3, items 1-2 (merge into `main`, then confirm the retuned
    planner in real Minecraft), then section 4 on the PC's GPU.
- **28 Oct 2026, Claude Code (cloud session, branch
  `claude/pensive-knuth-biwj11`):** turned this file into a living handoff
  for any model: current state with the discount-0.995 final policy and the
  DAgger neo results, what is left, open problems, a do-not-touch list;
  added `AGENTS.md` (+ `CLAUDE.md`), the generated snapshot
  (`parkour/handoff.py`, run by `parkour.report`) and its test; brought
  `TEAM_STATUS.md` up to date. Nothing that produces results changed
  (`parkour/report.py` only gained the snapshot refresh). Next: merge this
  branch into `main` (the PC session works there), then section 3.
- **28 Oct 2026, 12:34, main session on the PC (`a7a426c`):** evidence files
  for every neo and audit number (`results/neo/`, `results/audit_claims.json`).
