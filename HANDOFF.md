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
| **New map, no course list**: greedy route planner + final policy | `parkour/planner.py` = `bridge/planner.js` | 90.9% standard, 88.1% obstacles; **retuned** (`cost=distance, align=2.0`, chosen on validation): 92.3%, 90.5% | retuned: 51/60, 56/60, identical to offline and simulator, 0 corrections; now default (old configuration: 50/60, 55/60) |
| Learned no-route model (research comparison) | `runs/noroute-v2` | 85.4%, 70.9% | 55/60, 37/60 |
| **Neo policy** (imitation + 3 DAgger rounds) | `results/neo_policy.pt` / `.json`, exported from `runs/neo-dagger3` best and shipped in Git | 61/210 (29.0%) held-out level 6-8 courses with a neo; 97/224 neo jumps passed | 11/31 neo course set (`configs/neo_live_courses.json`), same as offline and the simulator |

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
- **New abilities (28 Oct evening, cloud session, CPU, simulator unless
  said otherwise; plan and status in `docs/abilities_plan.md`):**
  - *Neo assist* (`parkour/neo_assist.py`): when the bot first stands on a
    neo's takeoff, a cross-entropy search with the exact physics plans its
    actions from its exact state, and the bot plays the plan. Validation
    (218 held-out neo courses, `results/neo/assist_validation.jsonl`): the
    final policy alone completes 0 and passes 0 of the 140 neo jumps it
    reaches; with the assist it completes 49 and passes 131 of 179 reached
    neo jumps (every played plan landed as searched); 121 of its 169 failed
    courses stop at a jump that is not a neo (`failed_at_a_neo` 48).
    Planning the wide and momentum jumps too (`--jumps advanced`): 119 of
    218, 214 of 286 neos passed. These rows (tagged `code 587c9b1`) ran an
    earlier default that searched up to 32 states at once (batch 32: same
    method, other random draws); re-measured with batch 1, which the bots
    reproduce: 124 of 218, 221 of 289 neos passed, all 900 plans landed as
    searched (`results/neo/assist_details/official2_advanced.json`: the 94
    failed courses stop at 68 neos whose search found no landing, 22
    ordinary jumps the policy missed and 4 momentum jumps). A search prior
    from the policy (0.5) made it worse: 111 of 218. Planning every jump
    (`--jumps all`) with a stop and a second search after a failed one
    (`--retries 1 --brake 14 --retry-horizon 48`): 147 of 218, 246 of 311
    neos (`official2_all_brake.json`; 17 of 93 second searches found a
    landing). Random neo courses (family 13): the policy alone 27 of 200,
    planning every jump 106 (`random13_all.json`), with brake and retry 119,
    320 of 396 neos (`random13_all_brake.json`). Both bots
    have it (`parkour.live --assist '{"jumps": "advanced"}'`: the bot pauses
    on the takeoff, a Python planner service searches, prismarine-physics
    checks the plan, the bot plays it); the offline bot matches the
    simulator exactly when both search on the same device
    (`tests/test_neo_assist.py`). On the neo course set
    (`configs/neo_live_courses.json`, where the neo policy made 11/31) the
    final policy with the planner (every jump, brake and retry) makes 25/31
    in the offline bot, the same courses as the simulator
    (`results/games/neo-assist-offline-all-brake`). Not yet run in real
    Minecraft.
  - *Head hitters and ice run-ups*: new jump kinds (a run-up under a 2-high
    roof, a tunnel, an ice runway), feasibility-checked
    (`configs/jump_feasibility_extra.json`), in opt-in levels
    (`configs/jump_levels_extra.json`, `run.cmd train --extra`). Zero-shot,
    the final policy makes 68% / 22% / 49% of the head-hitter / hard
    head-hitter / ice kinds alone, and completes 25.5% / 1.5% / 7.0% of
    whole held-out courses of those levels; with the planner on every run-up
    jump, 31/40, 38/40, 32/40 (`results/extra/`).
  - *New blocks* (physics, opt-in, each exact against prismarine-physics
    over 60,000 ticks of random block fields, `results/physics_parity_*.json`):
    block shapes (`physics.World(shape=...)`: slabs, stairs, soul sand,
    dirt path, trapdoors, carpets, fence and wall posts, end rods, ladders;
    boxes from prismarine-block in `configs/block_shapes.json`), slime
    bounces (`World(slime=...)`), ladders and vines (`World(climb=...)`).
    In courses, as composite jump kinds that end on stone (shaped targets
    with an exit jump, slime pads with an exit above them, ladder climbs
    onto a wall): `JumpSpec.block`, `.bounce`, `.ladder`; the pool, the
    observation and its JavaScript mirror use the shapes' real tops and
    footprints, both bots build the real blocks, and the offline bot plays
    shaped, slime and ladder courses exactly like the simulator (tests).
    The route planner refuses shaped courses for now.
- What is running on the PC right now cannot be known from outside it: the
  last commits record finished results only. Check the processes there.

## 3. What is left to do

**PC completion update, 28 Oct 21:38:** cloud merged through `21188a6`; neo files shipped; retuned planner verified on both real-Minecraft sets (51/60 and 56/60, zero corrections) and adopted in Python/JS. Items 1, 2, 4 and 5 below describe the completed integration steps and ongoing merge/watchdog procedure. Training continues under the latest 70% compute/memory request.

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
   planner options, the neo assist, new jump kinds and blocks), all off by
   default and covered by tests (63 pass in the cloud container, CUDA test
   skipped; `tests/test_heldout_fingerprints.py` proves the held-out courses
   unchanged); the only default behaviour changes are the GPU watchdog also
   pausing `parkour.neo_dagger` and `parkour.neo_assist` (not the bots'
   planner service), and `run.cmd report` also refreshing this file's
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
   the PC by default): **20 pages before the references in the cloud export** (the merged source needs a fresh export and page-count check), so any addition needs a cut (figure 3 of the main run's
   training curves was already cut for space). Rules for it: every number
   comes from `results/` or the run files; Arial 12, 1.5 spacing,
   one-inch margins, 8 to 20 pages without the references. Still to do: a
   read-through by the team.
4. **Completed on PC:** `results/neo_policy.pt` / `.json` and provenance are shipped; final policy files retained. Original request: **ship the neo policy in Git** (it is nearly a one-policy candidate,
   section 4, item 2): it exists
   only as `runs/neo-dagger3/checkpoints/best.pt` on the PC. Export it next
   to the final policy (for example `results/neo_policy.pt` / `.json`)
   without replacing `results/final_policy.*`.
5. **Restart the GPU watchdog on the PC** after merging this branch: it
   now also pauses `parkour.neo_dagger`, but the copy already running uses
   the old job list.
6. Keep `TEAM_STATUS.md` and this file in step with each new result.
7. **More parkour abilities** (cloud session, in progress since 28 Oct
   evening): neo assist, head hitters, ice jumps, slime, block shapes.
   `docs/abilities_plan.md` lists the stages, their status and the design
   constraints found in the code; whatever needs training is queued for
   the PC with the gaming-safe settings of section 6. For the PC:
   - the neo assist in real Minecraft: `run.cmd live --checkpoint
     results\final_policy.pt --courses-file configs\neo_live_courses.json
     --assist "{}" --out runs\neo-assist-live` (and `--offline` first; the
     planner service runs on the PC's Python);
   - train the new kinds and the neo levels on the discount-0.995 policy,
     one run at a time with the gaming flags: `python scripts/sweep.py
     configs\sweep_new_skills.json --slots 1` (`extra-ft`: the extra levels
     after the basic ones, uniform curriculum; `advanced-g995`: levels 6-8).
     Pick on validation, then `parkour.jumpeval --extra`,
     `scripts/extra_preview.py` and the standard and obstacle test sets.
   - a learned neo on top of `advanced-g995` in one command (rounds of the
     recipe below, each kept only if it beats the last on the validation neo
     courses): `python scripts/neo_pipeline.py --start
     runs\advanced-g995\checkpoints\best.pt --name neo995 --rounds 3
     --gaming` (`--dry-run` prints the commands). By hand, the same recipe
     (the neo-dagger recipe, now with
     the assist's faster search): `python -m parkour.neo_dagger --checkpoint
     runs\advanced-g995\checkpoints\best.pt --out runs\neo995_r1.pt --prior
     0.5 --expert assist`, then `run.cmd train --run neo995-r1 --init-from
     runs\advanced-g995\checkpoints\best.pt --advanced --obstacle-prob 0.3
     --gamma 0.995 --reward-stall 5 --bc-coef 0.3 --bc-pairs runs\neo995_r1.pt
     --wide-neo --lr 3e-4 --lr-max 3e-4 --iterations 500` (neo-dagger3's
     settings), more rounds from its `best.pt`, each chosen on the
     validation neo courses (`parkour.neo_assist --set official --family 2
     --off`).

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
| `parkour/physics.py` | exact float64 port of prismarine-physics 1.11.1 (operand order kept) plus vanilla's 1e-7 `COLLISION_EPSILON`; the optional ice grid (`World(slip=...)`) and shape grid (`World(shape=...)`) are off unless passed | `tests/test_physics_parity.py` (also with ice and shapes), `test_shapes.py`, `test_corner_slip.py`, `test_inside_block.py` |
| `configs/block_shapes.json` | the collision boxes of the shaped blocks, written by `bridge/export_shapes.js` from prismarine-block | `node bridge/export_shapes.js --check` (in `test_physics_parity.py`) |
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
started before that does not pause it until restarted. Live Minecraft runs
are left alone on purpose (the bot plays in real time, and `parkour.live`
pumps the server's output): neither their process nor the live bot's planner
service (`neo_assist --serve --realtime`) is paused, while offline runs
(`parkour.live --offline`) and their planner service are (since 28 Oct late
evening; an older watchdog copy skipped every planner service). Log:
`runs/gpu_watchdog.log`. If the watchdog is ever killed while jobs are
paused: `python scripts/gpu_watchdog.py --resume-all`.

**Current PC instruction (28 Oct evening):** queue `g995-adv`, `g995-ice` concurrently, then `g995-extra`, with at most two trainers. The later user request caps both compute and aggregate memory at 70%, superseding the earlier unrestricted-GPU request. Overwatch was observed running at 22:04; gaming limits are active. Resume each same run from its own `latest.pt`; retain the newer 70% aggregate target. Other applications can independently exceed it, so monitor actual usage and do not stop the user's game.

**Earlier gaming guidance.** The user plays on the same GPU. On
28 Oct they first asked for "a safe buffer", then (same evening) "you can
use more GPU, don't leave like a 40% margin". So: one training process at
the normal size, using most of the GPU but not all of it (flags added by the
cloud session, not yet run on the PC):

```text
python scripts/gpu_watchdog.py                      (the user's rule: pause above 65 C, resume below 58 C)
run.cmd train --run NAME ... --gpu-mem-fraction 0.75 --gpu-duty 0.9 --threads 8
```

`--gpu-mem-fraction 0.75` caps PyTorch at 75% of the 16 GB (12 GB; one
16,384-environment run uses less than half the card, since two fit). An
allocation above the cap stops the run with "CUDA out of memory": then
resume from `runs/NAME/checkpoints/latest.pt` with a lower `--num-envs`.
`--gpu-duty 0.9` sleeps after every iteration so the GPU computes at most
90% of the time, a 10% margin for the game; `--threads 8` leaves CPU cores
to it. All three are off by default, and with the default 16,384
environments a run compares directly with every earlier run. If the game
stutters anyway, lower the duty (0.8) before anything else. To stop GPU
work completely while a game runs, start the watchdog with `--pause-while
VALORANT-Win64-Shipping.exe` (program names as in Task Manager's details
tab): it suspends the jobs while the game runs and resumes them after it
closes and the GPU is below `--cool`. This watchdog version also pauses
`parkour.neo_assist`; restart a copy started before it.

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
  real neos), and not by blocks under the target (a ladder climb's wall
  counted as a neo until 28 Oct evening; no held-out neo course changed).
- Planning with the exact physics (`parkour/neo_assist.py`): a plan
  searched from the bot's exact state lands exactly as searched, in the
  simulator and in the offline bot (prismarine-physics), so what limits the
  assist is only whether the search finds a landing from that state. A
  policy prior in the search made it worse on validation (111 vs 119 of 218
  neo courses). After a failed search, stopping on the takeoff and searching
  again from the standstill finds a landing for about a third of the
  failures (scratch experiment: 4 of 11; `--brake`).
- prismarine-physics takes version rules from its own feature table, not
  minecraft-data: in 1.21 soul sand and honey do not slow the player,
  pressing jump on a ladder climbs, trapdoors over ladders are not
  climbable. A slime bounce keeps the landing speed (upward), so how high it
  goes depends on how the pad is hit: the assist plans a slime jump and the
  exit after it as one plan.
- `train.py`: `bc_decay` defaults to False, as in every neo run (the
  imitation weight was constant).

## Training queue operation (PC, 28 Oct evening)

**Latest user decision:** bound both compute and memory to 70%. The user rejected the initial 10-second duty window because it produced long 98% bursts. The replacement watchdog uses shared 50-ms windows with measured-utilization feedback and headroom below 70%; telemetry and process discovery run separately from the pacing loop. This is a measured software target, not a hardware guarantee against every transient peak or game usage. Overwatch increased the non-training baseline to about 40% memory and 37% compute at 22:04. Each trainer now has a 13% allocator cap, with smaller working tensors, leaving room for the game and runtime overhead. Verify actual card memory with nvidia-smi. This supersedes the earlier unrestricted-GPU request and the 75% per-job gaming recipe.

The first startup attempts hit Windows shared-memory spill (the dense ice friction grid alone was 8 GiB). Their logs are retained in `runs/startup-memory-spill-20261028/` (advanced: 8 updates, ice: 1; neither had a checkpoint). `--compact-ice` stores one byte per cell and reconstructs the exact float64 friction on lookup. Validation and test pools use `--eval-batch-size 256`, keeping the same course sets with bounded memory; unused cache is released between batches. The later 28% ice run exceeded its allocator limit during PPO, after saving iteration 49; its full original log and config are retained in `runs/g995-ice/recovery/20261028-220658/`, including 41 uncheckpointed updates. Both jobs resumed their own iteration-49 checkpoint at 22:07. With Overwatch open, all queued recipes use a 1,024-course training pool, 8,192 environments, rollout 64 and 32 minibatches; game ticks per PPO iteration remain 524,288. The changed minibatch count and rollout shape affect training dynamics and must be reported. Test sets and checkpoint selection remain unchanged. Test pools allocate only at final testing with `--test-once`.

- `scripts/training_queue.py` reads `configs/training_queue.json`, keeps at most two logical training process trees, and writes `runs/training_queue_state.json`. Initial order: advanced and ice together, then extra. Later seeds use the best completed initial recipe on the common validation levels 0-5 (standard plus obstacles), never test scores.
- Windows/OneDrive can temporarily lock the queue state file. The queue retries atomic replacement, retains the previous valid file and its in-memory job inventory if the lock persists, and retries next loop. Configuration read errors prevent new launches until the next successful read. A restarted queue attaches to existing trainer process trees; do not reset its state or restart trainers merely to restore scheduling.
- Keep this queue as the sole training launcher. Before using `scripts/neo_pipeline.py`, integrate its stages with the two-slot/memory controls and `--test-once`; its direct training commands do not yet inherit these PC restrictions.
- `runs/training_control.json`: `gaming=true` requests each job to stop cleanly at its next 50-update checkpoint and resume under the same run name with the three gaming flags; `gaming=false` does the reverse. `hold=true` prevents new starts. Do not launch a second queue.
- `--test-once` validates every 50 iterations and tests only the selected best at completion. Resume restores iteration, optimizer, learning rate and validation best; simulator episodes/course pool restart, so it is not bit-identical continuation. Completed tested runs get a new name for further training.
- Current watchdog command: `python scripts/gpu_watchdog.py --hot 65 --cool 58 --duty 0.55 --period 0.05 --utilization-target 0.7 --every 1`. The host duty ceiling is deliberately below the GPU busy-time limit because queued kernels continue briefly after host suspension. Keep `scripts/keep_awake.py` running. A five-minute native heartbeat in this chat monitors progress and publishes completed results; the local queue fills training slots immediately.
- Neo policy now shipped as `results/neo_policy.pt` and `.json`; hashes in `results/neo_policy_provenance.json`. Final policy files retained.
- Normal Windows test run after the second merge: 85 passed, 8 subtests (`runs/pc-final-merge-tests.txt`). CPU assist parity now pins the subprocess service to CPU (device-specific search RNG differs from CUDA).

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

Generated on top of commit `888fe9f` (2026-10-29 00:29, branch `main`): Merge mixed and extra-course fingerprint guards

| Document | Last change | Commits since |
| --- | --- | --- |
| `AGENTS.md` | `85b8f50` 2026-10-28 14:27 | 82 |
| `HANDOFF.md` | updated together with this snapshot | 0 |
| `TEAM_STATUS.md` | updated together with this snapshot | 0 |
| `README.md` | `408de3d` 2026-10-28 22:10 | 40 |
| `docs/design.md` | `8adb005` 2026-10-28 18:22 | 57 |
| `docs/report_draft.md` | `408de3d` 2026-10-28 22:10 | 40 |
| `results/README.md` | updated together with this snapshot | 0 |
| `results/DATA.md` | updated together with this snapshot | 0 |

<!-- snapshot data: compared by `python -m parkour.handoff --check` -->

### Shipped policies

Simulator, held-out test sets (1,200 courses each), checkpoint picked on the validation family.

| File | Checkpoint | Iteration | Validation | Test, standard | Test, obstacles |
| --- | --- | --- | --- | --- | --- |
| `results/final_policy.pt` / `.json` | `runs/obstacles-g995/checkpoints/best.pt` | 699 | 97.6 | 98.4 | 95.2 |
| `results/final_policy_v1.pt` / `.json` | `runs/obstacles-ft/checkpoints/best.pt` | 999 | 96.6 | 97.1 | 93.6 |

### Training runs (`results/runs/*/summary.json`)

Simulator, held-out test set. "Picked" is the validation-selected checkpoint (`best.pt`); runs without one predate validation selection. Standard runs cover levels 0-5; advanced/neo runs add levels 6-8, while extra runs add their configured extra levels. Check each run's config: overall scores from different course sets are not comparable. Common levels 0-5 use the same courses.

| Run | Last iteration | Game ticks (B) | Picked: iteration | validation | test | Last: test | Last: obstacles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `advanced-ft` | 799 | 0.42 | 699 | 69.2 | 71.2 | 70.9 | 67.6 |
| `advanced-ft2` | 299 | 0.16 | 249 | 81.6 | 83.0 | 82.8 | 78.6 |
| `advanced-ft3` | 599 | 0.31 | 499 | 82.7 | 83.6 | 83.5 | 80.6 |
| `final-v2` | 499 | 0.26 | 499 | 89.6 | 88.5 | 88.5 | 88.1 |
| `full-s1` | 999 | 0.52 | - | - | - | 94.2 | - |
| `full-s2` | 999 | 0.52 | 949 | 93.0 | 93.1 | 91.9 | - |
| `g995-adv` | 599 | 0.31 | 99 | 81.8 | - | - | - |
| `g995-ice` | 699 | 0.37 | 449 | 97.3 | - | - | - |
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
| `neo-assist-offline-all-brake` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 25/31 | 80.6 | 100.0 |
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
| `v2-tuned-planner-live` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 51/60 | 85.0 | 100.0 |
| `v2-tuned-planner-live-obstacles` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 56/60 | 93.3 | 100.0 |
| `v2-tuned-planner-offline` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 51/60 | 85.0 | 100.0 |
| `v2-tuned-planner-offline-obstacles` | offline (prismarine-physics, no server) | `results/final_policy.pt` | 56/60 | 93.3 | 100.0 |
| `viewer-test` | live (Minecraft 1.21.4 server) | `results/final_policy.pt` | 1/1 | 100.0 | 100.0 |

### In `results/` but not yet in `results/DATA.md`

`operations/` (3 files), `report_draft.pdf`, `videos/ghost_race_02_level_1_course_0.mp4`, `videos/ghost_race_03_level_2_course_0.mp4`, `videos/ghost_race_04_level_3_course_0.mp4`, `videos/ghost_race_05_level_4_course_0.mp4`, `videos/replay_137_neo_policy_level_7_course_5.mp4`

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

- **28 Oct 2026, 21:38, Codex on training PC:** confirmed retuned planner in real Minecraft: 51/60 standard, 56/60 obstacles, all simulator outcomes matched, zero server corrections; defaults adopted in Python and JS. Evidence archived by report. Shared 70% compute plus conservative per-job allocator caps and compact/batched memory allocation are being checked before queue restart.
- **28 Oct 2026, 21:33, Codex on the training PC:** second merge brings cloud work through `21188a6`, including the CUDA memory-cap device-index fix and shapes/slime/ladders. Preserve the requested 70% shared compute and aggregate memory budget. Compact friction uses one-byte material ids with exact float64 lookup (stone, ice and slime). Initial cap startup failures occurred before any training and are logged.
- **28 Oct 2026, evening, Codex (main session on the training PC):** preserved the PC documentation updates, merged cloud work through `d9237e7`, retained both session histories and the full earlier PC handoff below. Preparing two unrestricted GPU runs (`g995-adv`, `g995-ice`), then `g995-extra`; validation-only checkpoint selection and one final test evaluation per run. Game is closed. When gaming resumes, wait for the next 50-iteration checkpoint and resume each same run from its own latest checkpoint with `--gpu-mem-fraction 0.75 --gpu-duty 0.9 --threads 8`.

Newest first. One entry per session; earlier history is in the
[TEAM_STATUS update log](TEAM_STATUS.md#update-log) and `git log`.

- **28 Oct 2026, late evening, Claude Code (the same cloud session after a
  context reset, branch `claude/pensive-knuth-biwj11`, CPU only):** the
  assist plans the new kinds (`--jumps all`; slime jumps planned to the
  exit, longer ladder plans, shape-aware plan checks in the bots), brakes
  and searches again after a failed search (`--brake`), reads course files
  with their jump specs (a review found the planner service had none), and
  has a `mixed` course set with every special kind (families 14/15);
  `jumpeval` keeps the blocks of composite kinds; ladder walls no longer
  count as neos; `scripts/neo_pipeline.py` runs learned-neo DAgger rounds
  on the PC in one command. Validation so far (simulator, 218 neo courses):
  `advanced` with the bots' search 124/218; the policy prior 111/218
  (rejected). Running when this was written: `all` on random 13, `all` +
  brake on official 2 and random 13, the extra feasibility study (writes
  the shaped, slime and ladder levels). Next: choose on validation, run the
  test sets once (official 1, random 12, mixed 14 after the level file),
  the offline bot on `configs/neo_live_courses.json`, the per-kind preview,
  `jumpeval --extra`, then this file, TEAM_STATUS and DATA.md.
- **28 Oct 2026, evening, Claude Code (cloud session, continued again,
  branch `claude/pensive-knuth-biwj11`, CPU only):** the user asked for
  more parkour abilities (other blocks, head hitters) and consistent neos
  while they game on the PC's GPU (first "a safe buffer", then "don't leave
  like a 40% margin"). This session cannot reach the PC, so it builds and
  measures on the CPU and prepares training for the PC. Plan and progress:
  `docs/abilities_plan.md`. Done: held-out fingerprint guard
  (`tests/test_heldout_fingerprints.py`, the 8 test/validation sets are
  byte-identical to commit 6170803); gaming flags `--gpu-mem-fraction`,
  `--gpu-duty`, `--threads` and `gpu_watchdog.py --pause-while` (section 6:
  75% memory, 90% duty); the neo assist in the simulator and both bots
  (section 2); head-hitter and ice run-up jump kinds and levels; block
  shapes in the physics with exact parity. Next: the assist on the test
  sets with the setting chosen on validation, shaped platforms in courses
  (stage D), slime, the PC training queue.
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

## Preserved PC handoff before the cloud merge (28 Oct 2026)

Historical text retained in full to preserve the PC audit and detailed evidence. The current state and operating instructions above supersede historical status statements below.

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
