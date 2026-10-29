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

## ON HOLD: the new-abilities test runs (paused 29 Oct, about 00:50 UTC)

The user paused this task (cloud session, branch `claude/pensive-knuth-biwj11`,
last commit before this note `d2b81dd`) to build a replay artifact. Nothing
here touches the training PC; its jobs keep running. The replay page is done
(29 Oct, session log); this task is next when the user resumes it.

**Done** (all pushed; simulator unless said; plan and status in
`docs/abilities_plan.md`):
- New blocks and jump kinds, final levels (`configs/jump_levels_extra.json`,
  fingerprinted): head hitters, ice run-ups, shaped blocks, slime bounces,
  ladder climbs; alone, the final policy makes 99% of the shaped-block jumps,
  68% head hitters, 49% ice, 40% slime, 38% ladders (`results/extra/`).
- Planning assist options `--jumps all`, `--brake`, `--approach`, chosen on
  validation by a rule fixed before the runs: `{"jumps": "all", "retries": 1,
  "brake": 14, "retry_horizon": 48, "approach": 48}`. Validation: official
  neo courses (family 2) 157/218 (brake only 147, advanced jumps 124, policy
  alone 0); random neo courses (13) 135/200 (brake only 119, policy alone 27);
  mixed courses (15) policy alone 37/200. Rows: `results/neo/assist_validation.jsonl`,
  details `results/neo/assist_details/`.
- Offline bot with `all` + brake on `configs/neo_live_courses.json`: 25/31,
  the same as the simulator (`results/games/neo-assist-offline-all-brake`).
- Test rows so far, policy alone: official 1 0/210, random 12 33/200, mixed
  14 35/200 (`results/neo/assist_test.jsonl`).

**Paused in the cloud container** (frozen with SIGSTOP at 00:55 UTC, so no
progress is lost while the container lives: `kill -CONT $(pgrep -f
"[p]arkour.neo_assist|[e]xtra_preview.py|[l]aunch_test.sh|[l]aunch_preview.sh")`
resumes them; after a container restart they are gone, but they are
restart-safe: finished parts are kept, so rerun the same commands):

```text
python -m parkour.neo_assist --set official --family 1 --threads 1 --jumps all --retries 1 --brake 14 --retry-horizon 48 --approach 48 --parts 4 --jobs 2 --tag "all_brake_approach; code cd81960" --name official1_all_brake_approach
python -m parkour.neo_assist --set random --family 12 (same flags) --name random12_all_brake_approach
then: --set mixed --family 15 (same flags) --name mixed15_all_brake_approach, and --set mixed --family 14 ... --name mixed14_all_brake_approach
python scripts/extra_preview.py --family 2 --assist-per-level 20 --threads 1 --assist '{"jumps": "advanced"}' --assist '{"jumps": "all", "retries": 1, "brake": 14, "retry_horizon": 48}'
```

The tag must stay exactly `all_brake_approach; code cd81960` (each part
checks it). Parts are in `runs/assist_parts/NAME/`; a finished run appends
its row to `results/neo/assist_test.jsonl` (families 1, 12, 14) or
`assist_validation.jsonl` (15) and writes `results/neo/assist_details/NAME.json`;
the preview keeps finished kinds in `runs/extra_preview/family2/` and writes
`results/extra/course_preview_family2.json` when all six are done.

**To resume:** commit any new rows and details files; run the commands above
for whatever has no `merged.json` yet; then `scripts/extra_preview.py
--family 1` with the same arguments; the offline bot with the chosen setting
(`python -m parkour.live --offline --checkpoint results/final_policy.pt
--courses-file configs/neo_live_courses.json --assist '{"jumps": "all",
"retries": 1, "brake": 14, "retry_horizon": 48, "approach": 48}' --out
runs/neo-assist-offline-chosen`); `python -m parkour.report`; update section
2 and 3 below, TEAM_STATUS, `results/DATA.md` (add `assist_parts`, the new
details and preview files), `docs/abilities_plan.md`; a final review of `git
diff e2381cc HEAD`; `pytest tests -q`; commit and push.

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
    landing). Adding approach planning completes 157/218 official validation
    courses (72.02%; `official2_all_brake_approach.json`), versus 147/218
    (67.43%) without approach. Random neo courses (family 13): the policy alone 27 of 200,
    planning every jump 106 (`random13_all.json`), with brake and retry 119,
    320 of 396 neos (`random13_all_brake.json`), and with an approach plan
    over the preceding jump and the neo (`--approach 48`) 135, 332 of 392
    neos (`random13_all_brake_approach.json`, run in 4 parts). Both bots
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

**PC completion update, 28 Oct 21:38:** cloud merged through `21188a6`; neo files shipped; retuned planner verified on both real-Minecraft sets (51/60 and 56/60, zero corrections) and adopted in Python/JS. Items 1, 2, 4 and 5 below describe the completed integration steps and ongoing merge/watchdog procedure. The latest user request (29 Oct) permits 100% GPU use, with the temperature rule retained.

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
   constraints found in the code. The current two-slot queue and full-GPU
   instruction in section 6 govern PC training. Commands below are recipe
   references; do not start a competing sweep or pipeline. For the PC:
   - the neo assist in real Minecraft: `run.cmd live --checkpoint
     results\final_policy.pt --courses-file configs\neo_live_courses.json
     --assist "{}" --out runs\neo-assist-live` (and `--offline` first; the
     planner service runs on the PC's Python);
   - train the new kinds and the neo levels on the discount-0.995 policy,
     original one-slot gaming sweep reference: `python scripts/sweep.py
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
1 s it reads nvidia-smi; above 65 C it suspends the project's GPU jobs by
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

**Current PC instruction (29 Oct, 00:51):** the user now permits 100% GPU use, superseding the 70% compute/aggregate-memory target. Initial order was `g995-adv` and `g995-ice` together, then `g995-extra`, with at most two trainers. Both initial runs, extra and ice seeds 1-3 have completed; the queue continues additional validation-selected ice seeds. Read `runs/training_queue_state.json` and the process inventory for current jobs. The shared throttle and all trainer memory, duty and thread flags were removed; the initial runs resumed from their own checkpoints 699/799. Keep thermal pause above 65 C/resume below 58 C and keep-awake. Preserve the existing rollout shape and recipe. If the user says gaming has resumed, use another checkpoint-safe transition with their requested gaming flags; do not stop their game.

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

## Training queue operation (PC, updated 29 Oct)

**Latest user decision (29 Oct):** 100% GPU use is permitted. The earlier 70% compute/aggregate-memory target and 13% per-trainer allocator caps are superseded. Use thermal protection only, without a duty or utilization target, and omit `--gpu-mem-fraction`, `--gpu-duty` and `--threads` from training commands. Current queue control: `gaming=false`, `memory_fraction_per_job=0`, `shared_compute_duty=1`, `small_memory=true`, `hold=false`. The shared duty field records intent; the watchdog's command actually controls shared throttling. Keep the existing compact tensors and rollout dimensions for continuity. Full GPU availability does not require filling unused memory or guarantee a constant 100% utilization reading.

The first startup attempts hit Windows shared-memory spill (the dense ice friction grid alone was 8 GiB). Their logs are retained in `runs/startup-memory-spill-20261028/` (advanced: 8 updates, ice: 1; neither had a checkpoint). `--compact-ice` stores one byte per cell and reconstructs the exact float64 friction on lookup. Validation and test pools use `--eval-batch-size 256`, keeping the same course sets with bounded memory; unused cache is released between batches. The later 28% ice run exceeded its allocator limit during PPO, after saving iteration 49; its full original log and config are retained in `runs/g995-ice/recovery/20261028-220658/`, including 41 uncheckpointed updates. Both jobs resumed their own iteration-49 checkpoint at 22:07. With Overwatch open, all queued recipes use a 1,024-course training pool, 8,192 environments, rollout 64 and 32 minibatches; game ticks per PPO iteration remain 524,288. The changed minibatch count and rollout shape affect training dynamics and must be reported. Test sets and checkpoint selection remain unchanged. Test pools allocate only at final testing with `--test-once`.

- `scripts/training_queue.py` reads `configs/training_queue.json`, keeps at most two logical training process trees, and writes `runs/training_queue_state.json`. Initial order: advanced and ice together, then extra. Later seeds use the best completed initial recipe on the common validation levels 0-5 (standard plus obstacles), never test scores.
- Windows/OneDrive can temporarily lock the queue state file. The queue retries atomic replacement, retains the previous valid file and its in-memory job inventory if the lock persists, and retries next loop. Configuration read errors prevent new launches until the next successful read. A restarted queue attaches to existing trainer process trees; do not reset its state or restart trainers merely to restore scheduling.
- Keep this queue as the sole training launcher. Before using `scripts/neo_pipeline.py`, integrate its stages with the two-slot/memory controls and `--test-once`; its direct training commands do not yet inherit these PC restrictions.
- `runs/training_control.json`: a change in `gaming` requests each job to stop cleanly at its next 50-update checkpoint and resume under the same run name. `gaming=true` adds duty 0.9 and threads 8; memory is controlled separately by `memory_fraction_per_job` (set it to 0.75 for the user's original gaming recipe, or their newer requested cap). `gaming=false` plus memory fraction 0 removes all three limits. `hold=true` prevents new starts. Do not launch a second queue.
- `--test-once` validates every 50 iterations and tests only the selected best at completion. Resume restores iteration, optimizer, learning rate and validation best; simulator episodes/course pool restart, so it is not bit-identical continuation. Completed tested runs get a new name for further training.
- Current watchdog command: `python scripts/gpu_watchdog.py --hot 65 --cool 58 --every 1` (default duty 1, utilization target 0). After forcibly stopping an older watchdog, resume its suspended project jobs before starting the replacement. Keep `scripts/keep_awake.py` running. The five-minute native heartbeat has the same unrestricted-GPU instruction and monitors progress/publication; the local queue fills training slots immediately.
- Combined advanced/ice completion notification delivered on 29 Oct at about 01:24 Israel time. Both initial runs are complete; full comparison and delivery record are in `results/operations/g995-initial-comparison.json`. Do not repeat this notification. Continue extra and validation-selected seeds.
- Neo policy now shipped as `results/neo_policy.pt` and `.json`; hashes in `results/neo_policy_provenance.json`. Final policy files retained.
- Normal Windows test run after the second merge: 85 passed, 8 subtests (`runs/pc-final-merge-tests.txt`). CPU assist parity now pins the subprocess service to CPU (device-specific search RNG differs from CUDA).

**Resumable assist evaluations (cloud `c38de17`, hardened on PC):**
`parkour.neo_assist --name NAME --parts N --jobs J --device cpu` saves each
part and can resume or republish completed parts without re-evaluation.
Named manager/part locks prevent duplicate workers; completed parts must
match checkpoint content, ordered course content, settings and partition
bounds. Publication uses a stable run ID to avoid a second result row after
an interrupted append. `scripts/extra_preview.py --level NAME` similarly
caches each kind, with checkpoint/device/settings/level identity and atomic
publication. Older caches without these identities fail closed: preserve
them separately rather than silently treating them as current evidence.
These opt-in helpers do not inherit the training queue's two-slot controls;
do not launch additional GPU evaluations alongside the active trainers.
After a forcibly killed coordinator, check its remaining worker processes
before restarting; held part locks prevent duplicate evaluation.

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
  `results/showcase.html`, `results/overnight_plan.html`,
  `results/showcase_replays.html` (`scripts/make_showcase_replays.py`). Videos:
  `results/videos/` (`scripts/render_videos.py`, `scripts/film_pov.py`).

## 9. Snapshot (generated)

Do not edit between the markers: `python -m parkour.handoff` rewrites it from
Git and `results/` (the report step runs it too).

<!-- BEGIN AUTO-SNAPSHOT: written by `python -m parkour.handoff`; edits inside are overwritten -->

Generated on top of commit `466519b` (2026-10-29 08:25, branch `main`): Archive validation progress for ice seeds 10 and 11

| Document | Last change | Commits since |
| --- | --- | --- |
| `AGENTS.md` | `85b8f50` 2026-10-28 14:27 | 184 |
| `HANDOFF.md` | updated together with this snapshot | 0 |
| `TEAM_STATUS.md` | updated together with this snapshot | 0 |
| `README.md` | `408de3d` 2026-10-28 22:10 | 142 |
| `docs/design.md` | `8adb005` 2026-10-28 18:22 | 159 |
| `docs/report_draft.md` | `408de3d` 2026-10-28 22:10 | 142 |
| `results/README.md` | `99f6ff2` 2026-10-29 02:38 | 69 |
| `results/DATA.md` | `a4d31ba` 2026-10-29 03:17 | 61 |

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
| `g995-adv` | 999 | 0.52 | 99 | 81.8 | 82.3 | - | - |
| `g995-extra` | 999 | 0.52 | 849 | 90.7 | 91.2 | - | - |
| `g995-ice` | 999 | 0.52 | 449 | 97.3 | 97.5 | - | - |
| `g995-ice-s1` | 999 | 0.52 | 949 | 97.7 | 97.2 | - | - |
| `g995-ice-s10` | 749 | 0.39 | 149 | 96.4 | - | - | - |
| `g995-ice-s11` | 699 | 0.37 | 649 | 97.8 | - | - | - |
| `g995-ice-s2` | 999 | 0.52 | 749 | 96.9 | 97.4 | - | - |
| `g995-ice-s3` | 999 | 0.52 | 249 | 96.8 | 97.4 | - | - |
| `g995-ice-s4` | 999 | 0.52 | 449 | 97.0 | 97.2 | - | - |
| `g995-ice-s5` | 999 | 0.52 | 99 | 97.0 | 98.5 | - | - |
| `g995-ice-s6` | 999 | 0.52 | 499 | 97.2 | 97.7 | - | - |
| `g995-ice-s7` | 999 | 0.52 | 749 | 97.0 | 96.7 | - | - |
| `g995-ice-s8` | 999 | 0.52 | 199 | 96.8 | 97.3 | - | - |
| `g995-ice-s9` | 999 | 0.52 | 149 | 96.8 | 97.3 | - | - |
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

`operations/` (15 files), `report_draft.pdf`, `videos/ghost_race_02_level_1_course_0.mp4`, `videos/ghost_race_03_level_2_course_0.mp4`, `videos/ghost_race_04_level_3_course_0.mp4`, `videos/ghost_race_05_level_4_course_0.mp4`, `videos/replay_137_neo_policy_level_7_course_5.mp4`

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

- **29 Oct 2026, 08:30, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s10: iteration 749, validation 95.58%, selected best 149 (96.42%), 6 levels; g995-ice-s11: iteration 699, validation 96.17%, selected best 649 (97.83%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 08:25, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s10: iteration 699, validation 95.83%, selected best 149 (96.42%), 6 levels; g995-ice-s11: iteration 599, validation 96.08%, selected best 99 (96.50%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 08:20, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s10: iteration 599, validation 96.08%, selected best 149 (96.42%), 6 levels; g995-ice-s11: iteration 549, validation 95.75%, selected best 99 (96.50%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 08:15, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s10: iteration 549, validation 93.33%, selected best 149 (96.42%), 6 levels; g995-ice-s11: iteration 499, validation 95.92%, selected best 99 (96.50%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 08:10, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s10: iteration 499, validation 94.33%, selected best 149 (96.42%), 6 levels; g995-ice-s11: iteration 399, validation 92.17%, selected best 99 (96.50%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 08:05, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s10: iteration 399, validation 94.33%, selected best 149 (96.42%), 6 levels; g995-ice-s11: iteration 349, validation 95.92%, selected best 99 (96.50%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 08:00, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s10: iteration 349, validation 95.33%, selected best 149 (96.42%), 6 levels; g995-ice-s11: iteration 249, validation 95.50%, selected best 99 (96.50%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 07:55, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s10: iteration 249, validation 96.08%, selected best 149 (96.42%), 6 levels; g995-ice-s11: iteration 199, validation 96.25%, selected best 99 (96.50%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 07:50, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s10: iteration 199, validation 94.00%, selected best 149 (96.42%), 6 levels; g995-ice-s11: iteration 149, validation 95.67%, selected best 99 (96.50%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 07:45, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s10: iteration 149, validation 96.42%, selected best 149 (96.42%), 6 levels; g995-ice-s11: iteration 49, validation 96.17%, selected best 49 (96.17%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 07:41, Codex heartbeat on training PC:** g995-ice-s9 completed all 1,000 iterations. Its checkpoint 149 was selected only by simulator validation families 2/5 (96.7500%, first maximum); final standard test 97.3333% and obstacle test 93.9167%, versus the existing final policy 97.5833% validation, 98.4167% standard and 95.2500% obstacles (differences -0.8333, -1.0833 and -1.3333 percentage points). Comparisons use common levels 0-5 and identical held-out course sets; validation has 100 courses per level from each family 2/5, and each test has 200 per level. Exactly one standard and one obstacle test row at the selected checkpoint; no evaluation rerun. Archived `results/operations/g995-ice-s9-comparison.json`; completion comparison delivered with this heartbeat, do not repeat. The queue automatically started seed 11 after seed 9 exited; seeds 10/11 occupy the two training slots. Seed 10 archived iteration 49 validation 96.0833%, also its selected best; seed 11 has no archived validation checkpoint yet. Both remain untested. Full GPU settings, unchanged rollout shape, thermal 65/58 C safeguards and keep-awake remain active; sampled GPU use 77-81%, temperature 40.9-43.8 C, memory 5,703/16,303 MiB. Final policy files retained. Cloud evaluation remains on hold; no new cloud commits. Report/handoff refreshed; CSV/summary checks passed, no executable changes and prior test suite remains applicable.

- **29 Oct 2026, 07:35, Codex heartbeat on training PC:** g995-ice-s8 completed all 1,000 iterations. Its checkpoint 199 was selected only by simulator validation families 2/5 (96.8333%, first maximum); final standard test 97.3333% and obstacle test 95.25%, versus the existing final policy 97.5833% validation, 98.4167% standard and 95.25% obstacles (differences -0.75, -1.0833 and 0.00 percentage points). Comparisons use common levels 0-5 and identical held-out course sets; validation has 100 courses per level from each family 2/5, and each test has 200 per level. Exactly one standard and one obstacle test row at the selected checkpoint; no evaluation rerun. Archived `results/operations/g995-ice-s8-comparison.json`; completion comparison delivered with this heartbeat, do not repeat. The queue automatically started seed 10 after seed 8 exited; seeds 9/10 occupy the two training slots. Seed 9 archived iteration 949 validation 96.0833%, selected best 149 (96.7500%), still untested. Full GPU settings, unchanged rollout shape, thermal 65/58 C safeguards and keep-awake remain active; final policy files retained. Cloud evaluation remains on hold; no new cloud commits. Report/handoff refreshed; CSV/summary checks passed, no executable changes and prior test suite remains applicable.

- **29 Oct 2026, 07:30, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 949, validation 95.58%, selected best 199 (96.83%), 6 levels; g995-ice-s9: iteration 849, validation 95.25%, selected best 149 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 07:25, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 849, validation 95.25%, selected best 199 (96.83%), 6 levels; g995-ice-s9: iteration 799, validation 94.58%, selected best 149 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 07:20, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 799, validation 94.33%, selected best 199 (96.83%), 6 levels; g995-ice-s9: iteration 699, validation 94.67%, selected best 149 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 07:15, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 749, validation 93.00%, selected best 199 (96.83%), 6 levels; g995-ice-s9: iteration 649, validation 95.83%, selected best 149 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 07:10, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 649, validation 95.50%, selected best 199 (96.83%), 6 levels; g995-ice-s9: iteration 599, validation 95.17%, selected best 149 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 07:05, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 599, validation 96.08%, selected best 199 (96.83%), 6 levels; g995-ice-s9: iteration 499, validation 95.42%, selected best 149 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 07:00, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 499, validation 96.08%, selected best 199 (96.83%), 6 levels; g995-ice-s9: iteration 449, validation 96.33%, selected best 149 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 06:55, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 449, validation 94.83%, selected best 199 (96.83%), 6 levels; g995-ice-s9: iteration 349, validation 96.00%, selected best 149 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 06:50, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 399, validation 96.83%, selected best 199 (96.83%), 6 levels; g995-ice-s9: iteration 299, validation 96.00%, selected best 149 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 06:45, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 299, validation 95.50%, selected best 199 (96.83%), 6 levels; g995-ice-s9: iteration 249, validation 95.67%, selected best 149 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 06:40, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 249, validation 95.92%, selected best 199 (96.83%), 6 levels; g995-ice-s9: iteration 149, validation 96.75%, selected best 149 (96.75%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 06:35, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 149, validation 96.25%, selected best 149 (96.25%), 6 levels; g995-ice-s9: iteration 99, validation 96.58%, selected best 99 (96.58%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 06:30, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s8: iteration 99, validation 90.25%, selected best 49 (95.92%), 6 levels; g995-ice-s9: iteration 49, validation 96.25%, selected best 49 (96.25%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 06:27, Codex heartbeat on training PC:** g995-ice-s7 completed all 1,000 iterations. Its checkpoint 749 was selected only by simulator validation families 2/5 (97.00%); final standard test 96.6667% and obstacle test 94.1667%, versus the existing final policy 97.5833% validation, 98.4167% standard and 95.25% obstacles (differences -0.5833, -1.75 and -1.0833 percentage points). Comparisons use common levels 0-5 and identical held-out course sets; validation has 100 courses per level from each family 2/5, and each test has 200 per level. Exactly one standard and one obstacle test row at the selected checkpoint; no evaluation rerun. Archived `results/operations/g995-ice-s7-comparison.json`; completion comparison delivered with this heartbeat, do not repeat. The queue automatically started seed 9 after seed 7 exited; seeds 8/9 occupy the two training slots. Seed 8 archived iteration 49 validation 95.9167%, selected best 49 (95.9167%), still untested. Full GPU settings, unchanged rollout shape, thermal 65/58 C safeguards and keep-awake remain active; final policy files retained. Cloud evaluation remains on hold; no new cloud commits. Report/handoff refreshed; CSV/summary checks passed, no executable changes and prior test suite remains applicable.

- **29 Oct 2026, 06:22, Codex heartbeat on training PC:** g995-ice-s6 completed all 1,000 iterations. Its checkpoint 499 was selected only by simulator validation families 2/5 (97.25%); final standard test 97.6667% and obstacle test 94.75%, versus the existing final policy 97.5833% validation, 98.4167% standard and 95.25% obstacles (differences -0.3333, -0.75 and -0.50 percentage points). All comparisons use common levels 0-5 and identical held-out course sets; validation has 100 courses per level from each family 2/5, and each test has 200 per level. Exactly one standard and one obstacle test row at the selected checkpoint; no evaluation rerun. Archived `results/operations/g995-ice-s6-comparison.json`; completion comparison delivered with this heartbeat, do not repeat. Corrected the validation-family description in seed 1-5 comparison artifacts; all scores and prior notification records are unchanged. The queue automatically started seed 8 after seed 6 exited; seeds 7/8 occupy the two training slots. Seed 7 archived iteration 949 validation 95.4167%, selected best 749 (97.00%), still untested. Full GPU settings, unchanged rollout shape, thermal 65/58 C safeguards and keep-awake remain active; final policy files retained. Cloud evaluation remains on hold; no new cloud commits. Report/handoff refreshed; CSV/summary checks passed, no executable changes and prior test suite remains applicable.

- **29 Oct 2026, 06:15, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 899, validation 94.92%, selected best 499 (97.25%), 6 levels; g995-ice-s7: iteration 849, validation 95.58%, selected best 749 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 06:10, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 849, validation 94.33%, selected best 499 (97.25%), 6 levels; g995-ice-s7: iteration 749, validation 97.00%, selected best 749 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 06:08, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 799, validation 94.75%, selected best 499 (97.25%), 6 levels; g995-ice-s7: iteration 749, validation 97.00%, selected best 749 (97.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 06:00, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 699, validation 96.58%, selected best 499 (97.25%), 6 levels; g995-ice-s7: iteration 599, validation 95.33%, selected best 449 (96.58%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 05:55, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 599, validation 94.00%, selected best 499 (97.25%), 6 levels; g995-ice-s7: iteration 549, validation 96.33%, selected best 449 (96.58%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 05:50, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 549, validation 96.67%, selected best 499 (97.25%), 6 levels; g995-ice-s7: iteration 499, validation 95.75%, selected best 449 (96.58%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 05:45, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 499, validation 97.25%, selected best 499 (97.25%), 6 levels; g995-ice-s7: iteration 399, validation 95.42%, selected best 99 (96.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 05:40, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 399, validation 96.17%, selected best 99 (96.33%), 6 levels; g995-ice-s7: iteration 349, validation 95.42%, selected best 99 (96.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 05:35, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 349, validation 96.00%, selected best 99 (96.33%), 6 levels; g995-ice-s7: iteration 249, validation 94.92%, selected best 99 (96.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 05:30, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 249, validation 88.17%, selected best 99 (96.33%), 6 levels; g995-ice-s7: iteration 199, validation 95.33%, selected best 99 (96.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 05:25, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 199, validation 95.83%, selected best 99 (96.33%), 6 levels; g995-ice-s7: iteration 149, validation 92.42%, selected best 99 (96.00%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 05:20, Codex heartbeat on training PC:** archived simulator validation (families 2/5): g995-ice-s6: iteration 149, validation 94.75%, selected best 99 (96.33%), 6 levels; g995-ice-s7: iteration 49, validation 87.83%, selected best 49 (87.83%), 6 levels. Scores over different level counts are not directly comparable. Summaries and validation-selected maxima match the CSVs; active test logs remain empty. The existing two-slot queue, unrestricted trainers, thermal protection (65/58 C) and keep-awake are healthy. No executable changes; prior checks remain applicable. Report/handoff refreshed. The initial advanced/ice comparison was already delivered and is not repeated.

- **29 Oct 2026, 05:16, Codex heartbeat on training PC:** g995-ice-s5 completed all 1,000 iterations in the simulator. Validation-selected checkpoint 99: validation 97.0000%, final standard test 98.5000%, obstacle test 95.1667%; versus existing final policy checkpoint 699, differences are -0.5833/+0.0833/-0.0833 percentage points (validation/standard/obstacles), equivalent to one additional standard course and one fewer obstacle course completed. Comparison uses identical common levels 0-5, 200 courses per level and final test families 1/4. Exactly one final standard and obstacle test row at checkpoint 99; no evaluation rerun. Verified comparison: `results/operations/g995-ice-s5-comparison.json`. This completion notification is delivered in this chat now; do not repeat. Existing final policy files retained. Archived seed 6 validation through iteration 49 (95.92%), retained best 49 (95.92%). The queue automatically started g995-ice-s7 alongside g995-ice-s6 using the recipe selected from common validation only; exactly two unrestricted trainers, thermal watchdog (65/58 C) and keep-awake remain healthy. Cloud evaluation hold remains in effect. Report/handoff refreshed; independent completion audit and CSV checks pass. No executable changes; the 109-test CPU suite from 03:16 remains applicable.

- **29 Oct 2026, 05:11, Codex heartbeat on training PC:** g995-ice-s4 completed all 1,000 iterations in the simulator. Validation-selected checkpoint 449: validation 97.0000%, final standard test 97.1667%, obstacle test 94.2500%; versus existing final policy checkpoint 699, differences are -0.5833/-1.2500/-1.0000 percentage points (validation/standard/obstacles). Comparison uses identical common levels 0-5, 200 courses per level and final test families 1/4. Exactly one final standard and obstacle test row at checkpoint 449; no evaluation rerun. Verified comparison: `results/operations/g995-ice-s4-comparison.json`. This completion notification is delivered in this chat now; do not repeat. Existing final policy files retained. Archived seed 5 validation through iteration 949 (96.00%), retained best 99 (97.00%). The queue automatically started g995-ice-s6 alongside g995-ice-s5 using the recipe selected from common validation only; exactly two unrestricted trainers, thermal watchdog (65/58 C) and keep-awake remain healthy. Cloud evaluation hold remains in effect. Report/handoff refreshed; independent completion audit and CSV checks pass. No executable changes; the 109-test CPU suite from 03:16 remains applicable.

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

- **28 Oct 2026, 21:38, Codex on training PC:** confirmed retuned planner in real Minecraft: 51/60 standard, 56/60 obstacles, all simulator outcomes matched, zero server corrections; defaults adopted in Python and JS. Evidence archived by report. Shared 70% compute plus conservative per-job allocator caps and compact/batched memory allocation are being checked before queue restart.
- **28 Oct 2026, 21:33, Codex on the training PC:** second merge brings cloud work through `21188a6`, including the CUDA memory-cap device-index fix and shapes/slime/ladders. Preserve the requested 70% shared compute and aggregate memory budget. Compact friction uses one-byte material ids with exact float64 lookup (stone, ice and slime). Initial cap startup failures occurred before any training and are logged.
- **28 Oct 2026, evening, Codex (main session on the training PC):** preserved the PC documentation updates, merged cloud work through `d9237e7`, retained both session histories and the full earlier PC handoff below. Preparing two unrestricted GPU runs (`g995-adv`, `g995-ice`), then `g995-extra`; validation-only checkpoint selection and one final test evaluation per run. Game is closed. When gaming resumes, wait for the next 50-iteration checkpoint and resume each same run from its own latest checkpoint with `--gpu-mem-fraction 0.75 --gpu-duty 0.9 --threads 8`.

Newest first. One entry per session; earlier history is in the
[TEAM_STATUS update log](TEAM_STATUS.md#update-log) and `git log`.

- **29 Oct 2026, night, Claude Code (the same cloud session, branch
  `claude/pensive-knuth-biwj11`, CPU only):** the user put the new-abilities
  test runs ON HOLD (section above; container jobs frozen, nothing on the PC
  touched) and asked for a 3D page of the 45 showcase recordings exported
  on branch `codex/advanced-ice-replay-export-20261029`
  (`exports/advanced-ice-replays/`, data commit `6994797`): offline
  prismarine-physics episodes of the validation-selected g995-adv
  (checkpoint 99) and g995-ice (checkpoint 449) policies, recorded after
  training, not training rollouts or benchmark results. Done:
  `results/showcase_replays.html`, built by `scripts/make_showcase_replays.py`
  (reads the branch with git, checks the files against the export's
  manifest, embeds the data in `scripts/showcase_replays_template.html`) and
  published as a private claude.ai artifact
  (https://claude.ai/artifact/RcaELhrJsAs28YaBGZQ81N; to update it, publish
  the rebuilt page to that URL). It animates the recorded frames at 20 ticks
  per second (policy, level and course selectors, timeline, speeds, follow,
  orbit and whole-course cameras, trail, the action of each tick, the jump
  list, failures with where they happened), labelled "Recorded offline
  simulation — showcase courses"; its counts (advanced 20 of 27 made it, 2
  fell, 5 stalled; ice 17 of 18, 1 fell) are over these recordings only.
  Next: resume the ON HOLD task.

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
  Later the same night: done since then: `all` + brake (official 2:
  147/218; random 13: 119/200), the offline bot (25/31), the extra levels
  (shaped blocks, slime, ladders), `jumpeval --extra`, the mixed set's
  policy-alone row (37/200), `--approach`, and fixes from a review
  workflow. The container restarted twice and lost two 2.5-hour runs, so
  long runs are now restart-safe: `parkour.neo_assist ... --name NAME
  --parts 4 --jobs 4` (rerun the same command after a restart: finished
  parts in `runs/assist_parts/NAME/` are kept) and
  `scripts/extra_preview.py` (keeps finished kinds in
  `runs/extra_preview/`). Running: `all` + brake + approach on official 2
  and random 13 (names `official2_all_brake_approach`,
  `random13_all_brake_approach`).
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
