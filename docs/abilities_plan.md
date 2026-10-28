# More parkour abilities: plan and progress

Started 28 Oct 2026 (evening) by the cloud session on branch
`claude/pensive-knuth-biwj11`. The user asked for more of the bot's parkour
abilities (other blocks, more jump types such as head hitters) and for neos
made consistently on a random course, while they play a game on the PC's
GPU (first "leave a safe buffer", then "you can use more GPU, don't leave
like a 40% margin": about 10%). The cloud session has no GPU and cannot
reach the PC, so it builds, verifies and measures on the CPU, and prepares
the training for the PC (gaming settings, HANDOFF section 6).

Rules that shape every stage (AGENTS.md, HANDOFF section 5): every feature
is opt-in and off by default; the held-out test and validation courses stay
byte-identical (`tests/test_heldout_fingerprints.py`); new jump kinds go
into new levels after level 8, never into the existing level lists; what the
bot mirrors gets a parity test; settings are chosen on validation families
(2, 5), test families (1, 4) are reported once; every number comes from a
file in `results/` and says where it was measured.

| Stage | What | Status |
| --- | --- | --- |
| 0 | Held-out fingerprint guard; gaming-safe GPU flags (`--gpu-mem-fraction`, `--gpu-duty`, `--threads`, watchdog `--pause-while`) | done |
| A | Neo assist: an exact-model search at every neo takeoff, in the simulator (`parkour/neo_assist.py`) and both bots (`bridge/assist.js`, `parkour.live --assist`); options `--jumps all` (every jump), `--brake` (stop and retry after a failed search), `--approach` (one plan over the two jumps before a neo first), per-kind plan targets and horizons | built; validation (official, family 2): `advanced` 124/218 (batch 1; 119 batched), with the policy prior 111/218 (prior stays 0); `all` + brake running, test run after it |
| A2 | The assist on random courses made of ordinary jumps plus neos (new seed families 12 test, 13 validation), and a mixed set with every special kind (14 test, 15 validation) | validation (random, family 13): policy alone 27/200, `all` 106/200; `all` + brake and `all` + brake + approach running; mixed after the extra level file |
| Q | PC training queue with gaming settings (`configs/sweep_new_skills.json`: `extra-ft`, `advanced-g995`) | written; `extra-ft` smoke-tested on the CPU (3 iterations) |
| B | Head hitters (2-high roofs over a runway, tunnels) and ice run-ups as new jump kinds, opt-in levels, feasibility-checked | done (zero-shot results in `results/extra/`; training is for the PC) |
| E1 | Slime bounce: physics (opt-in, parity 1.4e-14), slime pads in courses and both bots | done |
| C | Block shapes in the physics (slabs, soul sand, fence and wall posts, ...), opt-in, parity | done (1.4e-14 blocks, `results/physics_parity_shapes.json`) |
| D | Shaped platforms end to end (courses, observation and rules mirrored in `bridge/policy.js`, both bots) | done (route planner refuses shaped courses) |
| E2 | Ladders and vines: physics (opt-in, parity 1.4e-14); ladder climbs in courses and both bots | done (vines only in the physics) |
| F | A learned neo from DAgger rounds with the assist's search as expert (data here, training on the PC) | built: `scripts/neo_pipeline.py` (rounds chosen on validation; whole chain smoke-tested on the CPU); rounds for the PC |

## Design notes from the review of the code (28 Oct)

- **Observation and ceilings.** The scan's ceiling flag is "solid 2 or 3
  rows above the feet cell", so a 2-air ceiling over a gap looks exactly like
  the 3-air head-hitter beam that obstacle courses already have. Beams never
  sit over platforms, so a ceiling over the runway is a new, visible pattern.
  Head hitters are therefore runway composites (like the momentum jumps):
  a run-up under a 2-high ceiling, optionally continued over the gap.
  A ceiling over a landing would sit in the next jump's flight corridor
  (`courses._fits`), so it is left out.
- **Rise under a 2-high ceiling** is capped at 0.2 blocks, so head-hitter
  jumps only go level or down.
- **Curriculum.** The adaptive curriculum never reached level 8 in the
  advanced runs (level 7 stayed below the 0.6 needed to advance), so levels
  after 8 need a level-subset option for training (`--levels`).
- **Obstacle cells.** `MAX_EXTRA` (256 cells per course) is not checked by
  the generator; new kinds that add ceiling cells must check it.
- **Ice in specs.** A training pool built without an ice grid cannot take
  icy courses later (`CoursePool.replace` raises), so pools are built with
  the grid whenever the level table has ice kinds.
- **Block shapes.** Stairs have 2-3 boxes, fences up to 3, walls up to 5;
  fence and wall shapes depend on their neighbours (so only isolated posts);
  carpets, ladders and vines drop without support; snow layers report
  `boundingBox: 'empty'` although they have boxes.
- **Neo search prior.** `neo_dagger --prior` defaults to 0.95; for a policy
  that stands still at neos that prior means "stand still", so the assist
  tunes it, including 0.
- **Version features.** prismarine-physics decides version-dependent rules
  from its own table (`bridge/node_modules/prismarine-physics/lib/features.json`),
  not minecraft-data's: for 1.21 soul sand and honey do not slow the player,
  trapdoors over ladders are not climbable, and pressing jump on a ladder
  climbs. The port follows that table.
- **Course pieces for new blocks** are composites that end on stone (shaped
  blocks: an exit jump; slime: an exit platform above the pad; ladders: the
  target on top of the wall), so the other jumps of a course keep the
  difficulty the feasibility study measured.
- **Planning the new kinds** (`--jumps all` plans every jump). A slime jump
  is planned from its takeoff to the exit after the pad in one plan
  (`neo_assist.plan_target`): how high the bounce goes depends on how the
  pad is hit, so a plan that only lands on the pad can leave the exit out of
  reach. Plans get at least 80 ticks for slime jumps and 64 for ladder
  climbs (`plan_horizon`; climbing 5 blocks takes about 43 ticks from a
  standstill, and a 32-tick search found no plan). The bots' check of a plan
  (`bridge/assist.js verifyPlan`) uses the shapes' tops and standing areas,
  like `CoursePool.top` / `.foot`.
- **Course files of hand-designed courses** (`courses.from_specs`) list only
  their steps, not one spec per jump as generated courses do (no runway
  steps, no exits); `neo_assist.specs_from_keys` puts those back, so the
  assist plans the same jumps on both.
