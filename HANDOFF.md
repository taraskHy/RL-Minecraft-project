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

## 2. Current state (28 Oct 2026, commit `a7a426c`)

Every number below is in `results/README.md` or `results/neo/`, and says where
it was measured.

| What | File / run | Simulator (1,200 held-out courses) | Real Minecraft 1.21.4 |
| --- | --- | --- | --- |
| **Final policy** (discount 0.995) | `results/final_policy.pt` / `.json` = `runs/obstacles-g995` best, iteration 699 (validation 97.6) | 98.4% standard (level 5: 93.5), 95.2% obstacles (level 5: 81.0) | 59/60 standard, 57/60 obstacles, 3/4 hand-designed (the miss is the neo course, not in its training); 0 server corrections; same outcome as the simulator on every course |
| First final policy (discount 0.99) | `results/final_policy_v1.*` = `runs/obstacles-ft` | 97.1%, 93.6% | 59/60, 58/60, 3/3 |
| **New map, no course list**: greedy route planner + final policy | `parkour/planner.py` = `bridge/planner.js` | 90.9% standard, 88.1% obstacles | 50/60, 55/60 |
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
- Two policies, not one: the final policy for levels 0-5 and obstacles, the
  neo policy for levels 6-8. Merging them was planned
  (`docs/overnight_plan.md`) but not done.
- What is running on the PC right now cannot be known from outside it: the
  last commits record finished results only. Check the processes there.

## 3. What is left to do

Concrete work, most important first. Nothing here needs new research.

1. **Report** (`docs/report_draft.md`, for the course): 7.3 to 7.9 now
   report the final policy (first final policy in brackets). Still to do:
   7.10 stops at "demonstrations are the natural next step" and needs the
   imitation + DAgger results (`results/README.md` section "Neo jumps",
   `results/neo/`); 7.11 and 7.12 need the final policy's planner numbers
   next to the first policy's; section 8 is bullet points; section 9
   (conclusion) is "to be written".
2. **`final-v2`** (`results/runs/final-v2`: 3,000 iterations planned, stopped
   at 499, discount 0.99, obstacles 30%) is not described in any document.
   Ask the PC session or the user what it was, then document it or mark it
   aborted.
3. **Ship the neo policy in Git** if the team will present it: it exists
   only as `runs/neo-dagger3/checkpoints/best.pt` on the PC. Export it next
   to the final policy (for example `results/neo_policy.pt` / `.json`)
   without replacing `results/final_policy.*`.
4. **Restart the GPU watchdog on the PC** after merging this branch: it
   now also pauses `parkour.neo_dagger`, but the copy already running uses
   the old job list.
5. Keep `TEAM_STATUS.md` and this file in step with each new result.

## 4. Open problems (unsolved, up for grabs)

1. **Neo jumps are still mostly failed.** 29% of neo courses; courses with
   two or more neos almost never (5/81). DAgger round 3 beat round 2 on
   validation but not on the test courses (61 vs 63 of 210).
   Untried directions: more DAgger rounds with more arrival states, a
   separate neo skill the main policy hands over to, a curriculum of neo
   tightness. Keep picking checkpoints on the validation family.
2. **One policy for everything:** the final policy fails the neo course,
   and the neo policy was never run through the final policy's checks (the
   60 + 60 offline and real-Minecraft courses, the stress tests). Whether
   one policy can do both is open.
3. **Hardest level:** level 5 is 93.5% standard and 81.0% with obstacles;
   most remaining failures of the final policy are there.
4. **Planner shortcuts:** most planner failures take a shortcut to a later
   platform that is in reach, and that jump is harder
   (`results/planner/`, report 7.11).
5. **Learned no-route model and obstacles:** 10 of its 23 obstacle-course
   falls end on top of an obstacle (6 on a beam, 4 on a decoy pillar).
6. **More block types:** slabs, ladders, ice, fences and water need physics
   ports (prismarine-physics has them), parity tests and course pieces.
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
| `parkour/physics.py` | exact float64 port of prismarine-physics 1.11.1 (operand order kept) plus vanilla's 1e-7 `COLLISION_EPSILON` | `tests/test_physics_parity.py`, `test_corner_slip.py`, `test_inside_block.py` |
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
python -m parkour.planner --checkpoint results/final_policy.pt [--obstacles] [--family 2]   tune on 2, report 1
python -m parkour.jumpeval --checkpoint CKPT --out FILE [--advanced]   every jump type alone
python -m parkour.stress --checkpoint CKPT --out FILE    course styles outside training
run.cmd report                                           refresh results/ (and this file's snapshot)
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
feasibility, cem_momentum_neo; suspended, so no state is lost) and resumes them below
58 C. Since 28 Oct afternoon it also covers `parkour.neo_dagger` (restart
the watchdog to pick that up); the live bot is left alone on purpose (it
plays in real time). Log:
`runs/gpu_watchdog.log`. If the watchdog is ever killed while jobs are
paused: `python scripts/gpu_watchdog.py --resume-all`.

**Anywhere else** (a cloud container, another model's sandbox): no GPU, no
torch, no `runs/`, no Minecraft server. What works there: reading and
editing documents, analysing the files in `results/`, and
`python -m parkour.handoff` (standard library only). Training, evaluation
and live runs happen on the PC.

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
- Route planner results: `results/planner/` and the `planner-*`,
  `v2-planner-*` games.
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

Generated on top of commit `60924b6` (2026-10-28 12:52, branch `claude/pensive-knuth-biwj11`): Index every results file in DATA.md; GPU watchdog also pauses neo_dagger

| Document | Last change | Commits since |
| --- | --- | --- |
| `AGENTS.md` | `a37dea9` 2026-10-28 12:26 | 1 |
| `HANDOFF.md` | updated together with this snapshot | 0 |
| `TEAM_STATUS.md` | `a37dea9` 2026-10-28 12:26 | 1 |
| `README.md` | `a37dea9` 2026-10-28 12:26 | 1 |
| `docs/design.md` | `ed915f0` 2026-10-28 10:02 | 6 |
| `docs/report_draft.md` | updated together with this snapshot | 0 |
| `results/README.md` | `6170803` 2026-10-28 11:40 | 3 |
| `results/DATA.md` | `60924b6` 2026-10-28 12:52 | 0 |

<!-- snapshot data: compared by `python -m parkour.handoff --check` -->

### Shipped policies

Simulator, held-out test sets (1,200 courses each), checkpoint picked on the validation family.

| File | Checkpoint | Iteration | Validation | Test, standard | Test, obstacles |
| --- | --- | --- | --- | --- | --- |
| `results/final_policy.pt` / `.json` | `runs/obstacles-g995/checkpoints/best.pt` | 699 | 97.6 | 98.4 | 95.2 |
| `results/final_policy_v1.pt` / `.json` | `runs/obstacles-ft/checkpoints/best.pt` | 999 | 96.6 | 97.1 | 93.6 |

### Training runs (`results/runs/*/summary.json`)

Simulator, held-out test set. "Picked" is the validation-selected checkpoint (`best.pt`); runs without one predate validation selection. Runs trained on levels 0-8 (`advanced-*`, `neo-*`) are evaluated on levels 0-8, the others on levels 0-5, so the two groups are not comparable.

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

- **28 Oct 2026, Claude Code (cloud session, continued, branch
  `claude/pensive-knuth-biwj11`):** the user asked this session to keep
  working through section 3 on its own (no questions; they are away from
  the PC), pushing after every step so the next model can continue. Done
  so far: `results/DATA.md` indexes everything in `results/` (the snapshot's
  "not yet indexed" list is empty); the GPU watchdog also pauses
  `parkour.neo_dagger`; report 7.3-7.9 updated to the final policy. In
  progress: report 7.10-7.12, then sections 8 and 9. Decision: CPU torch
  cannot be installed here (`download.pytorch.org` is blocked by this
  environment's network), so no new numbers are computed in this session;
  everything comes from files already in `results/`. This
  branch is not merged into `main`; the PC session should merge it.
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
