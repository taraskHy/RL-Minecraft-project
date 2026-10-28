# Handoff for the next assistant session

Read this, then [TEAM_STATUS.md](TEAM_STATUS.md), [docs/design.md](docs/design.md)
and [results/DATA.md](results/DATA.md).

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
Windows 11, clock in JST. Checkout:
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
10 s it reads nvidia-smi; above 65 C it suspends every GPU job of the
project (train, neo_demos, cem, planner, jumpeval, stress, feasibility;
suspended, so no state is lost) and resumes them below 58 C. Log:
`runs/gpu_watchdog.log`. If the watchdog is ever killed while jobs are
paused: `python scripts/gpu_watchdog.py --resume-all`.

## Key facts learned

- Physics: prismarine-physics 1.11.1 ported exactly, **plus vanilla's 1e-7
  collision tolerance** in both the simulator (`COLLISION_EPSILON`) and the
  bot (`bridge/physics_patch.js`). Without it a player resting against a face
  can slip into a block corner through rounding. Parity with the patched
  engine: 1.4e-14 blocks.
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
- Neos: a single neo (wall flush with a 1-wide takeoff) is possible in all
  18 layouts tried; double and triple neos were not found possible from a
  standing start (`configs/jump_cem.json`) nor, so far, with a 3-4 block
  run-up (`scripts/cem_momentum_neo.py`). In courses a neo must take off
  from a 1-wide block. Standing on any obstacle ends an episode as a fall.

## Where things are

- Final policy (obstacles): `results/final_policy.pt` / `.json`.
- Runs: `runs/<name>/` (ignored by Git; logs copied to `results/runs/`).
- Neo training: `runs/advanced-ft3` (levels 0-8, from `advanced-ft2`).
- Sweeps: `scripts/sweep.py configs/<sweep>.json --slots N` (skips
  finished runs).
- Pages: `results/replay_viewer.html`, `results/progress.html`,
  `results/neo_preview.html` (published as private artifacts).
