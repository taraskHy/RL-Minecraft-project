# Handoff for the next assistant session

Read this, then [TEAM_STATUS.md](TEAM_STATUS.md) and [docs/design.md](docs/design.md).

## Goal and how the user works

- Goal (since 27 Oct 2026): RL bot that completes **unseen** Minecraft
  parkour courses and reaches the target. The earlier survival project is
  archived (`archive/survival` tag = old `codex/pc-canopy-descent` head;
  `codex/*` branches untouched).
- The user manages at a high level and wants as few questions as possible.
  Do routine work autonomously; ask only about direction or anything that is
  the user's own decision (e.g. the Minecraft EULA).
- Team of three students; course-level methods (PPO, actor-critic and GAE are
  in the course material).
- Repository: <https://github.com/taraskHy/RL-Minecraft-project>, branch
  `main`. Never force-push. Push and update TEAM_STATUS after each result.

## Machine

Intel Core Ultra 7 270K Plus (24 cores), 128 GB RAM, RTX 5080 16 GB,
Windows 11. Checkout: `C:\Users\ethan\OneDrive\Desktop\etgar\RL\RL-Minecraft-parkour`
(OneDrive folder; checkpoint saves retry on file locks). Python 3.12 venv in
`.venv` (torch 2.11.0+cu128). Portable Node 24.22.1, Java 21 and the 1.21.4
server jar in `.local` (ignored by Git).

## Minecraft EULA

Not recorded for this project. A consent file from 23 Oct 2026 exists in the
old folder `RL-Minecraft-project/.local`, but the agreed rule is to ask the
user on every machine. Live runs need `run.cmd accept-eula --agree` from the
user first.

## Key facts learned

- prismarine-physics 1.11.1 ported exactly (float64, same operand order);
  parity tests pass to 1e-14 on CPU and CUDA.
- The env step is synchronisation-free; the simulator is kernel-launch bound
  on Windows (~850 launches per tick), so more environments per process is
  almost free: 16,384 envs give ~1M ticks/s; 3 training processes can share
  the GPU.
- 1.21 servers send a sprinting modifier with a different id from
  prismarine's, so the bot deletes the movement-speed attribute each tick
  (see `bridge/parkour_bot.js`). Verify in the first live run: the bot logs
  the attribute it removed.
- `bot.look` quantises yaw; the bot sets `bot.entity.yaw` directly instead.

## Where things are

- Runs: `runs/<name>/` (`config.json`, `train_log.csv`, `eval_log.csv`,
  `checkpoints/best.pt`, `policy_best.json`, `summary.json`). `runs/` is
  ignored by Git; copy what the report needs into `results/`.
- Sweep runner: `scripts/sweep.py configs/sweep_ablation.json --slots 2`.
- Offline end-to-end check: `run.cmd offline --checkpoint ... --out ...`.
