# Instructions for AI agents

For any model or tool working in this repository (Claude Code reads it
through `CLAUDE.md`). The project: a PPO agent, trained in a GPU port of
Mineflayer's physics engine, that completes **unseen** Minecraft parkour
courses; the same network then drives a Mineflayer bot on a real Minecraft
1.21.4 server. A student course project (team of three).

## Start here

1. Read [HANDOFF.md](HANDOFF.md): current state, what is left, what not to
   touch, open problems, and the session log.
2. Run `python -m parkour.handoff --check` to see what changed since the
   handoff was last updated (standard library only; works anywhere).
3. Then, as needed: [TEAM_STATUS.md](TEAM_STATUS.md),
   [docs/design.md](docs/design.md), [results/README.md](results/README.md),
   [results/DATA.md](results/DATA.md).

## Rules

- **One session changes the project:** the main session on the training PC.
  Before editing, committing or starting a run, check `git log` and the
  running processes. Elsewhere, push only to the branch you were given.
- Never force-push or rewrite history. Never delete runs, results or
  recordings.
- Every number says where it was measured: simulator, offline bot
  (prismarine-physics without a server), real Minecraft, or the physics
  library. Every number comes from a file under `runs/` or `results/`.
- Checkpoints are picked on the validation family (2, obstacles 5). The test
  families (1 standard, 4 obstacles) are only reported, never used to choose.
- Do not change the parts listed in HANDOFF.md section 5 ("Do not touch");
  add opt-in flags instead and run the listed tests.
- GPU (training PC): at most two training processes at once; above 65 C the
  GPU work pauses (`scripts/gpu_watchdog.py`, a rule from the user).
- After each result: `python -m parkour.report` (refreshes `results/` and the
  handoff snapshot), update `TEAM_STATUS.md`, commit, push.
- Ask the user only about direction or their own decisions; they manage at a
  high level.

## Keep the handoff current

Before you end a session, or after each result:

1. Edit the hand-written sections of `HANDOFF.md` that are no longer true.
2. Add one line at the top of its session log: date, who you are (model or
   tool, and where you ran), what changed, what is next.
3. Run `python -m parkour.handoff` and commit `HANDOFF.md` with the work.

## Commands (Windows PC, from the repository root)

```text
setup.cmd                 Python venv, portable Node and Java, Minecraft server jar
run.cmd test              all checks (physics parity needs Node from setup.cmd)
run.cmd train --run NAME  PPO training (flags: parkour/train.py TrainConfig)
run.cmd offline / live    play held-out courses with the bot (live = real Minecraft)
run.cmd report            refresh results/ and the handoff snapshot
run.cmd handoff           refresh only the handoff snapshot
```

Without the PC (a cloud container or sandbox) there is no GPU, `runs/` or
Minecraft server. Documents, `python -m parkour.handoff` and analysis of
`results/` always work; with PyTorch from PyPI and the bridge packages
(HANDOFF.md section 6 has the commands) the test suite and CPU evaluations
with the checkpoints in `results/` run too. Training and real-game runs need
the PC. A real Minecraft server also needs the user's EULA acceptance on
that machine: do not accept it for them.
