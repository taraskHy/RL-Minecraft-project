# Minecraft parkour with reinforcement learning

A bot that learns to run **parkour courses it has never seen** in Minecraft
Java 1.21.4, and reach the goal.

The agent is trained with PPO in a GPU simulator that reproduces Mineflayer's
own physics engine to about 1e-14 blocks, on endlessly generated courses of
increasing difficulty. The trained network then drives a Mineflayer bot on a
real Minecraft server, on new courses.

> The earlier course project (wood and stone tools in Survival) is archived
> on the `archive/survival` tag and the `codex/*` branches.

**Status:** see [TEAM_STATUS.md](TEAM_STATUS.md) (updated by hand, so it can lag behind [results/README.md](results/README.md), which is generated from the run files).
**Design and method:** [docs/design.md](docs/design.md).
**Assistant handoff:** [HANDOFF.md](HANDOFF.md).

## Results so far

| What | Result | Where measured |
| --- | --- | --- |
| Unseen 10-jump courses, all 6 levels (1,200) | 97.1% (hardest level 90.5%) | simulator |
| Unseen courses with obstacles (1,200) | 93.6% | simulator |
| All 194 jump types of levels 0-5 from a standing start (wide and neo jumps not part of this test) | all made (hardest level 99.6%) | simulator |
| The bot's JavaScript on Mineflayer's physics engine | 59/60 standard, 58/60 obstacle, 3/3 hand-designed; identical to the simulator | prismarine-physics, no server |
| Physics port against the real library | 1.4e-14 blocks over 99,000 ticks (`results/physics_parity.json`) | Node + prismarine-physics |
| **Real Minecraft 1.21.4 server** | **59/60 standard, 58/60 obstacle, 3/3 hand-designed; identical to the simulator, 0 server corrections** | real game, Mineflayer bot |

Details, figures, the reward ablation and the other experiments:
[results/README.md](results/README.md). 3D replays of the real-game runs: open
`results/replay_viewer.html` in a browser. Ghost races of ten training stages on the same
unseen course: `results/theatre.html` (Overview camera), rendered to 1080p MP4 in
`results/videos/` by `scripts/render_videos.py`. Final policy in Git:
`results/final_policy.pt` (weights, for `run.cmd offline` / `run.cmd live --checkpoint`) and
`results/final_policy.json` (the exported network the bot loads).

## How it works

```text
parkour/physics.py      batched port of prismarine-physics (the engine Mineflayer moves bots with)
parkour/courses.py      procedural courses: chains of platforms with gaps, height changes, turns
parkour/feasibility.py  which jumps are possible, and how forgiving each one is -> 6 difficulty levels
parkour/env.py          batched GPU environment: observation, reward, termination, curriculum
parkour/model.py        actor-critic network, exported to JSON for the bot
parkour/train.py        PPO training and held-out evaluation
parkour/live.py         run a policy in real Minecraft (or offline on Mineflayer's physics) and compare
parkour/server.py       local void-world Minecraft server
bridge/policy.js        the bot's observation, rules and network (mirrors the Python code)
bridge/parkour_bot.js   the live Mineflayer bot
bridge/offline_bot.js   the same bot loop on prismarine-physics without a server
tests/                  physics parity with the real engine, JS/Python parity, env and course checks
```

## Quick start (Windows)

```bat
setup.cmd                 :: Python packages, portable Node/Java, the Minecraft 1.21.4 server jar
run.cmd test              :: all checks, including physics parity against the real engine
run.cmd train --run my-first-run --iterations 300
```

A 300-iteration run (157 million game ticks) takes 10 to 18 minutes on an RTX
5080 (15.6 for `runs/main-s0`) and already solves most easy and medium unseen
courses. Results go to `runs/my-first-run/` (`train_log.csv`, `eval_log.csv`,
checkpoints and the exported `policy_best.json`).

Play held-out courses with the trained policy:

```bat
run.cmd offline --checkpoint runs\my-first-run\checkpoints\best.pt --out runs\offline-check
run.cmd accept-eula --agree    :: once per PC, only if you accept https://aka.ms/MinecraftEULA
run.cmd live --checkpoint runs\my-first-run\checkpoints\best.pt --out runs\live-check --viewer YourMinecraftName
```

With `--viewer`, join `127.0.0.1:25570` in Minecraft Java 1.21.4. The bot puts
you in spectator mode, and from the second course on your camera follows it.

## Rules that keep the results honest

- Every reported number comes from files under `runs/` or `results/`, and
  says whether it is the simulator, the offline JavaScript bot, real
  Minecraft, or (for physics parity) the real physics library.
- Test courses come from a separate seed family and never appear in training.
- Course building and the start teleport use server commands (the
  environment's reset); during an episode the bot only uses movement
  controls.
