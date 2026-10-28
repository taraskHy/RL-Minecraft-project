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
**For AI agents and new contributors:** [AGENTS.md](AGENTS.md) (rules) and [HANDOFF.md](HANDOFF.md) (current state, what is left, what not to touch, open problems; kept up to date every session).

## Results so far

Final policy (discount 0.995, `results/final_policy.pt`):

| What | Result | Where measured |
| --- | --- | --- |
| Unseen 10-jump courses, all 6 levels (1,200) | 98.4% (hardest level 93.5%) | simulator |
| Unseen courses with obstacles (1,200) | 95.2% | simulator |
| All 194 jump types of levels 0-5 from a standing start (wide and neo jumps not part of this test) | all made (hardest level 99.1%) | simulator |
| The bot's JavaScript on Mineflayer's physics engine | 59/60 standard, 57/60 obstacle, 3/4 hand-designed (misses the neo course); identical to the simulator | prismarine-physics, no server |
| Physics port against the real library | 1.4e-14 blocks over 99,000 ticks (`results/physics_parity.json`) | Node + prismarine-physics |
| **Real Minecraft 1.21.4 server** | **59/60 standard, 57/60 obstacle, 3/4 hand-designed; identical to the simulator, 0 server corrections** | real game, Mineflayer bot |
| No course list: route planner + final policy | 90.9% standard, 88.1% obstacles; 50/60 and 55/60 in real Minecraft | simulator, real game |
| Neo jumps: separate neo policy (imitation + DAgger) | 29.0% of 210 held-out neo courses (from 0%); 11/31 neo courses in real Minecraft | simulator, real game |

The first final policy (discount 0.99, `results/final_policy_v1.pt`) made 97.1% and
93.6% in the simulator and 59/60, 58/60 and 3/3 in real Minecraft.

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
