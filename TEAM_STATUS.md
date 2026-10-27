# Team status

One page for everyone on the team: what works, what is running, what is open.
Updated and pushed after every result.

**Last updated:** 27 October 2026, afternoon.

## The project now

We changed direction on 27 October 2026. The new goal: **a bot that learns
Minecraft parkour and completes courses it has never seen.** The survival
work (wooden pickaxe, stone design) is archived on the `archive/survival` tag
and the `codex/*` branches; nothing of it is needed for parkour.

## What works (verified)

- **Simulator = Mineflayer's physics.** Our GPU simulator reproduces
  prismarine-physics, the engine a Mineflayer bot moves with, to 1.4e-14
  blocks on 99,000 random test ticks, with every collision flag matching.
  `run.cmd test` checks it against the real library.
- **Jump feasibility table.** 243 jump types tested with 4,032 scripted
  strategies each; 194 are possible from a standing start. They define six
  difficulty levels (`configs/jump_levels.json`, `docs/design.md` section 4).
- **PPO learns fast.** In the first test, 70 seconds of training (21 million
  game ticks) already finished 91 to 97% of *unseen* 10-jump courses at
  levels 0 to 3 (simulator).
- **The bot's JavaScript stack matches the simulator exactly.** Run offline on
  Mineflayer's own physics engine, the bot finished 28 of 30 held-out courses,
  the same courses, with the same tick counts, as the simulator; recorded
  trajectories match to 7e-15 blocks.

## Running now

- `main-s0`: the main training run, 3,000 PPO iterations (1.57 billion
  ticks), adaptive curriculum, full reward.
- An ablation sweep (`configs/sweep_ablation.json`), 3 seeds per arm, 1,000
  iterations each:
  - `full`: progress + landing + goal + fall penalty (main settings);
  - `sparse`: no distance shaping (landings and goal only: "count the jumps");
  - `nolanding`: no landing bonus (distance only);
  - `nofall`: no fall penalty (no height term);
  - `uniform`: no curriculum (all levels equally often).

## Not done yet

- **Real Minecraft run.** Everything is ready (`run.cmd live`), but the
  Minecraft EULA must first be accepted on this PC by its owner
  (`run.cmd accept-eula --agree`).
- Report figures and tables from the finished runs.

## Open problems anyone can pick up

1. Obstacles: ceilings (head-hitters), walls, ladders, slabs and ice need
   physics ports (prismarine-physics has them) and new course pieces.
2. Hand-built maps: extract the platform chain from a region of a real map so
   the bot can run a course a person built.
3. Momentum jumps: level 5 jumps (4-block flat gaps) are rarely possible from
   a standing start; does the agent learn to carry speed between jumps?
4. Presentation footage: record a `run.cmd live --viewer` session.

## Update log

- **27 Oct 2026:** new direction (parkour). Simulator, physics parity,
  feasibility study, environment, PPO, JS bot and offline check built and
  tested; main run and ablation sweep started.
