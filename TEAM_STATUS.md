# Team status

One page for everyone on the team: what works, what is running, what is open.
Updated and pushed after every result. Every number is in
[results/README.md](results/README.md) (generated from the run files).

**Last updated:** 27 October 2026, 23:45.

## The project now

We changed direction on 27 October 2026. The goal: **a bot that learns
Minecraft parkour and completes courses it has never seen.** The survival
work (wooden pickaxe, stone design) is archived on the `archive/survival` tag
and the `codex/*` branches.

## Headline results (simulator unless stated)

- **Real Minecraft works.** On a local 1.21.4 server the bot completed
  **59/60 unseen courses, 58/60 unseen obstacle courses and 3/3 hand-designed
  courses**, exactly the courses the simulator predicted. The server corrected
  its position 0 times in 123 episodes, and replaying its recorded moves in the
  simulator reproduces every real trajectory to 2e-14 blocks.
- **Final policy** (`runs/obstacles-ft`, main policy fine-tuned with
  obstacles): **97.1% of 1,200 unseen test courses** (level 5: 90.5%) and
  **93.6% of unseen courses with obstacles** (head-hitter beams, decoy pillars).
- **Main run** (no obstacles, 1.57 billion ticks, 2.6 hours): 95.5% of the
  test courses at its last checkpoint.
- **Every possible jump type** is made from a standing start at all 16 start
  angles (level 5: 99.6%); 4 types that no scripted strategy makes are made
  sometimes, one of them every time.
- **New course styles:** 15-jump courses 95%, any start direction 97%,
  single-block courses 91%, a turn at every jump 90%.
- **The bot's real code agrees with the simulator:** the JavaScript bot on
  Mineflayer's physics engine (no server) completed 59/60 standard, 58/60
  obstacle and 3/3 hand-designed courses, exactly the simulator's results,
  trajectories equal to 7e-15 blocks.
- **Your reward ideas, tested** (3 seeds each): the *fall penalty* (height)
  matters most (without it the hardest level drops to 54-57% from 69-77%);
  *distance shaping* makes learning about 3 times faster but does not change
  the final score; the *landing bonus* and the *curriculum* show no reliable
  effect at this budget.
- **Without being told the route** (only the blocks around it and the goal
  direction) the agent still completes 82% of test courses (3 seeds).
- **3D replays:** open `results/replay_viewer.html` in a browser (123 courses).

## How we know the simulator is right

Mineflayer bots move with a physics library called prismarine-physics. We
ported it to the GPU and check it against the real library on random worlds:
largest difference 1.4e-14 blocks over 99,000 ticks. `run.cmd test` runs this.

## Running now

- **Harder jumps** (requested): neo, double neo, 4-block momentum jumps and
  wide diagonals are built (levels 6-8). A search over action sequences is
  checking which neo layouts are physically possible; training on them follows.

- The hyperparameter study (`configs/sweep_hparams.json`): entropy, discount,
  network size, KL target and number of environments, one at a time, compared
  on the validation set. Results appear in `results/README.md`.

## Waiting on the PC owner

- **GitHub sign-in** on this PC, so the work can be pushed.

## Open problems anyone can pick up

1. More block types: slabs, ladders, ice and fences need physics ports
   (prismarine-physics has them) and new course pieces.
2. Hand-built maps: extract the platform chain from a region of a real map.
3. Presentation footage: record a `run.cmd live --viewer` session.
4. The report: `docs/report_draft.md` is complete except the hyperparameter
   and real-Minecraft results (7.3, 7.9), the discussion and the conclusion.

**Try the final policy without training:** `results/final_policy.pt` (weights)
and `results/final_policy.json` (what the bot loads), for example
`run.cmd offline --checkpoint results\final_policy.pt --out runs\my-check`.

## Update log

- **27 Oct 2026, 23:45:** EULA accepted by the PC owner; first real-Minecraft
  runs: 59/60, 58/60 (obstacles) and 3/3, identical to the simulator. Neo,
  double-neo, momentum and wide jumps added.

- **27 Oct 2026, 23:00:** ablation (15 runs), obstacle fine-tune and no-route
  runs finished; the obstacle-trained policy is the final policy (97.1% /
  93.6%). Report draft section 7 filled in. Hyperparameter study running.

- **27 Oct 2026, evening:** main run finished (95.5% held-out). Analyses,
  replay viewer and offline bot check done. PPO stabilised after early
  collapses (first runs discarded). Obstacles, validation-based checkpoint
  selection and the hyperparameter study added.
- **27 Oct 2026, afternoon:** new direction (parkour). Simulator, physics
  parity, feasibility study, environment, PPO, JS bot and offline check built
  and tested.
