# Team status

One page for everyone on the team: what works, what is running, what is open.
Updated and pushed after every result. Every number is in
[results/README.md](results/README.md) (generated from the run files).

**Last updated:** 27 October 2026, evening.

## The project now

We changed direction on 27 October 2026. The goal: **a bot that learns
Minecraft parkour and completes courses it has never seen.** The survival
work (wooden pickaxe, stone design) is archived on the `archive/survival` tag
and the `codex/*` branches.

## Headline results

- **95.5% of 1,200 unseen test courses** (10 jumps each, 6 difficulty levels)
  after 1.57 billion game ticks of PPO training, about 2.6 hours on the RTX
  5080: levels 0 to 4 at 97 to 100%, the hardest level at 80%. *Simulator,
  last checkpoint, no checkpoint picking.*
- **Every possible jump type** at levels 0 to 5 is made from a standing start
  at all 16 tested start angles (one type at 94%), and 3 jump types that none
  of 4,032 scripted strategies could make are made sometimes.
- **The bot's real code agrees with the simulator.** The JavaScript bot,
  running on Mineflayer's own physics engine without a server, finished 58 of
  60 held-out courses, the same courses as the simulator, with trajectories
  equal to 9e-16 blocks. All three hand-designed demo courses
  (`configs/showcase_courses.json`) are completed.
- **Generalisation beyond training** (100 new courses per level each):
  15-jump courses 93.5%, any start direction 95%, single-block courses 89%,
  constant zigzags 85%. Courses with obstacles (head-hitter beams, decoy
  pillars) drop it to 47%, so an obstacle fine-tune is running.
- **3D replays** of the bot on unseen and hand-designed courses:
  `results/replay_viewer.html` (open in a browser).

## How we know the simulator is right

Mineflayer bots move with a physics library called prismarine-physics. We
ported it to the GPU and check it against the real library on random worlds:
largest difference 1.4e-14 blocks over 99,000 ticks. `run.cmd test` runs this.

## Running now (overnight)

1. Reward and curriculum ablation, 3 seeds per arm (`configs/sweep_ablation.json`):
   - `full`: all reward terms (main settings);
   - `sparse`: no distance shaping, landings and goal only ("count the jumps");
   - `nolanding`: no landing bonus ("distance only");
   - `nofall`: no fall penalty ("no height term");
   - `uniform`: no curriculum.
   First seeds: distance shaping makes learning about 2.5 times faster but is
   not needed for the final score; the landing bonus matters most for the
   hardest level (47% without it against 74 to 80%). Waiting for all seeds.
2. `obstacles-ft`: the main policy fine-tuned with obstacles.
3. `noroute`: the bot is not told where the platforms are; it sees only the
   blocks around it and the goal direction.
4. A hyperparameter study (`configs/sweep_hparams.json`), compared on a
   separate validation set.

## Waiting on the PC owner

- **Minecraft EULA** (`run.cmd accept-eula --agree`, only if you accept it).
  Then `run.cmd live ... --viewer YourName` runs the bot in real Minecraft.
- **GitHub sign-in** on this PC, so the work can be pushed.

## Open problems anyone can pick up

1. More block types: slabs, ladders, ice and fences need physics ports
   (prismarine-physics has them) and new course pieces.
2. Hand-built maps: extract the platform chain from a region of a real map.
3. Presentation footage: record a `run.cmd live --viewer` session.
4. The report: `docs/report_draft.md` has sections 1 to 6; section 7 comes
   from `results/README.md`.

## Update log

- **27 Oct 2026, evening:** main run finished (95.5% held-out). Analyses,
  replay viewer and offline bot check done. PPO stabilised after early
  collapses (first runs discarded). Obstacles, validation-based checkpoint
  selection and the hyperparameter study added.
- **27 Oct 2026, afternoon:** new direction (parkour). Simulator, physics
  parity, feasibility study, environment, PPO, JS bot and offline check built
  and tested.
