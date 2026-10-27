# Team status

One page for everyone on the team: what works, what is running, what is open.
Updated and pushed after every result. Every number is in
[results/README.md](results/README.md) (generated from the run files); every
piece of data is listed in [results/DATA.md](results/DATA.md).

**Last updated:** 28 October 2026, 00:10.

## The project

**A bot that learns Minecraft parkour and completes courses it has never
seen.** We changed direction to this on 27 October 2026; the survival work
(wooden pickaxe, stone design) is archived on the `archive/survival` tag and
the `codex/*` branches.

How it works, in one paragraph: a Mineflayer bot moves with a physics library
called prismarine-physics. We ported that library to the GPU (it matches the
real one to 1e-14 blocks), generate endless random parkour courses at six
difficulty levels, and train a neural network with PPO on 16,384 courses at
once (about a million game ticks per second). The trained network then drives
the Mineflayer bot on a real Minecraft server. Details: `docs/design.md`.

## Results

**Real Minecraft (1.21.4 server), final policy, courses never seen in training:**

| Courses | Completed | Same as the simulator? |
| --- | --- | --- |
| standard, 10 jumps, all levels | **59 / 60** | yes, every course and tick count |
| with obstacles (head-hitter beams, decoy pillars) | **58 / 60** | yes |
| hand-designed (tour, stairs, sprint) | **3 / 3** | yes |

The server corrected the bot's position 0 times in 123 episodes; replaying the
recorded moves in the simulator reproduces every real trajectory to 2e-14
blocks.

**Simulator, 1,200 held-out courses per test set:**

- Final policy: **97.1%** of standard courses (hardest level 90.5%) and
  **93.6%** with obstacles.
- Training from scratch (main run, 2.6 hours): success goes from 0% (the
  untrained bot falls off the first block; mean reward -5.4) to 95.5% (mean
  reward +53); falls drop from 100% to about 5%. See
  `results/figures/training_main-s0.png` (success, reward, losses, entropy,
  KL and learning rate, every iteration).
- Every possible jump type is made from a standing start at every tested angle.
- New course styles: 15-jump courses 95%, any start direction 97%, single
  blocks 91%, a turn at every jump 90%.

**Our reward ideas, tested (3 seeds each, `results/figures/ablation_curves.png`):**

- *Height / fall penalty*: matters most. Without it the hardest level drops to
  54-57% from 69-77%, in every seed.
- *Distance to the goal* (progress reward): learning about 3 times faster;
  same final score.
- *Number of jumps* (landing bonus) and the *curriculum*: no reliable effect
  at this budget.
- Not told where the platforms are (only the blocks it sees and the goal
  direction): 82% instead of 93%.

**Problems we found and fixed** (good presentation material):

- PPO collapsed twice in the first runs (success 80% to 15% in one update):
  fixed with a KL stop, dual clipping and a KL-adaptive learning rate
  (`results/figures/training__v1_unstable_main-s0.png` shows the collapse).
- Minecraft centres whole-number `/tp` coordinates on the block, which broke
  the first live run: the bot now sends decimals.
- The double-neo search started the player inside the wall (Minecraft lets
  you walk out of a block you overlap), so its first "proofs" were invalid;
  starts inside blocks are now rejected and tested.

## Running now

1. **Harder jumps (neo, double neo, 4-block momentum jumps, wide diagonals)**,
   levels 6-8: training `runs/advanced-ft` from the final policy. At iteration
   100 it completes about 45% of level-6 courses; on neo walls it still stalls
   or falls (recordings kept in `results/games/neo-preview__bot`).
2. **Which double neos are possible:** a tick-by-tick search, rerun after the
   fix above. Single neos are proven possible (13 layouts).
3. **Learning-progress recordings:** a fresh run that saves snapshots, played
   on the same courses at every stage (untrained to trained), offline and in
   real Minecraft, for the presentation (`results/progress.html`).
4. The last two hyperparameter runs; the rest of that study resumes after.

## For the presentation

- Pages (open in a browser): `results/replay_viewer.html` (real-game
  replays), `results/progress.html` (learning progress, coming),
  `results/neo_preview.html` (neo training).
- Figures: `results/figures/`. Data: `results/DATA.md`.
- Watch it live: `run.cmd live --checkpoint results\final_policy.pt --out
  runs\demo --viewer YourMinecraftName`, then join `127.0.0.1:25570` in
  Minecraft Java 1.21.4.

## Open problems anyone can pick up

1. More block types: slabs, ladders, ice and fences need physics ports
   (prismarine-physics has them) and new course pieces.
2. Hand-built maps: extract the platform chain from a region of a real map.
3. The report: `docs/report_draft.md` needs the hyperparameter section, the
   neo results, the discussion and the conclusion.

## Update log

- **28 Oct 2026, 00:10:** all training data, evaluation logs and game
  recordings (real and offline, every tick) archived in `results/`; training
  curve figures; GitHub push working. Neo training running.
- **27 Oct 2026, 23:45:** EULA accepted; first real-Minecraft runs: 59/60,
  58/60 (obstacles) and 3/3, identical to the simulator. Neo, double-neo,
  momentum and wide jumps added.
- **27 Oct 2026, 23:00:** ablation (15 runs), obstacle fine-tune and no-route
  runs finished; the obstacle-trained policy is the final policy.
- **27 Oct 2026, evening:** main run finished (95.5% held-out); PPO
  stabilised after early collapses.
- **27 Oct 2026, afternoon:** new direction (parkour). Simulator, physics
  parity, feasibility study, environment, PPO, JS bot and offline check built.
