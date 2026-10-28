# Team status

One page for everyone on the team: what works, what is running, what is open.
Updated and pushed after every result. Every number is in
[results/README.md](results/README.md) (generated from the run files); every
piece of data is listed in [results/DATA.md](results/DATA.md).

**Last updated:** 28 October 2026, 02:00.

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

## Harder jumps (neo, double neo, 4-block)

- **Single neos are possible** (a tick-by-tick search lands 16 of 18 layouts),
  but very tight: in the search, only 1 in 2,000 to 1 in 16,000 sampled
  action sequences landed one.
- **Double and triple neos are impossible** in Minecraft's physics as
  Mineflayer runs it: 0 of 64 layouts landed (36 from standing, 28 with a
  3-4 block run-up), searched with the same tool that found the single neos.
- **4-block jumps with a run-up:** the trained bot makes some that no
  scripted strategy makes (12-38% of start angles).
- **Neo training** (`runs/advanced-ft3`, levels 0-8): the bot completes 99%
  of easy courses and 83% overall, but levels 6 and 7 (neo courses) stayed
  flat at about 70% and 30% for 850 iterations. Diagnosis
  (`results/runs/advanced-ft3/jump_eval_advanced.json`): the bot makes
  none of the 13 neo jump types. On 11 of them it **does not fall: it
  refuses to jump** and stands still until the 10-second stall limit,
  every time. Standing still cost less reward than a
  likely fall (-5), so refusing was the rational choice under our reward.
  This is a good presentation point about reward design.
- **Stall penalty tried** (`--reward-stall 5`: a stall ends the episode as
  a failure, like a fall; `runs/neo-stall`, 1,000 iterations): still 0 of
  13 neo types; refusals drop from 95% to 89%, the rest are now falls.
  Exploration is the real problem: random tries almost never find the
  curved run a neo needs. Next idea: learn from the search's successful
  move sequences (demonstrations).

## No course list: the bot finds its own route

Requested from the side chat: run the trained bot on maps where nobody gives
it the list of platforms. A **greedy route planner** (`parkour/planner.py`,
same code in `bridge/planner.js`) looks at the blocks around the bot after
every landing, finds the platforms, and picks the next three; the existing
final policy jumps to them, no retraining. Only the goal position is given.

| Unseen courses | Course list given | Planner |
| --- | --- | --- |
| 1,200 standard, simulator | 97.1% | **89.6%** |
| 1,200 with obstacles, simulator | 93.6% | **86.2%** |
| 60 standard, **real Minecraft** | 59/60 | **52/60** |
| 60 with obstacles, **real Minecraft** | 58/60 | **53/60** |
| 3 hand-designed, **real Minecraft** | 3/3 | **3/3** |

Real Minecraft matches the simulator on every course, 0 server corrections.
Most failures: the planner takes a shortcut to a later platform that is in
reach, and that jump is harder. Videos: `real_minecraft_showcase_no_course_list.mp4`
(first person) and `replay_233_*` / `replay_240_*` (the plan drawn as
numbered cyan platforms). Report section 7.11.

**Learned model that finds its route itself** (`runs/noroute-v2`, 3.2 hours
of training; wide block scan, memory of its last three landings, platform
list and progress hidden): 85.4% of standard and 70.9% of obstacle courses
in the simulator; in real Minecraft 55/60 standard and 37/60 obstacle
courses, matching the simulator, 0 corrections. Better than the earlier
no-route arm (82.4%) but below the planner on every large test set, and
decoy pillars fool it (10 of its 23 obstacle-course falls end on top of an
obstacle). So: **the planner is the "runs on a new map" model**; the learned
one is the research comparison (report 7.12). Video:
`replay_295_learned_*` (level 5, a course the planner failed).

## Running now

1. **Neos from demonstrations** (requested from the side chat): a search
   in the bot's own action space collects thousands of successful neo
   move sequences; then training with an imitation loss next to PPO.
2. Hyperparameter study: all 9 settings done; gamma 0.995 led on
   validation (95.2% and 95.8% in two seeds against a defaults mean of
   91.8%); a third seed is finishing.
3. Only two training runs fit on the GPU at once now (a third spills out
   of GPU memory and makes all of them 8 times slower).
4. **GPU temperature watchdog** (user rule, 28 Oct 07:47): above 65 C every
   GPU job is suspended (no state lost) and resumed below 58 C; each pause
   is logged in `runs/gpu_watchdog.log`. At start the GPU was at 42-47 C.

## For the presentation

- **Videos** (`results/videos/`, 1080p MP4):
  - `ghost_race_00_tour.mp4` and `ghost_race_01..06`: ten training stages
    of the same agent (untrained to 400 updates) run one unseen course at
    once, seen from above. The untrained bot falls at once; later stages get
    further; the trained ones reach the goal. Level 5 is hard: every stage
    falls at its first jump.
  - `real_minecraft_showcase.mp4` (35 s): the bot's own first-person view
    in **real Minecraft 1.21.4** on three hand-designed courses it never saw
    (`real_minecraft_tour/stairs/sprint.mp4` separately). All completed, 0
    server corrections.
- Pages (open in a browser): `results/theatre.html` (Minecraft-style 3D
  replays and the ghost races, Overview camera), `results/replay_viewer.html`
  (real-game replays), `results/progress.html` (learning progress),
  `results/neo_preview.html` (neo attempts).
- Figures: `results/figures/`. Data: `results/DATA.md`; every run's training
  and evaluation logs are in `results/runs/` (now in Git too: before 28
  October 01:50 a `.gitignore` rule kept them out of GitHub).
- Make new videos: `scripts/render_videos.py` (ghost races) and
  `scripts/film_pov.py` (real Minecraft, first person).
- Watch it live: `run.cmd live --checkpoint resultsinal_policy.pt --out
  runs\demo --viewer YourMinecraftName`, then join `127.0.0.1:25570` in
  Minecraft Java 1.21.4.

## Open problems anyone can pick up

1. More block types: slabs, ladders, ice and fences need physics ports
   (prismarine-physics has them) and new course pieces.
2. Hand-built maps: extract the platform chain from a region of a real map.
3. The report: `docs/report_draft.md` needs the hyperparameter section, the
   neo results, the discussion and the conclusion.

## Update log

- **28 Oct 2026, 02:00:** ghost-race and real-Minecraft first-person
  videos; double and triple neos shown impossible; neo plateau diagnosed
  (the bot refuses tight neos) and a stall-penalty fix in training; the
  archived run logs pushed to GitHub.

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
