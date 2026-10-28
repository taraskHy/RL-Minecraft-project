# Team status

One page for everyone on the team: what works, what is running, what is open.
Updated and pushed regularly; the newest results can reach
[results/README.md](results/README.md) first. The main results are in that
README (generated from the run files, neo results included); the rest come
from the files in `runs/`, `results/` and `configs/`. [results/DATA.md](results/DATA.md) lists most of
the data, but has not been updated since the night of 27-28 October; the
generated snapshot at the end of [HANDOFF.md](HANDOFF.md) lists what it is
missing.

**Last updated:** 28 October 2026, afternoon: every training result up to
the PC's commit `a7a426c` (12:34), plus CPU results from a cloud session on
branch `claude/pensive-knuth-biwj11` (not merged into `main` yet).

## The project

**A bot that learns Minecraft parkour and completes courses it has never
seen.** We changed direction to this on 27 October 2026; the survival work
(wooden pickaxe, stone design) is archived on the `archive/survival` tag and
the `codex/*` branches.

How it works, in one paragraph: a Mineflayer bot moves with a physics library
called prismarine-physics. We ported that library to the GPU (it matches the
real one to 1e-14 blocks), generate endless random parkour courses at six
difficulty levels, and train a neural network with PPO on 16,384 parallel
games that share a pool of 8,192 courses (about 170,000 game ticks per second,
network updates included). The trained network then drives
the Mineflayer bot on a real Minecraft server. Details: `docs/design.md`.

## Results

**The final policy changed on 28 October:** the hyperparameter study found
one setting that beats the defaults in all three seeds (discount 0.995
instead of 0.99), so the main run and the obstacle fine-tune were repeated
with it (`main-g995`, `obstacles-g995`) and the result, chosen on the
validation courses (97.6% against 96.6%), is now `results/final_policy.pt`.
The first final policy is kept as `results/final_policy_v1.pt`.

**Real Minecraft (1.21.4 server), final policy, courses never seen in training:**

| Courses | Completed | Same as the simulator? | First final policy |
| --- | --- | --- | --- |
| standard, 10 jumps, all levels | **59 / 60** | yes, every course | 59 / 60 |
| with obstacles (head-hitter beams, decoy pillars) | **57 / 60** | yes | 58 / 60 |
| hand-designed (tour, stairs, sprint, and a neo course) | **3 / 4** (misses the neo course, not in its training) | yes | 3 / 3 (without the neo course) |

The server corrected the bot's position 0 times in 124 episodes; replaying the
recorded moves in the simulator reproduces every real trajectory to 2e-14
blocks.

**Simulator, 1,200 held-out courses per test set:**

- Final policy: **98.4%** of standard courses (hardest level 93.5%) and
  **95.2%** with obstacles (first final policy: 97.1% and 93.6%).
- Training from scratch (main run, 2.6 hours): success goes from 0% (the
  untrained bot falls off the first block; mean reward -5.4) to 95.5% (mean
  reward +53); falls drop from 100% to about 5%. See
  `results/figures/training_main-s0.png` (success, reward, losses, entropy,
  KL and learning rate, every iteration).
- Every possible jump type (levels 0-5) is made from a standing start, all but
  one at all 16 tested angles (that one, a 4-block gap with a sideways
  offset, at 14 of 16).
- New course styles: 15-jump courses 96.5%, any start direction 96.7%, single
  blocks 93.0%, a turn at every jump 94.2%.

**Our reward ideas, tested (3 seeds each, `results/figures/ablation_curves.png`):**

- *Height / fall penalty*: matters most. Without it the hardest level drops to
  54-57% from 69-77%, in every seed.
- *Distance to the goal* (progress reward): learning about 3 times faster;
  same final score.
- *Number of jumps* (landing bonus) and the *curriculum*: no reliable effect
  at this budget.
- Not told where the platforms are (only the blocks it sees and the goal
  direction): 82% instead of 93%.
- *Hyperparameters* (one at a time, chosen on validation): only the
  discount 0.995 beats the defaults by more than 3 points (95.4% against
  91.8% on validation, 3 seeds); it is in the final policy now.

**Problems we found and fixed** (good presentation material):

- PPO collapsed three times in the first runs (success 75-80% to about 15%
  within two updates):
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
- **No double or triple neo was found** in Minecraft's physics as
  Mineflayer runs it: 0 of 64 layouts landed (36 from standing, 28 with a
  3-4 block run-up), searched with the same tool that found the single neos,
  so we treat them as impossible.
- **4-block jumps with a run-up:** the trained bot makes some that no
  scripted strategy makes (12-38% of start angles).
- **Neo training** (`runs/advanced-ft3`, levels 0-8): the bot completes 99%
  of easy courses and 83% overall, but levels 6 and 7 (the levels with neo
  jumps) stayed flat at about 70% and 30% for 850 iterations. Diagnosis
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
  curved run a neo needs.
- **Learning from demonstrations works, partly.** A search in the bot's
  own action space collected 50,706 successful neo move sequences; the bot
  imitated 12,800 of them next to PPO (`neo-bc`), then three DAgger rounds
  let the search correct the bot from the exact states it really reaches the
  neo takeoffs in (`neo-dagger`, `-dagger2`, `-dagger3`). On 210 held-out
  level 6-8 courses that contain a neo:

  | Neo policy | Courses completed | Neo jumps passed / reached |
  | --- | --- | --- |
  | stall penalty only | 0 / 210 | 0 / 202 |
  | + imitation | 49 / 210 | 82 / 227 |
  | + DAgger round 1 | 62 / 210 | 99 / 228 |
  | + DAgger round 2 | 63 / 210 | 112 / 239 |
  | + DAgger round 3 (chosen on validation) | **61 / 210 (29%)** | 97 / 224 |

  Courses with one neo: 56 of 129; with two or more: 5 of 81. On the neo
  course set (the first 10 held-out courses with a neo of each of levels 6,
  7 and 8, plus the hand-designed neo course) it completes **11 / 31** in real Minecraft,
  exactly as offline and in the simulator, 0 server corrections. Video:
  `real_minecraft_neo_level7.mp4`. The neo policy is separate from the
  final policy (`runs/neo-dagger3`, on the training PC), but nearly does
  both jobs: on the same level 0-5 test courses it completes 96.5% (final
  policy 98.4%) and 92.5% with obstacles (95.2%), while the final policy
  completes none of the 210 neo courses.

## No course list: the bot finds its own route

Requested from the side chat: run the trained bot on maps where nobody gives
it the list of platforms. A **greedy route planner** (`parkour/planner.py`,
same code in `bridge/planner.js`) looks at the blocks around the bot after
every landing, finds the platforms, and picks the next three; the existing
final policy jumps to them, no retraining. Only the goal position is given.

| Unseen courses | Final policy, course list given | Final policy + planner | First final policy + planner |
| --- | --- | --- | --- |
| 1,200 standard, simulator | 98.4% | **90.9%** | 89.6% |
| 1,200 with obstacles, simulator | 95.2% | **88.1%** | 86.2% |
| 60 standard, **real Minecraft** | 59/60 | **50/60** | 52/60 |
| 60 with obstacles, **real Minecraft** | 57/60 | **55/60** | 53/60 |
| 3 hand-designed, **real Minecraft** | 3/3 | not run yet | 3/3 |

Real Minecraft matches the simulator on every course, 0 server corrections.
Most failures: the planner takes a shortcut to a later platform that is in
reach, and that jump is harder. **Retuned on 28 Oct** (chosen on the
validation courses): ranking platforms by gap length with a weaker pull to
the goal gives **92.3%** standard and **90.5%** obstacle courses in the
simulator (from 90.9% and 88.1%), and 51/60 and 56/60 with the offline bot;
real Minecraft still to be run on the PC. Videos: `real_minecraft_showcase_no_course_list.mp4`
(first person) and `replay_233_*` / `replay_240_*` (the plan drawn as
numbered cyan platforms). Report section 7.11.

**Learned model that finds its route itself** (`runs/noroute-v2`, 3.2 hours
of training; wide block scan, memory of its last three landings, platform
list and progress hidden): 85.4% of standard and 70.9% of obstacle courses
in the simulator; in real Minecraft 55/60 standard and 37/60 obstacle
courses, matching the simulator, 0 corrections. Better than the earlier
no-route arm (82.4%) but below the planner on every large test set, and
obstacles fool it (10 of its 23 obstacle-course falls end on top of one: 6
on a beam, 4 on a decoy pillar). So: **the planner is the "runs on a new map"
model**; the learned one is the research comparison (report 7.12). Video:
`replay_295_learned_*` (level 5, a course the planner failed).

## Last known state

As of commit `a7a426c` (28 Oct, 12:34) the neo demonstrations, the three
DAgger rounds, the hyperparameter study and the new final policy are all
finished and pushed. Whether anything new is running can only be seen on
the training PC.

- Only two training runs fit on the GPU at once (a third spills out of GPU
  memory and makes all of them 8 times slower).
- **GPU temperature watchdog** (user rule, 28 Oct 07:47): above 65 C every
  GPU job is suspended (no state lost) and resumed below 58 C; each pause
  is logged in `runs/gpu_watchdog.log`.

## For the presentation

- **Videos** (`results/videos/`, 1080p MP4):
  - `ghost_race_00_tour.mp4` and `ghost_race_01..06`: ten training stages
    of the same agent (untrained to 400 updates) run one unseen course at
    once, seen from above. The untrained bot falls at once; later stages get
    further; the trained ones reach the goal. Level 5 is hard: no stage
    gets past the first jump (nine fall there; the 10-update stage jumps in
    place on the start platform until the 10-second stall limit).
  - `real_minecraft_showcase.mp4` (35 s): the bot's own first-person view
    in **real Minecraft 1.21.4** on three hand-designed courses it never saw
    (`real_minecraft_tour/stairs/sprint.mp4` separately). All completed, 0
    server corrections.
  - `real_minecraft_neo_level7.mp4` and `replay_137_neo_policy_*`: the neo
    policy making a neo in real Minecraft (first person) and as a replay.
- Pages (open in a browser): `results/theatre.html` (Minecraft-style 3D
  replays and the ghost races, Overview camera), `results/replay_viewer.html`
  (real-game replays), `results/progress.html` (learning progress),
  `results/neo_preview.html` (neo attempts).
- Figures: `results/figures/`. Data: `results/DATA.md`;
  `python -m parkour.report` copies every run's training and evaluation logs to
  `results/runs/` (now in Git too: before 28 October 01:50 a `.gitignore`
  rule kept them out of GitHub). The last copy in Git is from 11:40
  (commit `6170803`).
- Make new videos: `scripts/render_videos.py` (ghost races) and
  `scripts/film_pov.py` (real Minecraft, first person).
- Watch it live: `run.cmd live --checkpoint results\final_policy.pt --out
  runs\demo --viewer YourMinecraftName`, then join `127.0.0.1:25570` in
  Minecraft Java 1.21.4.

## Open problems anyone can pick up

The full list, with what not to change, is in [HANDOFF.md](HANDOFF.md)
(sections 3 to 5).

1. Neos: 29% of neo courses, and courses with two or more neos almost never.
2. One policy for both normal and neo courses (now there are two).
3. More block types: slabs, ladders, ice and fences need physics ports
   (prismarine-physics has them) and new course pieces.
4. Hand-built maps: extract the platform chain from a region of a real map.
5. The report: `docs/report_draft.md` is complete in draft (results with
   the new final policy and the neo results, discussion, conclusion,
   figures). In the submission format (`results/report_draft.docx` and
   `.pdf`, Arial 12, 1.5 spacing) it is 20 pages before the references,
   exactly the limit. It needs a read-through by the team.

## Update log

- **28 Oct 2026, afternoon (cloud session):** this page brought up to date:
  final policy with discount 0.995 (98.4% / 95.2% in the simulator; 59/60,
  57/60, 3/4 in real Minecraft), neos learned from demonstrations and DAgger
  (0% to 29% of neo courses; 11/31 in real Minecraft), hyperparameter study
  finished. Then, on the CPU: the route planner retuned (92.3% / 90.5%
  without the course list, from 90.9% / 88.1%), the neo policy shown to be
  only 2-3 points behind the final policy on ordinary courses, a failure
  analysis by jump type, ice platforms ready to train (physics checked
  against the real library; the current policy already makes 95.6% of
  all-ice test courses without having seen ice), and the report
  completed (20 pages in Word/PDF). `HANDOFF.md` and `AGENTS.md` tell any AI
  model how to continue.
- **28 Oct 2026, 12:34:** neo evidence files and the audit of 829 claims
  (60 errors corrected).
- **28 Oct 2026, 11:40:** new final policy (discount 0.995); neo imitation
  and three DAgger rounds; neo course set in real Minecraft; neo video.

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
