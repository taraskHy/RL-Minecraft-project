# Team status

A one-page briefing for everyone on the team: what works, what is running, how
to try things, and what is still open. It is updated and pushed at the end of
every result. For the full technical history read [HANDOFF_PC.md](HANDOFF_PC.md);
for every measured number read
[results/pc_resource_runs/README.md](results/pc_resource_runs/README.md).

**Last updated:** 24 October 2026, 02:05 (branch `codex/pc-canopy-descent`,
pull request [#1](https://github.com/taraskHy/RL-Minecraft-project/pull/1),
not yet merged into `main`).

## Goal and where we are

The course goal is to start a fresh Minecraft Java 1.21.4 Survival world and
defeat the Ender Dragon, using course-level RL (tabular Q-learning and SARSA,
epsilon-greedy, potential-based reward shaping). The deadline is
31 December 2026.

We have finished the **first milestone only**: gather wood, craft and place a
crafting table, craft a wooden pickaxe. Nothing after the wooden pickaxe exists
yet.

## What is verified (real Minecraft, saved evidence)

- **The wood-to-pickaxe chain works live** in fresh Survival worlds with no
  commands, supplied items, teleports, or creative mode: 43 of 45 episodes
  succeeded at commit `5bb7b25`, with no deaths and no invalid runs.
- **With the action mask, learning does not beat random play.** 55% of
  decisions had only one allowed action, and random choices succeeded as often
  as the trained Q-learning and SARSA policies (10/10 against 9/10 on the
  validation worlds). The mask and the scripted helpers solved the task.
  Without the mask, learning does beat random play; see "Right now" (that
  result is retained when the batch ends).
- **Forest spawns are handled.** The bot often spawns on top of trees; a
  bounded helper clears leaves under it, and every planned fall matched the
  real fall and damage.
- Every number is computed by `run.cmd resource-report` from files retained
  under `results/pc_resource_runs/<commit>/`, each with SHA-256 digests.

## Right now

A pre-registered live experiment is running on commit `87673b8` (started
20:48, expected to finish in the early morning). It asks whether learning
beats random play once the action mask is removed:

- three conditions: `masked` (as before), `unmasked` (all eight actions always
  allowed; impossible ones just fail), and `unmasked_history` (unmasked, plus a
  "last action changed nothing" bit in the state);
- training: 20 live episodes per algorithm masked, 50 unmasked, no simulator
  pretraining;
- test: frozen policies and the matching random baseline on 20 held-out worlds
  (seeds 20000 to 20019) that nothing has been tuned on;
- rule: a benefit counts only if the learned policy's 95% interval lies
  entirely above random's.

The full plan is fixed in
[configs/resource_masking_experiment.json](configs/resource_masking_experiment.json).

**First result: without the mask, learning beats random play.** The masked
and unmasked conditions finished at 01:54 with no invalid episodes. These are
the frozen policies on the 20 held-out test worlds. `run.cmd resource-report`
computed them from the run folders; they are retained under `results/` when
the whole batch ends.

| Condition | Learned policy | Random baseline | Verdict (pre-registered rule) |
| --- | --- | --- | --- |
| masked, Q-learning | 16/20 (0.58–0.92) | 19/20 (0.76–0.99) | no benefit shown |
| masked, SARSA | 18/20 (0.70–0.97) | 19/20 (0.76–0.99) | no benefit shown |
| unmasked, Q-learning | **20/20 (0.84–1.00)** | 4/20 (0.08–0.42) | **benefit shown** (Fisher p = 1.5e-7) |
| unmasked, SARSA | **14/20 (0.48–0.85)** | 4/20 (0.08–0.42) | **benefit shown** (Fisher p = 0.004) |

Read it this way: with the mask, random play already succeeds 19 times in 20,
so nothing is left to learn. Without it, random play wastes 81% of its choices
on actions that cannot work there and mostly runs out of steps. After 50 live
training episodes, the frozen Q-learning policy picked such an action only 2%
of the time, SARSA 41%. That is one trained agent per algorithm; it shows the
task is learnable here, not how reliably every training run learns it.

Training (with exploration, so not test results): masked Q-learning 17 of 20,
masked SARSA 18 of 20 (one death, while exploring seed 118), unmasked
Q-learning 36 of 50 (3, 8, 9, 7, 9 per ten episodes), unmasked SARSA 41 of 50
(6, 8, 9, 9, 9).

The `unmasked_history` condition (50 + 50 training episodes, then its two
tests) is running now and should end around 5 am.

**Please do not run Minecraft on the PC while this batch runs**, and do not
join its world: it uses port 25567 and an extra player in the world can change
the results.

## Planned next

- **Parallel live runs.** Once this batch ends, the resource server gets a
  per-run port so three or four episodes can run at once (the PC used about
  31% of its CPU and had 30 of 63 GB of memory free with one episode running).
  Independent runs (Q-learning and SARSA training, the test worlds) would run
  side by side, cutting a batch like this from about six hours to about two.
  Contention is benchmarked first, and every arm of one experiment runs under
  the same setup, because task time is a measured outcome.
- **GPUs.** The PC has an RTX 2000 Ada (16 GB), and an RTX 5080 machine is
  coming. Today's work does not use a GPU: tabular Q-learning is a table
  lookup, and the Minecraft server and the bot run on the CPU. The GPU becomes
  useful when a later stage needs a neural network (for example DQN, which the
  course allows only when justified) or many fast simulator rollouts.

## Set up on your own machine

1. Install Python 3.12 or newer, clone the repository, and check out
   `codex/pc-canopy-descent` (or `main` once pull request #1 is merged).
2. Double-click `setup.cmd`. It installs portable Java, Node and the pinned
   Minecraft 1.21.4 server inside the project folder.
3. Run `.\run.cmd test` and check that everything passes.
4. To run real Minecraft, record **your own** decision on the Minecraft EULA
   with `.\run.cmd accept-eula`. It is stored only on your machine and never
   in Git. Simulator work does not need it.
5. Watch the bot learn: `.\run.cmd resource-train --viewer YourMinecraftName
   --condition unmasked --episodes 5`, then join `127.0.0.1:25567` in
   Minecraft Java 1.21.4. Your camera follows the bot, and chat shows rewards
   and Q-value updates.

The [Windows setup guide](docs/windows_setup.md) and
[natural-resource training guide](docs/resource_training.md) have every
command and option.

## Open problems anyone can pick up

Pick one, start a branch from `codex/pc-canopy-descent`, and add a line to the
update log below when you start, so we do not duplicate work.

1. **Stone tools, the next milestone.** Design a small state and action set
   for mining stone with the wooden pickaxe and crafting a stone pickaxe,
   reusing the wood chain. See [docs/full_survival_plan.md](docs/full_survival_plan.md).
2. **Sparse rewards.** The experiment uses potential-based shaping. A
   separate condition with only the step cost and the final +10/-10 would be
   the purest learning test; it needs more training episodes.
3. **Deep canopies.** A bot on a tree more than five blocks above the ground
   can still be stranded (recorded on validation seeds 10000 and 10003).
4. **Report and presentation.** The report needs 8 to 20 pages and the talk
   15 minutes; [docs/report_plan.md](docs/report_plan.md) has the outline. The
   honest negative result and the failure analysis are good material.
5. **Presentation footage.** Record a `--viewer` run once the PC is free.

## Rules that keep our results honest

- Never present simulator or development-arena results as Minecraft results.
- Never edit files inside `results/pc_resource_runs/<commit>/`; their digests
  would no longer match. New runs go in `runs/`, which Git ignores.
- No server commands, supplied items, teleports, or creative mode in any
  result we report.
- Decide an experiment's seeds, budget, and success rule before running it, and
  never tune on a held-out seed range.
- Run `.\run.cmd test` before pushing, never force-push, and work on your own
  branch.

## Update log

- **24 Oct 2026, 02:05:** masked and unmasked conditions finished. Without
  the mask both learners beat random play on the held-out worlds (Q-learning
  20/20, SARSA 14/20, random 4/20); with it, no benefit was shown (16/20 and
  18/20 against 19/20). History condition running; retention and write-up
  follow when it ends.
- **23 Oct 2026, 23:57:** all four training runs finished (no invalid
  episodes); test phase started. Frozen masked Q-learning: 16 of 20 test worlds,
  all four failures stranded on tall trees. Not yet compared with random.
- **23 Oct 2026, 22:20:** unmasked Q-learning training at 27 of 50 episodes
  (17 reached the pickaxe). Parallel live runs and a GPU plan added under
  "Planned next".
- **23 Oct 2026, 21:55:** this file created. The live masking experiment is
  running on `87673b8`; the masked training runs are finished.
- **23 Oct 2026:** wood-to-pickaxe milestone verified live (43 of 45 at
  `5bb7b25`); learning shown not to beat masked random play; pull request #1
  opened.
