# Team status

A one-page briefing for everyone on the team: what works, what is running, how
to try things, and what is still open. It is updated and pushed at the end of
every result. For the full technical history read [HANDOFF_PC.md](HANDOFF_PC.md);
for every measured number read
[results/pc_resource_runs/README.md](results/pc_resource_runs/README.md).

**Last updated:** 24 October 2026, 10:30 (branch `codex/pc-canopy-descent`,
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
- **Within the action mask, learning does not beat random play.** The mask
  and the scripted helpers solve the task, so random choices within the mask
  succeed about as often as the trained policies: 19/20 against 16/20 and
  18/20 at `87673b8`, and 10/10 against 9/10 at `5bb7b25`.
- **Without the mask, learning beats random play** (see the next section).
- **Forest spawns are mostly handled.** The bot often spawns on top of trees,
  and a bounded helper clears the leaves under it. It refuses drops it judges
  unsafe, and a bot over such a drop can still be stranded (see the open
  problems).
- **Parallel live runs work: two at a time on the desktop PC.** The
  scheduler (`.\run.cmd resource-parallel PLAN`) gives each run its own port
  and refuses anything that could mix up results. A contention benchmark,
  declared in advance, compared 1, 2 and 4 slots on 20 worlds:
  - 2 slots passed and were 1.57 times faster than 1 slot;
  - 4 slots failed, because each decision took 1.28 times as long, against
    an allowed 1.25.

  See [results/parallel_benchmark](results/parallel_benchmark/README.md).
  Rerun the benchmark on any other machine first.
- Every number is computed by `run.cmd resource-report` from files retained
  under `results/`, each with SHA-256 digests.

## Latest result: learning beats random play without the mask

A pre-registered live experiment ran on commit `87673b8` from 23 October 20:48
to 24 October 04:17. Its plan was fixed in
[configs/resource_masking_experiment.json](configs/resource_masking_experiment.json)
before any run.

- **Conditions.** `masked` is as before. `unmasked` always allows all eight
  actions, and impossible ones just fail. `unmasked_history` is unmasked plus
  a "last action changed nothing" bit in the state.
- **Training.** 20 live episodes per algorithm in the masked condition and 50
  in each unmasked one, with no simulator pretraining.
- **Test.** Frozen policies and the matching random baseline on 20 held-out
  worlds (seeds 20000 to 20019).
- **Rule.** A benefit counts only if the learned policy's 95% interval lies
  entirely above random's.

| Condition | Q-learning | SARSA | Random baseline | Verdict |
| --- | --- | --- | --- | --- |
| masked | 16/20 (0.58–0.92) | 18/20 (0.70–0.97) | 19/20 (0.76–0.99) | no benefit shown |
| unmasked | **20/20 (0.84–1.00)** | **14/20 (0.48–0.85)** | 4/20 (0.08–0.42) | **benefit shown for both** |
| unmasked + history bit | **16/20 (0.58–0.92)** | **17/20 (0.64–0.95)** | 4/20 (same run) | **benefit shown for both** |

**How to read it.**
- **With the mask, random play already succeeds 19 times in 20**, so this
  test cannot show any benefit of learning.
- **Without the mask, random play wastes its choices.** 81% of them went on
  actions that cannot work there, and it ran out of decisions on 16 of the 20
  worlds. After 50 live training episodes, the frozen Q-learning policy made
  such a choice 2% of the time and succeeded everywhere.
- **No effect of the history bit was shown.** Q-learning got 16/20 with it
  and 20/20 without; SARSA 17/20 against 14/20. The intervals overlap, which
  at this sample size is not evidence that the bit does nothing.
- **Each condition was trained only once.** That shows the task is learnable
  here, not how reliably every training run learns it.

**Why the trained policies still failed** (19 failures in 120 test episodes):
- **10 were canopy starts.** The bot spawned on a tree above a drop the
  helper refused.
- **7 were loops after running short of wood.** The policy had usually
  collected only one or two logs and lacked a plank for the pickaxe. In most
  of these loops, actions it had never tried in that state still had Q = 0.
  Useful actions it had tried there, like `SEARCH` and `CHOP`, sat slightly
  below that because of the step cost. So the frozen policy kept picking at
  random among useless actions.
- **1 ran out of decisions** in states training never reached.
- **1 found no wood.** It spawned below the surface.

**One surprise: a world seed does not fix the spawn point.** No test world gave
all eight arms the same start, and on 13 of the 20 worlds some arms spawned on
a canopy while others spawned on the ground. Seeds pair the terrain, not the
starting spot.

There were 400 live episodes in all. One was invalid: Windows refused to
replace a policy file, probably because another program had it open for a
moment, and the pre-registered continuation took over. One death occurred,
during training. Everything is retained in
[results/pc_resource_runs/87673b8](results/pc_resource_runs/87673b8), and the
[results README](results/pc_resource_runs/README.md) has every table and
failure.

## Planned next

- **Initial-Q experiment, ready to launch (not started).** It runs overnight
  on the desktop PC at 2 slots, for an estimated 10 to 12 hours, so start it
  in the early evening:

  ```powershell
  .\run.cmd resource-parallel configs\initial_q_plan.json
  ```

  - **What it compares.** Unmasked Q-learning and SARSA trained from an
    initial Q of 0 against -1, three agents each, 50 live episodes per agent.
    Every frozen policy and random play are then tested on new held-out
    worlds 40000-40019.
  - **Rules.** The plan, the comparison rule and the analysis script are
    fixed in advance in
    [configs/initial_q_experiment.json](configs/initial_q_experiment.json).
  - **What we expect.** The simulator showed that -1 is the most pessimistic
    value that does not harm learning. Theory predicts it will *not* fix the
    late-stage loops, because with shaping a start of 0 is optimistic by the
    state's potential (about 1.9 there). Either way, three agents per arm
    measure how reliably learning succeeds.
  - **Keep the PC free while it runs.**
- **Stone tools: the design is ready for review.**
  [docs/stone_tools_design.md](docs/stone_tools_design.md) has the full
  milestone: five new macro actions (including a safe dig-down staircase), a
  48-state learner after the frozen wood policy, the simulator study, the
  tests, a pre-registered experiment on new seeds 61000-64119, and a schedule
  to the 19 November assessment. Its section 11 lists the decisions the team
  needs to make. All stone code goes in a separate worktree, so tonight's run
  is not disturbed.
- **Use parallel live runs for the next experiment.** On the desktop PC, use
  `.\run.cmd resource-parallel PLAN` with `"slots": 2` (see the next
  section). Run every arm of one experiment in one schedule with the same
  slot count, because task time is a measured outcome.
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

1. **Stone tools, the next milestone.** The design is in
   [docs/stone_tools_design.md](docs/stone_tools_design.md) and was accepted
   on 24 October; section 9 splits it into ordered steps. Step S1 is taken
   (branch `codex/stone-tools`); later steps build on it.
2. **Sparse rewards.** The experiment uses potential-based shaping. A
   separate condition with only the step cost and the final +10/-10 would be
   the purest learning test; it needs more training episodes.
3. **Deep canopies, now the biggest failure.** A bot on a tree more than five
   blocks above the ground can still be stranded. This caused 10 of the 19
   learned-policy test failures at `87673b8`, and it was also recorded on
   validation seeds 10000 and 10003. It needs a safe way down, for example
   breaking a path down along the trunk.
4. **Frozen loops without the mask.** With wood running short, frozen
   policies looped through useless actions. Untried actions keep their initial
   Q of 0, which is above useful actions that cost a step, and the history bit
   did not fix this. Candidate fixes, each to be tested against the current
   result:
   - an initial Q value below the step cost (pessimistic initialisation);
   - more training episodes;
   - a state field saying whether the next craft can be afforded;
   - several training seeds, to see how often the loop happens.
5. **Training more than one agent.** Each condition was trained once (agent
   seed 0). The initial-Q experiment trains three agents per arm for unmasked
   Q-learning and SARSA; other conditions are still single runs.
6. **Report and presentation.** The report needs 8 to 20 pages and the talk
   15 minutes; [docs/report_plan.md](docs/report_plan.md) has the outline. The
   masked negative result, the unmasked positive one, and the failure analysis
   are good material.
7. **Presentation footage.** Record a `--viewer` run once the PC is free.
8. **A recipe graph and replanning after the wooden pickaxe.** Build the item
   dependency graph from `minecraft-data` (recipes, and which tool mines which
   block). Add a small planning layer that picks the next sub-goal, so the
   tabular learner works within one stage at a time. Plan4MC does this with a
   skill graph that an LLM generated, which needed 6 corrections out of 55
   skills; `minecraft-data` avoids that step. See the related-work notes in
   [docs/report_plan.md](docs/report_plan.md).
9. **A count-based exploration bonus instead of the scripted SEARCH.**
   Plan4MC learns exploration with a state-count reward (+1 per unvisited grid
   cell); without that skill its Mine-Stones success drops from 0.293 to
   0.097. A tabular version: learn the SEARCH heading with a count-based bonus
   over visited cells, and compare it with the scripted exploration rings on
   the same seeds. Its success rates are not comparable with ours, since the
   setups differ completely.

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

- **24 Oct 2026, afternoon:** the stone-tools design is accepted (all 12
  recommendations in its section 11). Offline step S1 has started on branch
  `codex/stone-tools`, in its own worktree, so this checkout's run code stays
  frozen for the initial-Q run. Please do not start S1 pieces in parallel.
  Someone still has to ask the instructor about the scripted dig-down helper
  and the hidden-block safety checks (design section 11, item 1).
- **24 Oct 2026, 10:30:** the initial-Q option (`resource-train --initial-q`)
  and a pre-registered overnight experiment (0 against -1, three agents per
  arm, 2 slots) are committed and reviewed, ready to launch. The stone-tools
  design is in `docs/stone_tools_design.md`.
- **24 Oct 2026, 05:40:** parallel live runs merged into
  `codex/pc-canopy-descent` (151 Python, 5 bridge and 31 Node tests pass) and
  verified live: 2 slots adopted on the desktop PC, 4 slots rejected by the
  declared rule. Plan4MC added to the related work and two open problems.
- **24 Oct 2026, 04:35:** the masking experiment ended at 04:17, and all 15
  runs are retained under `results/pc_resource_runs/87673b8`. Without the
  mask, all four learned policies beat random play on the held-out worlds
  (20/20, 14/20, and 16/20 and 17/20 with the history bit, against 4/20).
  No effect of the history bit was shown. One invalid training episode (a Windows
  file lock) was handled by the declared continuation.
- **24 Oct 2026, 01:55:** masked and unmasked conditions finished. Without
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
