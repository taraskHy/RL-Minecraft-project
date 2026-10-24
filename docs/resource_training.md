# Watch learning in natural Minecraft worlds

This stage starts with an empty inventory in a newly generated, normal-difficulty
Survival world. The resource task is to find natural wood, gather it, craft and
place a crafting table, then craft and hold a wooden pickaxe. The bot has no
operator permissions. The full Ender Dragon goal remains unfinished.

**Current verification:** real online Q-value updates and spectator messages
work. The natural-world wood/table/pickaxe sequence **completed live for the
first time on 23 October 2026 on the PC**: at commit `e88cb7c`, 38 of 43
headless episodes in fresh worlds ended with a held wooden pickaxe (Q-learning
and SARSA training on seeds 1 to 3, a masked random baseline, and frozen
evaluation on seeds 1 to 3 and on the declared validation seeds 10000 to
10009), with no deaths and no invalid episodes. The canopy-descent helper ran
in 16 of those episodes, and every planned fall matched the actual fall and
damage. The five failures exposed two frozen-policy loops (a CHOP approach
that hid the target log, and a landing walled in by foliage) and an
exploration coverage gap. After those and further boundary, visibility and
placement fixes, the whole batch was rerun at commit `5bb7b25`: 43 of 45
episodes succeeded with no deaths and no invalid episodes, and the masked
random baseline did as well as the learned policies (10/10 against 9/10 on the
validation seeds), so learning has not yet been shown to add a benefit within
the mask. Read [`results/pc_resource_runs/README.md`](../results/pc_resource_runs/README.md)
for the numbers and their limits, [`HANDOFF_PC.md`](../HANDOFF_PC.md) for the
session record, and the older laptop diagnostics in
[`results/resource_smoke/`](../results/resource_smoke/README.md).

## Start actual training

From the project terminal, run:

```powershell
.\run.cmd resource-train --viewer TaraskHy --episodes 5
```

Replace `TaraskHy` with your Java username if needed. Alternatively, double-click
`run.cmd` and choose **12**. Use **Minecraft Java 1.21.4**. When the terminal asks
you to join, connect to **127.0.0.1:25567** (or the `--port` you chose). This is a
separate server from the crafting arena (25565) and your ordinary Survival world (25566).

You become a spectator and the camera follows `RLResourceBot`. The action bar
shows the current action and exploration probability. Minecraft chat and the
terminal show each observed reward and Q-value update, for example:

```text
E1 step 4: CRAFT_PLANKS | r=+0.200 | Q +0.000->+0.040; TD +0.200 | crafted_oak_planks
```

That line is an illustrative format, not a recorded measurement. `Q` estimates
the discounted return for taking an action in the current state. `TD` is the
difference between the update target and the old estimate. `epsilon=0.60` means
a 60% chance of selecting a random **allowed** action. Ties are also random.

By default the Q-table starts empty. Every valid game transition updates it;
the command does not replay the old arena policy. It saves the policy after
each update. Five episodes are a short learning demonstration, not enough to
establish convergence or a reliable Minecraft success rate.

At the end of each episode, inspect the bot, then press **Enter in the terminal**.
The world is saved and the next fresh world starts. Reconnect to the same address
when prompted. This deliberate reconnect ensures every episode starts with a
new world and empty inventory, while the learned Q-table carries forward.
**Ctrl+C** stops the run and saves/stops the managed server; previous completed
updates and worlds remain available. The viewer has five minutes to join.

## Resume, compare, and evaluate

The terminal prints a results folder under `runs/`. Use its `policy.json` to
resume learning, for example (replace the example path with your actual folder):

```powershell
.\run.cmd resource-train --viewer TaraskHy --policy runs/your-training-run/policy.json --episodes 5 --world-seed 100
```

To watch that policy **without learning**, use different world seeds:

```powershell
.\run.cmd resource-demo --viewer TaraskHy --policy runs/your-training-run/policy.json --world-seed 10000
```

The display says `EVAL` and `frozen`; no Q-values change. Keep development,
training, validation, and final evaluation seeds separate. Each episode adds one
to `--world-seed`. Minecraft timing and its choice of spawn position can vary even
with the same terrain seed; this is not a deterministic replay guarantee.

To measure what the feasibility mask and scripted helpers achieve **without any
learning**, run the masked random baseline on the same seeds. It picks uniformly
among the allowed actions, uses exactly the same helpers, and never updates a
table:

```powershell
.\run.cmd resource-demo --random-baseline --episodes 5 --world-seed 10000
```

To summarize one or more retained run folders for the report:

```powershell
.\run.cmd resource-report runs/your-training-run runs/your-baseline-run --out results/your-summary
```

It writes `summary.json`, `episodes.csv`, and `report.md` with the success rate
and a Wilson 95% interval, decisions, task seconds, deaths, invalid episodes,
and failure causes, plus the canopy layers and wall leaf blocks cleared with
the recorded fall damage. Online training, frozen evaluation, and the random
baseline are always reported as separate groups; frozen evaluations of
different checkpoints, and runs with different step limits, are separate
groups too.
Failure causes count completed episodes only; invalid, interrupted, or
unfinished episodes appear on a separate "not measured" line. The command
refuses folders that are not natural-resource runs (arena, simulator, or watch
output), a folder passed twice, two folders with the same name, and runs
without any episode.

Other options:

- `--algorithm sarsa`: use SARSA instead of Q-learning for a new policy.
- `--epsilon 0.6 --epsilon-end 0.1`: linear exploration schedule over the requested episodes.
- `--max-steps 80`: truncate an episode after this many macro decisions.
- `--pause 1`: add one second between decisions for easier observation.
- Omit `--viewer` for unattended training without a graphical client.
- `--pretrain 2000`: optionally train first in the simple resource simulator.
  Those episodes are explicitly labeled **SIMULATOR**, saved separately, and
  never counted as Minecraft experience. Default is **zero** pretraining.
- `--out runs/my-new-run`: choose an empty output folder. Existing results are refused.
- `--condition masked|unmasked|unmasked_history`: the experiment condition
  (see the next section). New agents and baselines default to `masked`; a
  `--policy` carries its own condition and a different one is refused.

## Experiment conditions: is anything learned?

In the masked runs most decisions were made by the mask, not the policy: in
the final batch at `5bb7b25`, 370 of 672 decisions (55%) had a single allowed
action and 244 (36%) had two, and the masked random baseline did as well as
the learned policies. Three named conditions separate what the policy has to
decide. They change nothing else: the same helpers, rewards, discount, and
60-decision limit apply.

| Condition | Allowed actions | Learning state | Random baseline |
| --- | --- | --- | --- |
| `masked` | the disclosed mask `recipe_and_budget_v1` | `resource_v1` | masked random: uniform among the allowed actions |
| `unmasked` | all eight, always | `resource_v1` | uniform random over all eight |
| `unmasked_history` | all eight, always | `resource_v2` | the `unmasked` uniform random run (a random policy does not read the state) |

Without the mask, every action is sent to the bridge. An infeasible one (for
example `CRAFT_PICKAXE` with no table in reach) returns its refusal event such
as `missing_materials` or `not_at_own_table`, takes its real time, earns the
`-0.01` step penalty plus the potential shaping term `(gamma - 1) * Phi(s)`, and
changes nothing; it is an ordinary observed transition the learner updates on.
Actions the mask only called redundant (more wood than needed, a second table)
really happen and cost materials and time. Every transition records both the
condition's allowed actions and `masked_actions`, what the mask would have
allowed, so the report can say how often a policy chose outside it.

`resource_v2` appends one bit to the nine `resource_v1` fields:
`last_action_failed` is 1 when the previous decision left all nine fields
unchanged (an infeasible action, a refused helper, or a search that found
nothing new) and 0 at the start of an episode and after any change.
`resource_v1` itself is unchanged, and its checkpoints still load. The trade-off:
the bit gives the policy one step of memory. It separates a state reached by a
change from the same state after a decision that changed nothing, so a policy
can learn not to repeat a failed action in place, which matters most without
the mask, where infeasible actions are common. It cannot tell apart distinct
states in a cycle: in the frozen Q-learning `SEARCH`/`APPROACH_TABLE` loop at
`5bb7b25` every step changed `nearby_table`, and both states had all-zero Q
values because three training episodes never visited them, so that loop was
random tie-breaking and the bit would not have helped. Without the mask it
roughly doubles the number of distinct states visited, so each is visited less
often in the same number of episodes and learning is slower (in the simulator
budget study about 50 episodes instead of 30 to 50). `unmasked` and
`unmasked_history` differ only in the state, and whether the bit helped is
judged only by the declared rule in the experiment file.

The declared experiment, fixed before any live run, is in
[`configs/resource_masking_experiment.json`](../configs/resource_masking_experiment.json):
20 live training episodes per algorithm masked and 50 for each unmasked
condition (world seeds from 100), no simulator pretraining, and evaluation of
frozen Q-learning, frozen SARSA, and the matching random baseline on the
held-out test seeds 20000 to 20019 with `--max-steps 60`. The budgets come
from [`results/resource_budget_study`](../results/resource_budget_study/report.md),
which is simulator evidence only. A learned policy beats its baseline only if
its Wilson interval lies entirely above the baseline's; `resource-report`
prints that comparison for every frozen learned policy, together with the
share of single-choice decisions, choices outside the mask, and no-progress
decisions per group. With 20 test episodes per arm the rule has little power
when the baseline is strong: against a baseline of 14 or more successes no
benefit can be shown, so the masked comparison is expected to read "no benefit
shown" whatever the learned policies do, while against a 5% baseline a learned
policy at 50% is detected about 72% of the time. "No benefit shown" means not
detectable at this sample size. If an invalid episode stops a run, the rest of
its seeds are played by a continuation run that never replays the lost seed;
comparisons count the seeds both arms completed, list the lost ones, and add a
conservative check that counts every lost episode against the learned policy.
Produce the test report from this experiment's test and training runs only,
because evaluations of the same checkpoint on other seeds would be pooled.

**Result at `87673b8` (24 October 2026).**
- **Within the mask, no benefit.** The masked comparison read "no benefit
  shown", as predicted: frozen Q-learning 16/20 and SARSA 18/20, against
  masked random 19/20.
- **Without the mask, every learned policy beat uniform random play** (4/20):
  - unmasked Q-learning 20/20;
  - unmasked SARSA 14/20;
  - history Q-learning 16/20;
  - history SARSA 17/20.
- **No effect of the `last_action_failed` bit was shown** for either
  algorithm; the intervals overlap. Frozen policies with the bit still looped
  through useless actions after running short of wood. In a state they reach
  again unchanged, actions never tried there keep `initial_q` 0, which is
  above the slightly negative `SEARCH` and `CHOP`.

Every table, failure and limit is in
[`results/pc_resource_runs/README.md`](../results/pc_resource_runs/README.md).
Note there that a world seed fixes the terrain but not the spawn point, so
seeds pair terrain, not identical starts.

```powershell
.\run.cmd resource-train --condition unmasked --algorithm sarsa --episodes 50 --world-seed 100 --max-steps 60 --pause 0 --out runs/unmasked-sarsa-train
.\run.cmd resource-demo --policy runs/unmasked-sarsa-train/policy.json --episodes 20 --world-seed 20000 --max-steps 60 --pause 0 --out runs/unmasked-sarsa-test
.\run.cmd resource-demo --random-baseline --condition unmasked --episodes 20 --world-seed 20000 --max-steps 60 --pause 0 --out runs/unmasked-random-test
```

The laptop runs these tabular methods on the CPU; no GPU is used. Each fresh
world takes time to generate and is retained on disk, so many episodes cost
substantially more time and space than simulator training. Logs and worlds stay
under `runs/`; they are excluded from Git. Do not remove an active run's folder.

## Run several live episodes at once

A live episode mostly waits on one Minecraft server and one bot, so one PC can
run several. `run.cmd resource-parallel PLAN` runs the jobs of a plan file, at
most `slots` at a time. Each running job gets its own loopback port, from 25568
upward by default (25567 stays free for a run started by hand), and its own
folder under `runs/<plan name>/`. A job runs exactly the `resource-train` or
`resource-demo` command you would type by hand, with `--port`,
`--parallel-slots`, and `--schedule-id` added.

**Not yet verified live.** This was written while the masking batch held the
PC, and it has not yet run Minecraft. Before any experiment uses it, run the
contention benchmark declared in
[`configs/parallel_contention_benchmark.json`](../configs/parallel_contention_benchmark.json).
The helpers stop on wall-clock limits (SEARCH 36 s, canopy descent 20 s, table
approach 18 s, navigation 12 s), so a busy CPU can change successes and
decision counts, not only task time. The benchmark compares 4 slots (or 2)
against 1 slot on the same 20 worlds and decides, by a rule fixed in advance,
whether parallel runs are allowed.

A plan is a JSON file; [`configs/parallel_example_plan.json`](../configs/parallel_example_plan.json)
is a complete example (not an experiment):

```json
{
  "schema": 1, "name": "example-parallel", "slots": 4,
  "interleave": true, "evaluate_after_all_training": true,
  "defaults": {"agent_seed": 0, "max_steps": 60, "pause": 0},
  "jobs": [
    {"name": "ex-unmasked-q-train", "kind": "train", "condition": "unmasked", "algorithm": "q_learning",
     "episodes": 10, "first_world_seed": 50000, "epsilon": 0.6, "epsilon_end": 0.1},
    {"name": "ex-unmasked-q-test", "kind": "evaluate", "policy_from": "ex-unmasked-q-train",
     "episodes": 8, "first_world_seed": 50100, "chunks": 4},
    {"name": "ex-unmasked-random-test", "kind": "evaluate", "random_baseline": true, "condition": "unmasked",
     "episodes": 8, "first_world_seed": 50100, "chunks": 4}
  ]
}
```

- A `train` job is one run that learns, in order; it cannot be split.
- An `evaluate` job is frozen and names exactly one policy source:
  `policy_from` (a training job listed earlier, whose final policy it tests),
  `policy` (a file), or `random_baseline: true` with a `condition`. `chunks`
  splits its worlds into consecutive seed ranges that run side by side, and
  `resource-report` pools the chunks back into one group. Chunk *k* starts
  from agent seed `agent_seed + k - 1`, so no two chunks start from the same
  random stream. Within a chunk the draws for later worlds depend on how many
  decisions the earlier worlds took, and a continuation restarts its chunk's
  seed, as the declared `--seed` rule says.
- `interleave: true` starts the first chunk of every job, then every second
  chunk, and so on, so the arms share the load instead of running in blocks.
- `evaluate_after_all_training: true` holds every evaluation until all training
  has ended, so all test arms run under the same load. A plan with evaluations
  and more than one slot must set both options. Run every arm of one
  experiment in one schedule: the report groups and compares parallel runs
  only within one schedule.
- `defaults` sets `agent_seed`, `max_steps`, and `pause` for every job. They
  have no hidden defaults: each job must end up with all three.
- Names: every run name must be distinct (ignoring case), may not end in
  `-partN`, must leave room for a `-partN` continuation suffix within 64
  characters, and may not be the plan's own name, `logs`, `plan.json`,
  `schedule.json`, or `schedule.final.json`. Plans you want to report together
  need distinct run names.

What the scheduler guarantees:

- It refuses uncommitted run code (`minecraft_bridge`, `infrastructure`,
  `minecraft_rl`) and a plan that is not committed in this checkout, and
  records the commit in `schedule.json`. It checks the committed run code
  again before every launch and every minute while runs are going, and every
  episode checks its own sources before its world starts. If the run code
  changed, nothing more starts and the schedule ends as `code_changed`.
  Drafting or committing the next plan under `configs/` is not a code change.
- One live run or schedule runs at a time per user on the machine, from any
  checkout: a run started by hand holds the same machine-wide lock for its
  whole duration, and a schedule refuses to start while any Minecraft port
  (25565 to 25599) is in use, never stopping what holds it. A schedule's own
  runs carry its id, which is accepted only while that schedule holds the
  lock. Update every checkout on the PC, because older code takes no lock.
- It checks each slot's port again before every launch; a port taken
  meanwhile retires that slot for the rest of the schedule.
- It never reuses a folder: every schedule needs a new plan name.
- An invalid episode is continued as declared for the masking experiment: in
  `<run>-partN` from the next world seed (at most `max_continuations`,
  default 5), never replayed. A training continuation resumes from the last
  saved policy at the declared exploration rate for its next episode. If the
  invalid episode was the last one planned, the run is finished: the episode
  counts as attempted, and its policy is the one that is tested. A run that
  fails without attempting an episode is not retried, and no evaluation of its
  policy starts. A continuation that cannot start (after a code change or a
  scheduler error) is recorded as `continuation_not_started`.
- **Ctrl+C** starts nothing new and waits while each run stops its own server
  and saves its world; nothing is killed. A run launched at the very moment
  of the Ctrl+C may not receive it; the scheduler names it, and one more
  Ctrl+C reaches it. A second Ctrl+C only lists the runs still stopping and
  keeps waiting; a third leaves, and then the window must stay open until
  those runs exit. After any other scheduler error it also waits for the
  running runs before it stops. The closing record is written whatever is
  pressed while it is written, and a press then marks the schedule
  `interrupted`; git runs in its own process group, so Ctrl+C never ends a
  code check half-way. A run with a viewer cannot be scheduled; watch
  training alone with `resource-train --viewer`.

```powershell
.\run.cmd resource-parallel configs\parallel_benchmark_4slots.json --dry-run
.\run.cmd resource-parallel configs\parallel_benchmark_4slots.json
.\run.cmd resource-report runs\bench-1slot runs\bench-4slots --out results\your-benchmark-report
python -m infrastructure.retention runs\bench-1slot runs\bench-4slots --out results\your-retained-runs --label "..."
```

A schedule folder given to `resource-report` or to retention stands for every
run and continuation its `schedule.json` recorded, so no continuation is left
out. Retention keeps those runs nested under the schedule's folder, so a
retained schedule can be reported the same way, and keeps its `plan.json` and
`schedule.json` (not its logs) with digests. If `schedule.json` was held open
by another program when the schedule ended, the final record is written beside
it as `schedule.final.json`; the report and retention then read that newer
record (the report says so) and retention keeps both files. Every run records its port, `parallel_slots`, and schedule in
`metadata.json`, and every episode its UTC start and end. The report keeps
runs with different slot counts, and parallel runs of different schedules, in
separate groups, shows the slot count in its comparison tables, and compares
parallel runs only within one schedule;
runs recorded before this option existed ran one at a time and count as one
slot. `schedule.json` records each run's port, process id, start and end, exit
code, and the most runs going at once while it ran (`max_running`).

Each server may use up to 2 GB of Java heap (`server_memory_mb` in
`.local/machine.json`), so 4 slots need about 8 GB plus the bots. None of this
uses a GPU: the Minecraft server, the bot, and tabular Q-learning and SARSA run
on the CPU. A GPU matters only for a later stage that needs a neural network.

## What is learned and what is programmed

The policy chooses among `SEARCH`, `CHOP`, `CRAFT_PLANKS`, `CRAFT_STICKS`,
`CRAFT_TABLE`, `PLACE_TABLE`, `APPROACH_TABLE`, and `CRAFT_PICKAXE`.

The state contains capped log, plank, stick, table, and pickaxe counts, nearby
log/table flags, an own-table flag, and a health bin. This is a **partial
observation**: it does not encode arbitrary terrain, wood species, or every
Minecraft entity. The policy uses a new `resource_v1` checkpoint format; the
old arena's table cannot be loaded as a natural-world policy.

Recipe feasibility and a resource budget restrict which actions can be chosen.
The agent still chooses among alternative gathering/crafting orders, but this
mask supplies considerable task knowledge. A masked random policy can also
complete this small task. Compare decision counts, elapsed time, failures, and
returns against that baseline; success alone does not demonstrate a benefit
from learning. Keep the same mask and execution helpers across comparisons.

The game bridge supplies programmed, bounded walking, digging, item pickup,
recipe execution, placement, and equipment controls. It scans loaded blocks
with a line-of-sight check, searches locally, and stays within 64 horizontal
blocks of its starting point. It does not use the world seed to locate resources.
Pathfinding cannot mine or place blocks to build a route, and its ordinary
descent limit is three blocks.

**Leaf clearing** is the one bounded exception to "no digging outside CHOP",
in two situations: a *canopy descent* beneath the feet and a *foliage escape*
beside the bot. Forest spawns can put the bot on top of tree leaves, several
blocks above the ground, where no ordinary path exists; the retained laptop
runs and the PC runs show exactly this. When `SEARCH` starts with leaves under the bot's feet, the helper scans
the column below. Only if a solid landing lies within a recorded safe fall (no
damage, or at most two health points while health stays above ten, with no
water, lava, powder snow, or other hazard in the gap) does it clear the single
leaf block beneath its feet and drop one layer. It repeats at most six times,
may first step onto a neighbouring leaf block whose column is safer, and
refuses otherwise. The landing height is the block's collision surface, so a
carpet, slab, mud, or path counts as the longer fall it really is, and a big
dripleaf is treated as a hazard. Every cleared block, planned and actual fall,
and health change is saved in that transition's `recovery` field, and the step
event names it, for example `cleared_2_leaf_layers;descended_canopy;log_reached`.
Leaves broken by hand follow the vanilla loot table (about 5% sapling, 2% one
or two sticks, 0.5% apple on oak and dark oak); anything collected during the
descent is listed in the record as `items_picked_up` and counts in the
inventory like any other natural item. Each dig is bounded by its expected
duration and stops on death, so a bot killed mid-descent ends the episode as a
recorded death rather than a transport timeout. It never digs logs, dirt, or
stone, never places blocks, and the policy still only chose `SEARCH`. `CHOP` re-checks up to three logs that are already visible and in
reach from where the bot stands, digs the first that still passes (one log per
decision, lowest first), and only then walks to the dropped item; it never
approaches first, because an approach can hide a trunk's bottom log
behind the log above it, and a frozen policy whose compact state does not
change would repeat the same failing decision until truncation (observed on
the PC before this change). `CHOP` and `SEARCH` never target a block beneath
the bot's own hitbox (only the canopy descent removes the leaf block under the
feet): chopping its support dropped the bot into a two-by-two dark oak trunk on
the PC, where nothing could be placed. The transition's `action_detail` records
the candidates, the target, and whether the dropped item was seen. Visibility
is checked by casting rays to a grid of points on every face of a block that
looks towards the eyes, tested against each block's real collision shapes and
ignoring shapes behind the eyes. In two retained PC failures the library's
single ray to the table's centre did not reach a table one block away (its
range limit was not the cause, and the transitions do not record which block
stopped the ray), which left the placed table "out of reach". `PLACE_TABLE`
takes a level orthogonal neighbour of the bot's cell first and never a cell
that its own hitbox or another creature overlaps, nor the top of a container,
bed, door, trapdoor, workstation or other block that a click would use instead
(the server refuses or consumes such a placement). `APPROACH_TABLE` moves onto
the table's standable neighbour cells (level first, then one below and one
above, orthogonal first, with the pathfinder's rules for carpets and fences)
until a face is visible (`table_reached`), reporting `table_not_visible` or
`table_path_blocked` otherwise. Every navigation target's block coordinates
stay at least three blocks inside the 64-block radius (a target cell's centre
about 2.3 blocks), apart from re-centring on the leaf block the bot already
stands on, and the pathfinder only walks through cells whose centre is at
least 1.5 blocks inside it (a bot pushed past that line may still use its own
cell and step back inward), so a goal tolerance, a walk to a dropped item, or a
navigation cut off mid-path cannot carry the bot across the boundary (one PC
episode was invalidated that way). These geometry rules are
tested offline on real 1.21.4 block shapes, including cases modelled on the PC
failures. Difficult terrain, missing trees, hunger, mobs, or blocked
placement can still cause failure. This stage does not yet solve general
exploration or survival management.

**Foliage escape.** A descent through a canopy's two lowest leaf layers can
leave the bot standing in a one-block shaft. On validation seed 10006
(`pc-sarsa-frozen-val/episode-0007` at `e88cb7c`) the saved world shows oak
leaves at head height on all four sides, leaves at feet height on two sides,
and grass at feet height on the other two; the pathfinder, which never digs,
had no move, `SEARCH` returned `exploration_blocked` in about a quarter
second, and the frozen policy repeated it 59 times. Now, when the bridge's own
movement rules report no legal move from the current cell (no walk, one-block
jump up, or short drop onto a safe landing), `SEARCH` may clear at most two
leaf blocks beside the bot, feet and head height in one direction that has a
solid non-hazard floor at most one block down; the direction needing the
fewest blocks wins. `PLACE_TABLE` uses the same bounded clearing when the bot
is walled in with no placement surface. A trunk, dirt, or stone wall is never
dug, so a bot in a pit stays in the pit. The record is saved as `leaf_escape` (alone, or as the
`escape` field of the descent record that led to it) with the four wall
columns, the cleared blocks, any natural drop collected (`items_picked_up`),
and whether a move existed afterwards; the step event of the action that ran
it (`SEARCH` or `PLACE_TABLE`) includes `cleared_1_leaf_wall_block`,
`cleared_2_leaf_wall_blocks`, or `leaf_escape_refused:<reason>`, unless a
later part of that action fails, in which case the event is
`action_failed: <message>` and only the `leaf_escape` record shows the
clearing. After a `PLACE_TABLE` escape the table is never placed in the exit
just opened. Wall composition is recorded only from this
revision on; the `e88cb7c` transitions hold only the bot's own column and the
adjacent ground depths.

**Exploration order.** Without a visible trunk, `SEARCH` walks to
spawn-relative targets on rings of 8, 16, up to 48 blocks (diagonal targets
are capped so that every target stays inside the 64-block radius), covering
all eight headings of a ring before the radius grows, and tries a short local
move only after a long target could not be reached. Before 23 October 2026 the
long targets used only four of the eight headings; two treeless plains spawns
(validation seeds 10004 and 10009) exhausted sixty decisions under that order.

Only the spectator is moved or given night vision through server commands.
The bot receives no items, teleportation, healing, creative mode, prepared logs,
or supplied crafting table. Fresh worlds provide episode resets.

The base reward is `-0.01` per decision, plus `+10` for an actual pickaxe or
`-10` for death. Potential-based shaping adds `gamma * Phi(next) - Phi(current)`
using observable material/crafting progress. Terminal potential is zero;
time-limit truncation still bootstraps. Variable-duration helpers are discounted
per **decision**, not per second. These are modeling choices to explain in the report.

## Saved evidence

- `metadata.json`: parameters, action order, source hashes, initial policy provenance, and declared helper boundaries.
- `initial_policy.json` and `policy.json`: before/after tables, with no executable serialization.
- `episode-0001/start.json`: actual initial inventory, position, and connection details.
- `episode-0001/transitions.jsonl`: complete before/after observations (including `on_canopy`, `support_block`, adjacent drop depths, visible logs, any `recovery` record, and for `CHOP` an `action_detail` object naming the candidate logs, the dug target, whether the dropped item was seen and whether it was collected), action, reward components, TD error, and Q-values before/after.
- `episode-0001/steps.csv`: compact per-decision log, with a `recovery` note whenever the canopy-descent or foliage-escape helper ran.
- `episode-0001/outcome.json`: observed completion, death, or truncation; infrastructure errors are marked invalid rather than fabricated failures.
- `episodes.csv` and `summary.json`: completed episode results, labeled training or frozen evaluation.
- `worlds/episode-0001/`: original server settings, empty operator list, logs, and the saved world.

The update equations and state/mask are in `minecraft_rl/resources.py`; game
actions are in `minecraft_bridge/resource-bridge.js`. The training loop is in
`infrastructure/resource_session.py`. These three files are a useful starting
point for dividing the team's study and review work.
