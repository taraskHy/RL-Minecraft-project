# Watch learning in natural Minecraft worlds

This stage starts with an empty inventory in a newly generated, normal-difficulty
Survival world. The resource task is to find natural wood, gather it, craft and
place a crafting table, then craft and hold a wooden pickaxe. The bot has no
operator permissions. The full Ender Dragon goal remains unfinished.

**Current verification:** real online Q-value updates and spectator messages
work. The natural-world wood/table/pickaxe sequence has **not yet completed a
successful live run**. Forest tests exposed a bot spawning on a tree canopy with
ground below its permitted drop distance. Fixing canopy escape and verifying
the remaining action chain is the next development task. See
[`HANDOFF_PC.md`](../HANDOFF_PC.md) and the retained diagnostics in
[`results/resource_smoke/`](../results/resource_smoke/README.md).

## Start actual training

From the project terminal, run:

```powershell
.\run.cmd resource-train --viewer TaraskHy --episodes 5
```

Replace `TaraskHy` with your Java username if needed. Alternatively, double-click
`run.cmd` and choose **12**. Use **Minecraft Java 1.21.4**. When the terminal asks
you to join, connect to **127.0.0.1:25567**. This is a separate server from the
crafting arena (25565) and your ordinary Survival world (25566).

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

The laptop runs these tabular methods on the CPU; no GPU is used. Each fresh
world takes time to generate and is retained on disk, so many episodes cost
substantially more time and space than simulator training. Logs and worlds stay
under `runs/`; they are excluded from Git. Do not remove an active run's folder.

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
Pathfinding cannot mine or place blocks to build a route. Difficult terrain,
missing trees, hunger, mobs, or blocked placement can cause failure. This stage
does not yet solve general exploration or survival management.

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
- `episode-0001/transitions.jsonl`: complete before/after observations, action, reward components, TD error, and Q-values before/after.
- `episode-0001/steps.csv`: compact per-decision log.
- `episode-0001/outcome.json`: observed completion, death, or truncation; infrastructure errors are marked invalid rather than fabricated failures.
- `episodes.csv` and `summary.json`: completed episode results, labeled training or frozen evaluation.
- `worlds/episode-0001/`: original server settings, empty operator list, logs, and the saved world.

The update equations and state/mask are in `minecraft_rl/resources.py`; game
actions are in `minecraft_bridge/resource-bridge.js`. The training loop is in
`infrastructure/resource_session.py`. These three files are a useful starting
point for dividing the team's study and review work.
