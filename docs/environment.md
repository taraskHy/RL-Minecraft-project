# First milestone environment

This is an educational model of resource gathering and crafting. It is **not** a simulation of all Minecraft mechanics. The goal of this stage is to test learning and evaluation code before natural-world integration.

## State and initial conditions

The fully observed simulator state is `(x, z, remaining_log_mask, logs, planks, sticks)`. The map is fixed and known; coordinates range from 0 to 4. Bit 0 of the mask represents the log at `(1,1)` and bit 1 represents `(3,2)`. Inventory and the mask make consumed resources observable. The workbench station is `(4,4)` and lava is `(2,3)`. Reset selects uniformly among the 22 safe, unoccupied cells. Inventory starts empty and both logs are present.

```text
. . . . .
. T . . .
. . . T .
. . L . .
. . . . W
```

There is no health, hunger, lighting, enemy, tool durability, terrain generation, perception, or dragon. The workstation is supplied; learning to craft/place one is an essential later step for a fresh-world run. Changing episode RNG seeds changes starts and slips, not this map. Evaluation is not unseen-world generalization.

## Actions and dynamics

| Index | Action | Effect |
| --- | --- | --- |
| 0–3 | North, south, east, west | Move one cell unless blocked by a boundary or an uncut log |
| 4 | Chop | Collect one adjacent log; when both are adjacent, use the first in the fixed log list |
| 5 | Craft planks | Consume one log and produce four planks |
| 6 | Craft sticks | Consume two planks and produce four sticks |
| 7 | Craft pickaxe | At the workstation, consume three planks and two sticks; finish successfully |

Crafting planks and sticks requires no workbench, matching the 2x2 crafting-grid recipes. Invalid actions do nothing and still cost a step. Resources cannot be collected twice. Converting too many planks into sticks can make success impossible; the agent must learn to allocate inventory. There is no legal-action mask or scripted sequence guiding it.

For movement, with probability `slip_probability` (default 0.05), replace the chosen direction with a uniform random direction, including the original direction. Thus the actual probability of a different direction is 0.0375, not 0.05. The realized direction obeys the same blockage and lava rules. All randomness is seeded.

## Reward and episode boundary

Base task reward is `-0.01` per decision plus `+10` for success or `-10` for entering lava. Thus successful and lethal steps give `9.99` and `-10.01`. Here "sparse" means sparse progress/outcome feedback plus a step cost; it does not mean literally zero reward on every nonterminal step.

The shaped variant adds `gamma * Phi(next) - Phi(current)`, with:

```text
Phi(s) = 1[at least one log harvested] + 1[planks >= 3] + 1[sticks >= 2]
```

This progress function uses only the current observation. Consuming items can lower it. True terminal states have zero potential. This is potential-based shaping, a small optional extension to the course's reward topic; see the explanation and primary paper in [the learning guide](learning_guide.md#7-optional-reward-shaping). The code enforces the same discount for shaping and learning. Shaping does not guarantee faster learning.

Success and lava are genuine terminal transitions. The default 100-step limit is an **external sampling cutoff**, returned as `truncated=True`; it is not part of the underlying task objective. We bootstrap at a cutoff, including the SARSA next action, and preserve next-state potential. A finite-horizon objective that treats the last step as real termination would require different modeling, typically including remaining time in state.

The agent has no learned continuation after successful crafting; the terminal flag carries that information. The six-element tuple alone is not a specification of an absorbing terminal state. Never call `step` again until reset.

## Real-game differences

The Mineflayer adapter measures actual blocks, inventory, and position. Digging includes waiting for item collection; crafting invokes a recipe; moving uses a bounded low-level controller. These are programmed action implementations. The learned table selects the action, not the individual mouse/keyboard timing. The adapter does not model artificial random slips and cannot assume its motion, lava, or timing match the simulator. It uses one discount per high-level decision, not per Minecraft tick.

A frozen simulator policy completed a live 15-decision wooden-pickaxe arena run on 23 October 2026. This demonstrates one successful transfer trajectory; it does not establish a transfer success rate or natural-world performance. Later natural-world tasks need richer state, resource discovery, table placement, health/hunger, varying terrain, and skill failure/recovery. They cannot reuse this table unchanged and be called full Survival.
