# Natural-resource runs on the PC, 23 October 2026

**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION.** These
are genuine Minecraft Java 1.21.4 Survival episodes in freshly generated
worlds on the desktop PC: no commands, no supplied items, no teleports, no
creative mode. Success means the bot chopped natural wood, crafted planks,
sticks and a crafting table, placed the table, and crafted and held a wooden
pickaxe. It is the first milestone of the project, not an Ender Dragon result.
Every number below is computed by `run.cmd resource-report` from the retained
files in this folder; `retention.json` in each revision folder records a
SHA-256 digest of every retained file (`python -m infrastructure.retention
--verify` checks them).

Two code revisions were run. The second exists because the first exposed two
failure modes of a frozen greedy policy; both revisions are retained so the
failures stay visible.

## Revision `e88cb7c` (before the CHOP, foliage-escape and exploration fixes)

Folder `e88cb7c/`. All episodes used `--max-steps 60` except the single
seed-1 reproduction (`--max-steps 50`). Development seeds 1 to 3; validation
seeds 10000 to 10009, declared before they were run and never tuned on.

| Group | Episodes | Successes | Wilson 95% | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- |
| Q-learning online training, seed 1 reproduction | 1 | 1 | 0.21–1.00 | 11 / 11 | 16.5 / 16.5 | 0 |
| Q-learning online training, seeds 1–3 | 3 | 3 | 0.44–1.00 | 11.7 / 11.7 | 17.1 / 17.1 | 3 |
| SARSA online training, seeds 1–3 | 3 | 3 | 0.44–1.00 | 11.0 / 11.0 | 17.0 / 17.0 | 0 |
| Masked random baseline, seeds 1–3 | 3 | 3 | 0.44–1.00 | 13.3 / 13.3 | 24.6 / 24.6 | 13 |
| Frozen Q-learning policy, seeds 1–3 | 3 | 2 | 0.21–0.94 | 29.0 / 13.5 | 34.5 / 32.5 | 4 |
| Frozen Q-learning policy, validation 10000–10009 | 10 | 10 | 0.72–1.00 | 18.2 / 18.2 | 55.5 / 55.5 | 5 |
| Frozen SARSA policy, validation 10000–10009 | 10 | 8 | 0.49–0.94 | 23.9 / 14.9 | 62.8 / 29.9 | 7 |
| Masked random baseline, validation 10000–10009 | 10 | 8 | 0.49–0.94 | 28.1 / 20.1 | 93.5 / 32.5 | 7 |

No deaths and no invalid episodes occurred. Read the intervals as what they
are: with three or ten episodes per group they are wide, and the learned
policies and the masked random baseline overlap. The feasibility mask and the
scripted helpers carry most of this task; the measurable difference so far is
decision efficiency (a random policy wastes decisions on redundant crafting
and chopping), not capability. Training returns are exploratory and shaped and
must not be compared with frozen evaluation.

**Canopy descent verified live.** Sixteen of the 43 episodes started on a
tree canopy, and sixteen descents cleared 39 leaf layers for 10 health points
in total. In every recorded fall the planned distance equalled the actual
distance and the expected damage equalled the health lost (the longest
descent, ten layers over two decisions, cost two points; the 4- and 5-block
drops cost one or two as predicted). Four times the only drop was six blocks;
the helper refused, and the bot walked along the canopy to a trunk or, in one
case, completed the whole chain standing on the canopy. Two natural oak sapling
drops were attributed in `items_picked_up`.

**Two frozen-policy failure modes.**

1. *CHOP ping-pong* (`pc-q-frozen-dev/episode-0003`, seed 3): `nearby_log`
   was true, CHOP walked towards the trunk, and from the closer spot the bottom
   log was hidden behind the log above it, so CHOP returned `log_unreachable`.
   The compact state did not change, the greedy policy chose CHOP again, and
   the bot alternated between two spots for all 60 decisions. During training
   epsilon-greedy exploration hides this; a frozen policy cannot escape it.
2. *Walled in by foliage* (`pc-sarsa-frozen-val/episode-0007`, seed 10006):
   a clean two-layer descent ended in a one-block shaft with leaves at feet
   and head height on all four sides. The pathfinder never digs, `SEARCH`
   returned `exploration_blocked` in a quarter second, and the policy repeated
   it 59 times.

The other failures were exploration budget exhaustion on treeless spawns
(seeds 10004 and 10009): the bot kept walking without finding a tree within 60
decisions. Reading the exploration code against those trails showed that long
spawn-relative moves only ever used four of the eight headings.

## Revision after the fixes

*To be filled in from the rerun on the committed fixed revision.*

## What these runs do not show

- No stone tools, food, shelter, Nether, stronghold or dragon work exists.
- Three development seeds and ten validation seeds are a small sample; the
  intervals say so.
- Spawn position inside a world is not deterministic, so "seed 1" is terrain,
  not a replay; canopy starts are a matter of sampling.
- The compact `resource_v1` state cannot tell a repeated identical failure
  from a first attempt. Making the helpers robust (done here) narrows the
  problem; a state feature for "last action failed" would be a design change
  for the team to decide.
