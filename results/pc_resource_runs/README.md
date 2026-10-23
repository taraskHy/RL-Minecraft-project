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

## Revision `b9625f5` (CHOP in place, foliage escape, exploration rings)

Folder `b9625f5/`. Same seeds and step limit; the training runs were repeated
so that every number comes from one code revision. The frozen Q-learning
validation run stopped after its fifth episode was invalidated (see below),
so that condition has four measured validation episodes, not ten.

| Group | Episodes | Successes | Wilson 95% | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- |
| Q-learning online training, seeds 1–3 | 3 | 3 | 0.44–1.00 | 11.0 / 11.0 | 17.9 / 17.9 | 2 |
| SARSA online training, seeds 1–3 | 3 | 3 | 0.44–1.00 | 11.0 / 11.0 | 17.9 / 17.9 | 2 |
| Masked random baseline, seeds 1–3 | 3 | 3 | 0.44–1.00 | 10.7 / 10.7 | 17.4 / 17.4 | 6 |
| Frozen Q-learning policy, seeds 1–3 | 3 | 3 | 0.44–1.00 | 10.3 / 10.3 | 22.6 / 22.6 | 0 |
| Frozen SARSA policy, seeds 1–3 | 3 | 3 | 0.44–1.00 | 11.7 / 11.7 | 19.9 / 19.9 | 6 |
| Frozen Q-learning policy, validation 10000–10004 (one invalid) | 4 measured | 3 | 0.30–0.95 | 28.0 / 17.3 | 45.2 / 55.2 | 3 |
| Frozen SARSA policy, validation 10000–10009 | 10 | 9 | 0.60–0.98 | 22.0 / 17.8 | 54.6 / 58.2 | 3 |
| Masked random baseline, validation 10000–10009 | 10 | 8 | 0.49–0.94 | 25.6 / 17.0 | 40.8 / 44.1 | 15 |

Across the 40 episodes: 35 successes, no deaths, one invalid episode, 16
canopy starts, 19 descents clearing 37 leaf layers for 9 health points with no
plan-versus-outcome mismatch, 15 refusals of too-far drops, and the first two
foliage escapes (four leaf blocks, both opened a legal move). The two frozen
loops fixed by this revision did not recur. The five non-successes were new:

1. *Invalid episode* (`pc2-q-frozen-val/episode-0005`): the bot stood 63.4
   blocks from spawn beside a trunk, and the walk to the dropped log item took
   it past 64, which the bridge treats as leaving the search area. The whole
   run then stopped, so seeds 10005 to 10009 were not played by this policy.
2. *Dug into its own trunk* (`pc2-q-frozen-val/episode-0003`): `CHOP` took the
   log directly beneath the bot's feet three times, lowering it into a shaft
   inside a two-by-two dark oak trunk; no placement surface was visible or
   reachable, and the mask leaves `PLACE_TABLE` as the only action there.
3. *Table "out of reach" one block away* (`pc2-sarsa-frozen-val/episode-0006`,
   `pc2-random-val/episode-0008`): the table was placed diagonally adjacent,
   the library's visibility ray missed it, so `APPROACH_TABLE` had nothing to
   do and failed fifty times.
4. *Canopy over a deep drop* (`pc2-random-val/episode-0001`): a start at
   height 121 with no landing within the twelve-block scan; the helper refused
   sixty times, correctly. This remains an open limitation.

Fixes for 1 to 3 (a three-block target margin inside the radius, never digging
beneath the bot's own hitbox, a face-point visibility check, table approach
and placement changes) have been applied and checked offline; their live rerun
is described in the next section.

## Revision `5bb7b25` (margin, hitbox, visibility, placement and approach fixes)

Folder `5bb7b25/`. The whole batch was rerun on this committed revision: new
training runs, new frozen checkpoints, the masked random baseline, and frozen
evaluation on the same development and declared validation seeds, all with
`--max-steps 60`.

| Group | Episodes | Successes | Wilson 95% | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- |
| Q-learning online training, seeds 1–3 | 3 | 3 | 0.44–1.00 | 10.7 / 10.7 | 19.1 / 19.1 | 4 |
| SARSA online training, seeds 1–3 | 3 | 3 | 0.44–1.00 | 10.3 / 10.3 | 17.9 / 17.9 | 4 |
| Masked random baseline, seeds 1–3 | 3 | 3 | 0.44–1.00 | 11.7 / 11.7 | 19.3 / 19.3 | 2 |
| Frozen Q-learning policy, seeds 1–3 | 3 | 3 | 0.44–1.00 | 10.3 / 10.3 | 16.8 / 16.8 | 2 |
| Frozen SARSA policy, seeds 1–3 | 3 | 3 | 0.44–1.00 | 10.3 / 10.3 | 19.0 / 19.0 | 2 |
| Frozen Q-learning policy, validation 10000–10009 | 10 | 9 | 0.60–0.98 | 17.6 / 12.9 | 61.8 / 28.2 | 13 |
| Frozen SARSA policy, validation 10000–10009 | 10 | 9 | 0.60–0.98 | 18.8 / 14.2 | 46.7 / 41.7 | 14 |
| Masked random baseline, validation 10000–10009 | 10 | 10 | 0.72–1.00 | 14.8 / 14.8 | 43.2 / 43.2 | 1 |

Across the 45 episodes: 43 successes, no deaths, no invalid episodes, and no
server movement warnings. Fourteen episodes started on a canopy; 23 descents
cleared 42 leaf layers for 3 health points in total, again with every planned
fall equal to the actual fall and every expected damage equal to the health
lost. The descent refused a too-far drop 43 times in four episodes, 34 of
them in the stranded acacia episode described below. One foliage escape
cleared a single leaf block and opened a legal move; the ten refused escapes
all came from that same acacia episode. No `table_not_visible`, no
`action_failed`, and none of the loops seen at the earlier revisions recurred.

**What the comparison shows.** On the validation seeds the masked random
baseline succeeded 10 times out of 10 and each learned policy 9 out of 10; the
intervals overlap almost completely. The learned policies used slightly fewer
decisions in their successful episodes (12.9 and 14.2 against 14.8), which is
not a measured benefit at this sample size. The honest reading is that the
feasibility mask and the scripted helpers solve this milestone; the Q-learning
and SARSA agents learn a valid ordering of the remaining decisions but have not
yet been shown to beat chance within that mask.

**The two failures.**

1. *A learned-policy oscillation* (`pc3-q-frozen-val/episode-0010`, seed
   10009, snowy plains). With its table placed, the policy still needed one
   more log. Beside the table the mask allows only `SEARCH`; away from it the
   frozen policy preferred `APPROACH_TABLE`, so each search was undone by a walk
   back (16 times) until truncation. The masked random baseline and the frozen
   SARSA policy both solved this seed, so this is a weakness of that Q table in
   a treeless biome, not of the helpers.
2. *A high acacia canopy* (`pc3-sarsa-frozen-val/episode-0004`, seed 10003).
   The bot started on acacia leaves 8 blocks above the ground. The descent
   correctly refused that fall (it would cost five health points), and the bot
   ended on an acacia branch with no floored leaf exit and no placement
   surface in view. Deep canopies over drops of more than five blocks remain an
   open limitation (the same class as `pc2-random-val/episode-0001`).

The three revisions are not a controlled comparison: spawn position inside a
world is not deterministic, and each revision trained new policies. They show
which failure modes each fix removed, not an effect size.

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
- A bot on a canopy whose only drop is longer than five blocks (high acacia
  and dark oak, or terrain far below) can still be stranded.
- No result here shows that learning beats the masked random baseline; with
  the mask and helpers this milestone is nearly solved by chance.
