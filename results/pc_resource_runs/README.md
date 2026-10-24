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

Four code revisions were run. Each of the first three fixed failure modes that
the batch before it exposed, and all are retained so the failures stay
visible. The fourth, `87673b8`, ran the pre-registered experiment that asks
whether learning beats random play once the feasibility mask is removed.

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
   mask allowed `SEARCH` or `APPROACH_TABLE`. Both states had all-zero Q values,
   because three training episodes never visited them, so the frozen policy
   broke ties at random (13 searches, 13 walks back) and each search was undone
   by a walk back until truncation. The masked random baseline and the frozen
   SARSA policy both solved this seed, so this is a coverage gap in that Q
   table, trained on three episodes, in a treeless biome, not a helper failure.
2. *A high acacia canopy* (`pc3-sarsa-frozen-val/episode-0004`, seed 10003).
   The bot started on acacia leaves 8 blocks above the ground. The descent
   correctly refused that fall (it would cost five health points), and the bot
   ended on an acacia branch with no floored leaf exit and no placement
   surface in view. Deep canopies over drops of more than five blocks remain an
   open limitation (the same class as `pc2-random-val/episode-0001`).

The three revisions are not a controlled comparison: spawn position inside a
world is not deterministic, and each revision trained new policies. They show
which failure modes each fix removed, not an effect size.

## Revision `87673b8`: the pre-registered masking experiment

Folder `87673b8/`. The question, conditions, budgets, seeds, order and decision
rules were committed before any of these runs, in
[`configs/resource_masking_experiment.json`](../../configs/resource_masking_experiment.json):
**without the feasibility mask, do tabular Q-learning and SARSA beat random
play with the same execution helpers?**

- **Setup.** Every run used the code of commit `87673b8`. The batch started
  at that commit. All 15 run folders record the same SHA-256 digests of the
  six recorded sources (bridge, its lockfile, learner, live environment,
  session and server), and those digests match that commit. Each episode had
  one fresh world, `--max-steps 60`, `--pause 0` and agent seed 0, with no
  simulator pretraining.
- **Documentation commits during the batch.** Four commits (`c6623bc`,
  `e19d162`, `81a9f6f`, `3b5c609`) touched only `README.md` and
  `TEAM_STATUS.md`, to post status for the team. They changed no code, but
  HEAD had moved past `87673b8` when 12 of the 15 runs started.
- **The batch.** It ran in the declared order from 20:48 on 23 October to
  04:17 on 24 October, with no manual step recorded. `batch/masking_batch.sh`
  is a copy of the batch script, saved after the run. Its logic matches every
  line of `batch/status.txt` (the log) and the continuation's seed and
  epsilon, but no digest was taken at launch. Their SHA-256 digests are
  `ba768155…` and `7efda1b3…`; they are kept beside the runs, not listed in
  `retention.json`.
- **The log's `EXIT` values are always 0.** The script expands
  `$(basename …)` before `$?`, so they are not the runs' exit codes. This was
  harmless here: the script decides continuations by counting episode
  folders, not by exit codes. Every run outcome in this section comes from
  the run folders.
- **Reports.** `report-test/` covers the eight frozen test runs and
  `report-training/` the seven training folders.

| Condition | Allowed actions | State | Random baseline |
| --- | --- | --- | --- |
| `masked` | the disclosed mask `recipe_and_budget_v1` | `resource_v1` | masked random |
| `unmasked` | all eight, always; infeasible ones fail as observed transitions | `resource_v1` | uniform random over all eight |
| `unmasked_history` | all eight, always | `resource_v2` (adds `last_action_failed`) | the same uniform random run |

### Held-out test: frozen policies on worlds 20000–20019

All 160 test episodes were measured, with no deaths and no invalid episodes.
"Outside the mask" is the share of choices the disclosed mask would have
forbidden. "No progress" is the share of decisions that changed none of the
nine `resource_v1` fields.

| Condition | Policy | Successes | Wilson 95% | Mean decisions (all / success) | Mean task s (all / success) | Outside the mask | No progress |
| --- | --- | --- | --- | --- | --- | --- | --- |
| masked | frozen Q-learning | 16/20 | 0.58–0.92 | 22.1 / 12.6 | 37.3 / 29.0 | 0% | 56% |
| masked | frozen SARSA | 18/20 | 0.70–0.97 | 19.2 / 14.7 | 35.0 / 32.6 | 0% | 46% |
| masked | masked random | 19/20 | 0.76–0.99 | 14.6 / 12.2 | 31.2 / 29.1 | 0% | 26% |
| unmasked | frozen Q-learning | 20/20 | 0.84–1.00 | 14.1 / 14.1 | 39.1 / 39.1 | 2% | 19% |
| unmasked | frozen SARSA | 14/20 | 0.48–0.85 | 26.1 / 11.6 | 39.8 / 22.3 | 41% | 60% |
| unmasked | uniform random | 4/20 | 0.08–0.42 | 58.6 / 53 | 41.9 / 35.0 | 81% | 76% |
| unmasked_history | frozen Q-learning | 16/20 | 0.58–0.92 | 25.1 / 16.3 | 29.5 / 24.8 | 29% | 58% |
| unmasked_history | frozen SARSA | 17/20 | 0.64–0.95 | 21.8 / 15.1 | 28.4 / 23.7 | 32% | 52% |

With the mask, 79% (Q-learning), 70% (SARSA) and 54% (random) of decisions
had a single allowed action.

**Pre-registered verdicts.** The comparisons count all 20 seeds, because no arm
lost one. A benefit is claimed only when the learned Wilson interval lies
entirely above the baseline's. Fisher's exact p is two-sided and only a
supplement.

| Comparison | Learned | Baseline | Fisher p | Verdict |
| --- | --- | --- | --- | --- |
| masked Q-learning vs masked random | 16/20 | 19/20 | 0.342 | no benefit shown |
| masked SARSA vs masked random | 18/20 | 19/20 | 1 | no benefit shown |
| unmasked Q-learning vs uniform random | 20/20 | 4/20 | 1.54e-07 | **benefit shown** |
| unmasked SARSA vs uniform random | 14/20 | 4/20 | 0.00364 | **benefit shown** |
| history Q-learning vs uniform random | 16/20 | 4/20 | 0.00036 | **benefit shown** |
| history SARSA vs uniform random | 17/20 | 4/20 | 8.75e-05 | **benefit shown** |
| history bit, Q-learning (with vs without) | 16/20 | 20/20 | 0.106 | no effect of the bit shown |
| history bit, SARSA (with vs without) | 17/20 | 14/20 | 0.451 | no effect of the bit shown |

**What the comparison shows.**

1. **With the mask, random play already succeeds almost every time** (19 of
   20), so this comparison cannot show a benefit of learning. The
   pre-registration predicted that it would read "no benefit shown" whatever
   the learned policies did, because against 14 or more baseline successes the
   rule cannot detect a benefit. The masked learned policies did slightly
   worse on point estimates, with more decisions and task seconds.
2. **Without the mask, every learned checkpoint beat uniform random play** on
   the 20 test worlds. There were four of them, each trained for 50 live
   episodes: one agent per algorithm and state version, with 49 measured for
   history SARSA.
   - **What random play did.** It spent 81% of its choices on actions the
     mask would have forbidden, and ran out of decisions on 16 of the 20
     worlds.
   - **What the best policy did.** The frozen unmasked Q-learning policy
     chose such an action 2% of the time and succeeded on every world.
   - **False positives.** The rule's false-positive rate is about 0.32% per
     comparison at worst (0.322%, when both arms have a true success rate of
     0.5).
3. **No effect of the history bit was shown for either algorithm**, because
   the intervals overlap. The point estimates moved in opposite directions:
   Q-learning 16/20 with the bit against 20/20 without, SARSA 17/20 against
   14/20. At this sample size, that is not evidence that the bit has no
   effect. With one trained agent per condition and algorithm, it compares two
   checkpoints, not two state designs.

### Training (live, with exploration; not test results)

| Condition | Algorithm | Attempted | Successes | Per ten attempted episodes | Live decisions | Deaths | Invalid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| masked | Q-learning | 20 | 17 | 9, 8 | 389 | 0 | 0 |
| masked | SARSA | 20 | 18 | 9, 9 | 277 | 1 | 0 |
| unmasked | Q-learning | 50 | 36 | 3, 8, 9, 7, 9 | 1701 | 0 | 0 |
| unmasked | SARSA | 50 | 41 | 6, 8, 9, 9, 9 | 1444 | 0 | 0 |
| unmasked_history | Q-learning | 50 | 29 | 5, 6, 6, 4, 8 | 1997 | 0 | 0 |
| unmasked_history | SARSA | 50 (49 measured) | 37 | 6, 7, 8, 8, 8 | 1811 | 0 | 1 (seed 123) |

Epsilon fell linearly from 0.6 to 0.1, so training successes include
exploration and must not be compared with the frozen test.

- **History SARSA's 1811 decisions.** 22 of them were in the invalid seed-123
  episode, leaving 1789 in the 49 measured ones. The last of those 22 updates
  was recorded but never saved to the checkpoint that training resumed from
  (see the deviations below).
- **Seeds no run solved.** Seeds 101 and 118 failed in all six training runs,
  and seed 138 in all four 50-episode runs.
- **The one death.** It happened in masked SARSA training on seed 118, during
  the episode's fourth decision, a `SEARCH` across desert sand.
  - The server log records the bot as "pricked to death", by a cactus, and
    its health fell from 20 to 0.
  - The `SEARCH` ran on until its time budget ran out, so the step's event
    reads `exploration_time_budget;exploration_blocked`. That event came
    after the death and did not cause it.

### Why the learned policies failed on the test worlds

The six learned policies failed 19 times between them, in four ways.

1. **Stranded on a canopy (10 failures).**
   - Masked Q-learning on 20000, 20012, 20013 and 20016.
   - Masked SARSA on 20012 and 20015.
   - History Q-learning on 20012 and 20016.
   - History SARSA on 20013 and 20016.

   The bot spawned on leaves above a drop the descent helper judged unsafe,
   and all ten episodes ended still on the canopy.
   - In nine of them, the helper refused the drop 41 to 59 times. The reasons
     were `fall_too_far`, once `no_landing_within_scan` on masked SARSA
     20012, and `hazard_below` in the swamp at 20000.
   - The tenth, history SARSA on 20016, mixed 10 refusals with 12 refused
     crafts.
   - In five of the ten, logs chopped from the canopy fell out of reach
     (`log_broken_without_pickup`).
   - Unmasked SARSA on 20002, counted under kind 2, also started and ended on
     a canopy. It never tried to descend.

   This is the deep-canopy limitation already recorded at `5bb7b25`. Unmasked
   Q-learning succeeded from all nine of its canopy starts.
2. **A loop of useless actions after running short of wood (7 failures).**
   - Unmasked SARSA on 20002, 20010, 20015 and 20017.
   - History Q-learning on 20005 and 20009.
   - History SARSA on 20006.

   - **What ran out.** All seven ended with no logs and at most two planks.
     Six had already placed a table and lacked the third plank that a pickaxe
     needs. History SARSA on 20006 never made a table at all.
   - **Why.** Six of the seven had collected only one or two logs, fewer than
     the three that sticks, a table and a pickaxe need together. Only history
     Q-learning on 20009 lost its planks to surplus sticks.
   - **How the rest of the episode went.** About 69% of the remaining steps
     were refused crafts, with 11 to 55 `missing_materials` or
     `not_at_own_table` refusals per episode. The other steps placed a table
     that was already placed, or walked to the table the bot stood at.
     History Q-learning also chose `CHOP` 20 and 12 times; 13 and 9 of those
     found no reachable log.
   - **Why it never escapes.** A refused action leaves the state unchanged,
     so the frozen policy faces the same state again, and it picks
     uniformly among the tied best actions. The mechanism differed by policy:
     - *Unmasked SARSA, 4 loops.* Five actions never tried in the loop state
       kept their initial value 0: `CRAFT_PLANKS`, `CRAFT_TABLE`,
       `PLACE_TABLE`, `APPROACH_TABLE` and `CRAFT_PICKAXE`. They sat just
       above `SEARCH` and `CHOP` at −0.014, the only actions that could help,
       and the policy rotated among the five until the step limit.
     - *History Q-learning, 2 loops.* `CHOP` was among the actions tied at
       0, but no log was in reach.
     - *History SARSA on 20006.* `CRAFT_PLANKS` and `CRAFT_PICKAXE` tied as
       the best values (−0.013), and the policy repeated those two refused
       crafts.
   - **The history bit did not remove the loop.** The history policies chose
     outside the mask 29% (Q-learning) and 32% (SARSA) of the time, against
     2% and 41% for the unmasked policies without the bit.
3. **Out of decisions in unfamiliar states (1 failure).** Unmasked SARSA on
   20001 did not loop. It visited 23 distinct states, and in 44 of its 60
   decisions every Q value was 0, so it chose by random tie-breaks in states
   training never reached. It ended at its placed table with enough materials
   for a pickaxe.
4. **No wood found (1 failure).** Unmasked SARSA on 20007 spawned at y 45,
   below the surface of a sparse jungle. It explored locally 49 times without
   reaching a log.

The baselines failed differently.
- **Masked random failed once, on 20009.** It spawned on snow at y 126 in
  frozen peaks, and its first search took it onto spruce leaves. The canopy
  descent then refused the drop (`fall_too_far`) 51 times, so it too ended
  stranded on a canopy.
- **Uniform random failed 16 times.** 10 placed a table but never crafted the
  pickaxe, and 6 never placed a table. Those 16 failures hold 255
  `missing_materials` refusals (311 over all 20 episodes).

### The same seed does not give the same start

Minecraft chooses the spawn point within a world non-deterministically.

- **No seed gave all eight test arms the same spawn point.** On each seed, the
  largest horizontal distance between two arms' spawns was 16.0 to 25.6
  blocks (median 22.0).
- **Canopy starts varied by arm.** On 13 of the 20 seeds, some arms started on
  a tree canopy and others on the ground. Each arm had between 5 and 9 canopy
  starts:

| Arm | Canopy starts (successes) | Ground starts (successes) |
| --- | --- | --- |
| masked Q-learning | 8 (4) | 12 (12) |
| masked SARSA | 8 (6) | 12 (12) |
| masked random | 6 (6) | 14 (13) |
| unmasked Q-learning | 9 (9) | 11 (11) |
| unmasked SARSA | 7 (5) | 13 (9) |
| uniform random | 7 (0) | 13 (4) |
| history Q-learning | 6 (4) | 14 (12) |
| history SARSA | 5 (3) | 15 (14) |

A seed therefore fixes the world, not the start within it. Arms on the same
seed began up to 25.6 blocks apart and up to 19 blocks apart in height. On 5
of the 20 seeds they also began in different biomes: 20000, 20007, 20009,
20014 and 20018. The comparisons pair the same worlds, not identical starts.

Which arms faced a canopy start depended partly on where each one happened to
spawn. Masked Q-learning, for example, succeeded from all 12 of its ground
starts but only 4 of its 8 canopy starts. On world 20013 it spawned on a
canopy at y 84 and failed, while unmasked Q-learning spawned on the ground at
y 72; those two runs differ in policy as well as start.

### Deviations from the plan, and limits

- **One invalid episode.** History SARSA training episode 24 (seed 123) became
  invalid when Windows refused the `policy.tmp` → `policy.json` replace
  (WinError 5). Most likely another program, such as a virus scanner or the
  search indexer, briefly had `policy.json` open; the logs record only the
  access-denied error.
  - **The continuation.** The declared continuation ran the remaining 26
    episodes, from seed 124, in `mx-history-sarsa-train-part1`. It started
    from the last saved policy at the declared exploration rate 0.355102.
  - **What that policy lacks.** The update from the episode's 22nd and last
    recorded decision. One of its 1,440 Q values (180 states × 8 actions)
    differs from the unsaved copy: `PLACE_TABLE` in that state is −0.0146
    saved and 0.877 unsaved.
  - **The evidence.** The unsaved copy is kept beside the runs as
    `batch/mx-history-sarsa-train-unsaved-policy.tmp` (SHA-256 `416ce283…`,
    not listed in `retention.json`). The same difference can be rebuilt from
    `episode-0024/transitions.jsonl`.
  - **What was tested.** The tested history SARSA policy is the
    continuation's final one.
  - **The retry fix.** A retry for this lock was committed during the batch,
    about two minutes after the failure. It is `0a87040`, on the separate
    `codex/parallel-runs` branch, and no run in this batch used it.
- **One trained agent per condition and algorithm.** The runs show that the
  task is learnable here, not how reliably every training run learns it.
  The difference between unmasked Q-learning (20/20) and history Q-learning
  (16/20) is a difference between two checkpoints.
- **Wide intervals.** Twenty test worlds per arm make every interval wide.
  "No benefit shown" means not detectable at this sample size.
- **What is learned.** Only the order of eight macro-actions is learned. The
  helpers (search, chop, craft, place, approach) are scripted, and the reward
  uses potential-based shaping.

## What these runs do not show

- No stone tools, food, shelter, Nether, stronghold or dragon work exists.
- Three development seeds and ten validation seeds are a small sample; the
  intervals say so.
- Spawn position inside a world is not deterministic, so "seed 1" is terrain,
  not a replay; canopy starts are a matter of sampling.
- The compact `resource_v1` state cannot tell a repeated identical failure
  from a first attempt. The `resource_v2` "last action failed" bit was tested
  at `87673b8`, and no effect on success was shown at this sample size.
  Frozen policies with the bit still looped through useless actions.
- Initial Q values of 0, with a negative step cost, make actions a policy
  never tried in a state look better than useful actions it tried there.
  That drove most of the frozen loops at `87673b8`.
- A bot on a canopy whose only drop is longer than five blocks (high acacia
  and dark oak, or terrain far below) can still be stranded. At `87673b8` this
  caused 10 of the 19 learned-policy test failures.
- Within the feasibility mask, no result shows learning beating random play:
  the mask and helpers nearly solve this milestone by chance. Without the mask
  (`87673b8`), learned policies beat uniform random play. That result is about
  choosing among eight macro-actions with scripted helpers, not about learning
  low-level control.
- Each condition and algorithm was trained once, with one agent seed, so
  nothing here measures how much learning varies between training runs.
