# Natural-resource run summary

**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION**

| Group | Attempted | Completed | Invalid | Successes | Success rate | Wilson 95% | Deaths | Truncated | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked / frozen_evaluation / random / masked_random / max_steps 60 | 20 | 20 | 0 | 19 | 0.95 | 0.76–0.99 | 0 | 1 | 17.4 / 15.2 | 50.2 / 49.9 | 23 |
| masked / frozen_evaluation / random / masked_random / max_steps 60 / 2 parallel slots in schedule bench-2slots:aa1a66ef7229 | 20 | 20 | 0 | 17 | 0.85 | 0.64–0.95 | 0 | 3 | 20.9 / 13.9 | 58.8 / 43.3 | 30 |
| masked / frozen_evaluation / random / masked_random / max_steps 60 / 4 parallel slots in schedule bench-4slots:3a72a8ef4361 | 20 | 20 | 0 | 18 | 0.90 | 0.70–0.97 | 1 | 1 | 16.2 / 13.6 | 59.9 / 40.8 | 25 |

## Decisions

How much each decision was left to the policy: the share with a single allowed action, the share of choices the disclosed mask would have forbidden, and the share that changed none of the nine resource_v1 fields.

| Group | Decisions | Single allowed action | Choices outside the mask | No progress |
| --- | --- | --- | --- | --- |
| masked / frozen_evaluation / random / masked_random / max_steps 60 | 349 | 69% | 0% | 39% |
| masked / frozen_evaluation / random / masked_random / max_steps 60 / 2 parallel slots in schedule bench-2slots:aa1a66ef7229 | 417 | 72% | 0% | 51% |
| masked / frozen_evaluation / random / masked_random / max_steps 60 / 4 parallel slots in schedule bench-4slots:3a72a8ef4361 | 325 | 51% | 0% | 26% |

## masked / frozen_evaluation / random / masked_random / max_steps 60

Runs: bench1-masked-random-c1, bench1-masked-random-c2, bench1-masked-random-c3, bench1-masked-random-c4. World seeds: 30000, 30001, 30002, 30003, 30004, 30005, 30006, 30007, 30008, 30009, 30010, 30011, 30012, 30013, 30014, 30015, 30016, 30017, 30018, 30019.
Mean frozen-policy shaped return: 8.907.
Leaf clearing: 23 canopy layers, 4 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): pickaxe_not_crafted ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×58, leaf_escape_refused:no_leaf_exit ×53, exploration_blocked ×50, explored ×47, log_path_blocked ×23, crafted_crafting_table ×20, table_placed ×20, crafted_stick ×19.

## masked / frozen_evaluation / random / masked_random / max_steps 60 / 2 parallel slots in schedule bench-2slots:aa1a66ef7229

Runs: bench2-masked-random-c1, bench2-masked-random-c2, bench2-masked-random-c3, bench2-masked-random-c4. World seeds: 30000, 30001, 30002, 30003, 30004, 30005, 30006, 30007, 30008, 30009, 30010, 30011, 30012, 30013, 30014, 30015, 30016, 30017, 30018, 30019.
Mean frozen-policy shaped return: 7.908.
Leaf clearing: 30 canopy layers, 5 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×1, pickaxe_not_crafted ×1, table_not_placed ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: exploration_blocked ×112, canopy_descent_refused:fall_too_far ×58, log_collected ×54, explored ×53, explored_locally ×24, crafted_stick ×19, crafted_crafting_table ×18, table_placed ×18.

## masked / frozen_evaluation / random / masked_random / max_steps 60 / 4 parallel slots in schedule bench-4slots:3a72a8ef4361

Runs: bench4-masked-random-c1, bench4-masked-random-c2, bench4-masked-random-c3, bench4-masked-random-c4. World seeds: 30000, 30001, 30002, 30003, 30004, 30005, 30006, 30007, 30008, 30009, 30010, 30011, 30012, 30013, 30014, 30015, 30016, 30017, 30018, 30019.
Mean frozen-policy shaped return: 7.936.
Leaf clearing: 25 canopy layers, 5 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): death ×1, pickaxe_not_crafted ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: explored ×61, log_collected ×54, log_broken_without_pickup ×22, crafted_crafting_table ×19, table_placed ×19, crafted_stick ×18, crafted_wooden_pickaxe ×18, table_reached ×17.

## Episodes

| Run | Episode | Seed | Status | Success | Death | Steps | Task s | Failure cause | Final event |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bench1-masked-random-c1 | 1 | 30000 | completed | True | False | 31 | 140.5 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c1 | 2 | 30001 | completed | True | False | 13 | 33.6 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c1 | 3 | 30002 | completed | True | False | 26 | 153.3 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c1 | 4 | 30003 | completed | True | False | 11 | 23.1 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c1 | 5 | 30004 | completed | True | False | 17 | 112.0 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c2 | 1 | 30005 | completed | True | False | 15 | 31.6 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c2 | 2 | 30006 | completed | True | False | 38 | 201.6 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c2 | 3 | 30007 | completed | True | False | 12 | 29.6 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c2 | 4 | 30008 | completed | False | False | 60 | 56.8 | pickaxe_not_crafted | leaf_escape_refused:no_leaf_exit;exploration_blocked |
| bench1-masked-random-c2 | 5 | 30009 | completed | True | False | 11 | 17.3 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c3 | 1 | 30010 | completed | True | False | 12 | 21.8 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c3 | 2 | 30011 | completed | True | False | 10 | 17.5 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c3 | 3 | 30012 | completed | True | False | 11 | 20.6 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c3 | 4 | 30013 | completed | True | False | 12 | 26.1 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c3 | 5 | 30014 | completed | True | False | 11 | 19.4 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c4 | 1 | 30015 | completed | True | False | 11 | 17.4 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c4 | 2 | 30016 | completed | True | False | 12 | 18.3 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c4 | 3 | 30017 | completed | True | False | 12 | 21.9 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c4 | 4 | 30018 | completed | True | False | 13 | 26.9 | success | crafted_wooden_pickaxe |
| bench1-masked-random-c4 | 5 | 30019 | completed | True | False | 11 | 15.7 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c1 | 1 | 30000 | completed | False | False | 60 | 360.8 | table_not_placed | explored_locally |
| bench2-masked-random-c1 | 2 | 30001 | completed | True | False | 15 | 25.8 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c1 | 3 | 30002 | completed | True | False | 25 | 167.6 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c1 | 4 | 30003 | completed | True | False | 12 | 21.2 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c1 | 5 | 30004 | completed | True | False | 11 | 22.5 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c2 | 1 | 30005 | completed | True | False | 21 | 173.1 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c2 | 2 | 30006 | completed | True | False | 13 | 26.8 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c2 | 3 | 30007 | completed | True | False | 14 | 35.7 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c2 | 4 | 30008 | completed | False | False | 60 | 43.2 | no_wood_collected | exploration_blocked |
| bench2-masked-random-c2 | 5 | 30009 | completed | True | False | 11 | 17.5 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c3 | 1 | 30010 | completed | True | False | 11 | 21.7 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c3 | 2 | 30011 | completed | True | False | 11 | 17.2 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c3 | 3 | 30012 | completed | False | False | 60 | 36.4 | pickaxe_not_crafted | canopy_descent_refused:fall_too_far;exploration_blocked |
| bench2-masked-random-c3 | 4 | 30013 | completed | True | False | 11 | 14.9 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c3 | 5 | 30014 | completed | True | False | 11 | 17.6 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c4 | 1 | 30015 | completed | True | False | 11 | 21.6 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c4 | 2 | 30016 | completed | True | False | 17 | 28.4 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c4 | 3 | 30017 | completed | True | False | 11 | 19.2 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c4 | 4 | 30018 | completed | True | False | 11 | 18.1 | success | crafted_wooden_pickaxe |
| bench2-masked-random-c4 | 5 | 30019 | completed | True | False | 21 | 86.4 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c1 | 1 | 30000 | completed | False | False | 60 | 326.9 | pickaxe_not_crafted | table_path_blocked |
| bench4-masked-random-c1 | 2 | 30001 | completed | True | False | 13 | 39.0 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c1 | 3 | 30002 | completed | True | False | 25 | 162.1 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c1 | 4 | 30003 | completed | True | False | 11 | 14.9 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c1 | 5 | 30004 | completed | True | False | 25 | 110.7 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c2 | 1 | 30005 | completed | True | False | 11 | 31.6 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c2 | 2 | 30006 | completed | False | True | 21 | 136.8 | death | exploration_time_budget;exploration_blocked |
| bench4-masked-random-c2 | 3 | 30007 | completed | True | False | 14 | 31.2 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c2 | 4 | 30008 | completed | True | False | 14 | 40.4 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c2 | 5 | 30009 | completed | True | False | 11 | 24.9 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c3 | 1 | 30010 | completed | True | False | 11 | 23.1 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c3 | 2 | 30011 | completed | True | False | 11 | 20.4 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c3 | 3 | 30012 | completed | True | False | 20 | 79.4 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c3 | 4 | 30013 | completed | True | False | 11 | 18.2 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c3 | 5 | 30014 | completed | True | False | 10 | 14.7 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c4 | 1 | 30015 | completed | True | False | 11 | 18.8 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c4 | 2 | 30016 | completed | True | False | 11 | 23.4 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c4 | 3 | 30017 | completed | True | False | 12 | 24.1 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c4 | 4 | 30018 | completed | True | False | 11 | 21.3 | success | crafted_wooden_pickaxe |
| bench4-masked-random-c4 | 5 | 30019 | completed | True | False | 12 | 36.4 | success | crafted_wooden_pickaxe |

## Reading these numbers

- Success means a wooden pickaxe was crafted from naturally gathered wood and confirmed held; it is not full Survival or an Ender Dragon result.
- Rates are computed over completed episodes only; invalid or interrupted episodes are counted separately and never treated as policy failures or successes.
- The Wilson 95% interval describes these episodes for one checkpoint or baseline under one step limit; it does not measure variation across independently trained agents.
- Returns from online training include exploratory actions and shaped rewards; compare them only with other training runs, not with frozen evaluation.
- The feasibility mask and scripted helpers supply substantial task knowledge; compare learned policies against the random baseline of the same condition (masked random with the mask, uniform random over all eight actions without it) before claiming a benefit.
- Conditions are never pooled. A benefit is claimed only when a learned policy's Wilson interval lies entirely above its baseline's; overlapping intervals mean no benefit was shown. Fisher's exact test is reported as a supplement, not as the decision rule.
- Runs recorded before experiment conditions existed are reported as the masked condition, which is how they ran.
- Runs with a different parallel setup (live episodes at once on one machine) are separate groups and are never compared, because task time and the helpers' time budgets are wall-clock. Parallel runs are grouped and compared only within one schedule, whose plan rules (interleaved arms, evaluations only after all training) make the test arms share the load; that schedule's schedule.json records the concurrency each run actually had. Runs recorded before resource-parallel existed ran one episode at a time and are reported with one slot.
