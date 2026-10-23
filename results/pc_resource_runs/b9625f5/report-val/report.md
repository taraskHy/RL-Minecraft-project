# Natural-resource run summary

**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION**

| Group | Attempted | Completed | Invalid | Successes | Success rate | Wilson 95% | Deaths | Truncated | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_evaluation / q_learning / checkpoint a1f5984de1db / max_steps 60 | 5 | 4 | 1 | 3 | 0.75 | 0.30–0.95 | 0 | 1 | 28 / 17.3 | 45.2 / 55.2 | 3 |
| frozen_evaluation / random / masked_random / max_steps 60 | 10 | 10 | 0 | 8 | 0.80 | 0.49–0.94 | 0 | 2 | 25.6 / 17 | 40.8 / 44.1 | 15 |
| frozen_evaluation / sarsa / checkpoint c3b80be8eefd / max_steps 60 | 10 | 10 | 0 | 9 | 0.90 | 0.60–0.98 | 0 | 1 | 22 / 17.8 | 54.6 / 58.2 | 3 |

## frozen_evaluation / q_learning / checkpoint a1f5984de1db / max_steps 60

Runs: pc2-q-frozen-val. World seeds: 10000, 10001, 10002, 10003, 10004.
Evaluated checkpoint (policy.json SHA-256): a1f5984de1db88df6b8ec495256bdd02707a502ed1dcf750a02f17a4fa12bce5.
Mean frozen-policy shaped return: 6.617.
Leaf clearing: 3 canopy layers, 0 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): table_not_placed ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): invalid ×1.
Most frequent step events: no_placement_surface ×51, explored ×39, log_collected ×14, crafted_oak_planks ×7, log_path_blocked ×5, explored_locally ×5, crafted_stick ×5, crafted_crafting_table ×5.

## frozen_evaluation / random / masked_random / max_steps 60

Runs: pc2-random-val. World seeds: 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009.
Mean frozen-policy shaped return: 7.291.
Leaf clearing: 15 canopy layers, 2 wall blocks, 1 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×1, pickaxe_not_crafted ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: canopy_descent_refused:no_landing_within_scan ×59, exploration_blocked ×59, table_path_blocked ×50, explored ×30, log_collected ×27, crafted_oak_planks ×16, crafted_stick ×9, crafted_crafting_table ×9.

## frozen_evaluation / sarsa / checkpoint c3b80be8eefd / max_steps 60

Runs: pc2-sarsa-frozen-val. World seeds: 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009.
Evaluated checkpoint (policy.json SHA-256): c3b80be8eefd1ebcef787a7a72423c30d56533f7bb58949fb94d35001ddd0f20.
Mean frozen-policy shaped return: 8.299.
Leaf clearing: 3 canopy layers, 2 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): pickaxe_not_crafted ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: explored ×56, table_path_blocked ×50, log_collected ×30, crafted_oak_planks ×24, crafted_stick ×10, crafted_crafting_table ×10, table_placed ×10, log_reached ×9.

## Episodes

| Run | Episode | Seed | Status | Success | Death | Steps | Task s | Failure cause | Final event |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pc2-q-frozen-val | 1 | 10000 | completed | True | False | 30 | 116.5 | success | crafted_wooden_pickaxe |
| pc2-q-frozen-val | 2 | 10001 | completed | True | False | 11 | 17.0 | success | crafted_wooden_pickaxe |
| pc2-q-frozen-val | 3 | 10002 | completed | False | False | 60 | 15.2 | table_not_placed | no_placement_surface |
| pc2-q-frozen-val | 4 | 10003 | completed | True | False | 11 | 32.2 | success | crafted_wooden_pickaxe |
| pc2-q-frozen-val | 5 | 10004 | invalid | None | None | 36 | n/a | invalid_run | crafted_crafting_table |
| pc2-sarsa-frozen-val | 1 | 10000 | completed | True | False | 23 | 86.7 | success | crafted_wooden_pickaxe |
| pc2-sarsa-frozen-val | 2 | 10001 | completed | True | False | 11 | 17.0 | success | crafted_wooden_pickaxe |
| pc2-sarsa-frozen-val | 3 | 10002 | completed | True | False | 13 | 19.1 | success | crafted_wooden_pickaxe |
| pc2-sarsa-frozen-val | 4 | 10003 | completed | True | False | 11 | 19.7 | success | crafted_wooden_pickaxe |
| pc2-sarsa-frozen-val | 5 | 10004 | completed | True | False | 34 | 172.8 | success | crafted_wooden_pickaxe |
| pc2-sarsa-frozen-val | 6 | 10005 | completed | False | False | 60 | 22.4 | pickaxe_not_crafted | table_path_blocked |
| pc2-sarsa-frozen-val | 7 | 10006 | completed | True | False | 11 | 16.0 | success | crafted_wooden_pickaxe |
| pc2-sarsa-frozen-val | 8 | 10007 | completed | True | False | 10 | 15.4 | success | crafted_wooden_pickaxe |
| pc2-sarsa-frozen-val | 9 | 10008 | completed | True | False | 15 | 37.6 | success | crafted_wooden_pickaxe |
| pc2-sarsa-frozen-val | 10 | 10009 | completed | True | False | 32 | 139.6 | success | crafted_wooden_pickaxe |
| pc2-random-val | 1 | 10000 | completed | False | False | 60 | 36.4 | no_wood_collected | canopy_descent_refused:no_landing_within_scan;exploration_blocked |
| pc2-random-val | 2 | 10001 | completed | True | False | 11 | 33.9 | success | crafted_wooden_pickaxe |
| pc2-random-val | 3 | 10002 | completed | True | False | 18 | 19.4 | success | crafted_wooden_pickaxe |
| pc2-random-val | 4 | 10003 | completed | True | False | 20 | 32.6 | success | crafted_wooden_pickaxe |
| pc2-random-val | 5 | 10004 | completed | True | False | 26 | 99.6 | success | crafted_wooden_pickaxe |
| pc2-random-val | 6 | 10005 | completed | True | False | 11 | 14.9 | success | crafted_wooden_pickaxe |
| pc2-random-val | 7 | 10006 | completed | True | False | 12 | 23.5 | success | crafted_wooden_pickaxe |
| pc2-random-val | 8 | 10007 | completed | False | False | 60 | 18.8 | pickaxe_not_crafted | table_path_blocked |
| pc2-random-val | 9 | 10008 | completed | True | False | 19 | 63.7 | success | crafted_wooden_pickaxe |
| pc2-random-val | 10 | 10009 | completed | True | False | 19 | 65.2 | success | crafted_wooden_pickaxe |

## Reading these numbers

- Success means a wooden pickaxe was crafted from naturally gathered wood and confirmed held; it is not full Survival or an Ender Dragon result.
- Rates are computed over completed episodes only; invalid or interrupted episodes are counted separately and never treated as policy failures or successes.
- The Wilson 95% interval describes these episodes for one checkpoint or baseline under one step limit; it does not measure variation across independently trained agents.
- Returns from online training include exploratory actions and shaped rewards; compare them only with other training runs, not with frozen evaluation.
- The feasibility mask and scripted helpers supply substantial task knowledge; compare learned policies against the masked random baseline with the same helpers before claiming a benefit.
