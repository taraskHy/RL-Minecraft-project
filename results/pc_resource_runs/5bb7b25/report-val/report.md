# Natural-resource run summary

**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION**

| Group | Attempted | Completed | Invalid | Successes | Success rate | Wilson 95% | Deaths | Truncated | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_evaluation / q_learning / checkpoint 9fc5c3696093 / max_steps 60 | 10 | 10 | 0 | 9 | 0.90 | 0.60–0.98 | 0 | 1 | 17.6 / 12.9 | 61.8 / 28.2 | 13 |
| frozen_evaluation / random / masked_random / max_steps 60 | 10 | 10 | 0 | 10 | 1.00 | 0.72–1.00 | 0 | 0 | 14.8 / 14.8 | 43.2 / 43.2 | 1 |
| frozen_evaluation / sarsa / checkpoint aabbc34453d5 / max_steps 60 | 10 | 10 | 0 | 9 | 0.90 | 0.60–0.98 | 0 | 1 | 18.8 / 14.2 | 46.7 / 41.7 | 14 |

## frozen_evaluation / q_learning / checkpoint 9fc5c3696093 / max_steps 60

Runs: pc3-q-frozen-val. World seeds: 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009.
Evaluated checkpoint (policy.json SHA-256): 9fc5c3696093710481010d13f81ea78eecf9f0c93a92055366e8621d0560669d.
Mean frozen-policy shaped return: 8.391.
Leaf clearing: 13 canopy layers, 0 wall blocks, 3 fall damage in total.
Failure causes (completed episodes only): pickaxe_not_crafted ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: explored ×47, log_collected ×29, crafted_oak_planks ×19, table_reached ×16, crafted_stick ×10, crafted_crafting_table ×10, table_placed ×10, crafted_wooden_pickaxe ×9.

## frozen_evaluation / random / masked_random / max_steps 60

Runs: pc3-random-val. World seeds: 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009.
Mean frozen-policy shaped return: 9.487.
Leaf clearing: 1 canopy layers, 1 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: explored ×37, log_collected ×30, crafted_oak_planks ×18, crafted_crafting_table ×10, crafted_stick ×10, table_placed ×10, crafted_wooden_pickaxe ×10, log_reached ×8.

## frozen_evaluation / sarsa / checkpoint aabbc34453d5 / max_steps 60

Runs: pc3-sarsa-frozen-val. World seeds: 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009.
Evaluated checkpoint (policy.json SHA-256): aabbc34453d5b098a6140911dba65700fc50a1c46a913b5eaf6054b38b4c6b3f.
Mean frozen-policy shaped return: 8.533.
Leaf clearing: 14 canopy layers, 0 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): table_not_placed ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: canopy_descent_refused:fall_too_far ×30, log_collected ×29, explored_locally ×21, crafted_oak_planks ×18, explored ×17, log_broken_without_pickup ×15, log_path_blocked ×14, crafted_stick ×10.

## Episodes

| Run | Episode | Seed | Status | Success | Death | Steps | Task s | Failure cause | Final event |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pc3-q-frozen-val | 1 | 10000 | completed | True | False | 11 | 19.4 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-val | 2 | 10001 | completed | True | False | 10 | 14.6 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-val | 3 | 10002 | completed | True | False | 13 | 22.3 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-val | 4 | 10003 | completed | True | False | 11 | 18.2 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-val | 5 | 10004 | completed | True | False | 26 | 96.7 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-val | 6 | 10005 | completed | True | False | 11 | 21.6 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-val | 7 | 10006 | completed | True | False | 11 | 15.2 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-val | 8 | 10007 | completed | True | False | 11 | 23.0 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-val | 9 | 10008 | completed | True | False | 12 | 23.0 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-val | 10 | 10009 | completed | False | False | 60 | 363.7 | pickaxe_not_crafted | explored |
| pc3-sarsa-frozen-val | 1 | 10000 | completed | True | False | 11 | 16.8 | success | crafted_wooden_pickaxe |
| pc3-sarsa-frozen-val | 2 | 10001 | completed | True | False | 11 | 17.6 | success | crafted_wooden_pickaxe |
| pc3-sarsa-frozen-val | 3 | 10002 | completed | True | False | 26 | 110.7 | success | crafted_wooden_pickaxe |
| pc3-sarsa-frozen-val | 4 | 10003 | completed | False | False | 60 | 91.6 | table_not_placed | leaf_escape_refused:no_leaf_exit;exploration_blocked |
| pc3-sarsa-frozen-val | 5 | 10004 | completed | True | False | 26 | 99.9 | success | crafted_wooden_pickaxe |
| pc3-sarsa-frozen-val | 6 | 10005 | completed | True | False | 11 | 47.2 | success | crafted_wooden_pickaxe |
| pc3-sarsa-frozen-val | 7 | 10006 | completed | True | False | 11 | 20.7 | success | crafted_wooden_pickaxe |
| pc3-sarsa-frozen-val | 8 | 10007 | completed | True | False | 10 | 14.8 | success | crafted_wooden_pickaxe |
| pc3-sarsa-frozen-val | 9 | 10008 | completed | True | False | 11 | 26.2 | success | crafted_wooden_pickaxe |
| pc3-sarsa-frozen-val | 10 | 10009 | completed | True | False | 11 | 21.2 | success | crafted_wooden_pickaxe |
| pc3-random-val | 1 | 10000 | completed | True | False | 11 | 22.5 | success | crafted_wooden_pickaxe |
| pc3-random-val | 2 | 10001 | completed | True | False | 10 | 19.6 | success | crafted_wooden_pickaxe |
| pc3-random-val | 3 | 10002 | completed | True | False | 11 | 22.2 | success | crafted_wooden_pickaxe |
| pc3-random-val | 4 | 10003 | completed | True | False | 11 | 15.3 | success | crafted_wooden_pickaxe |
| pc3-random-val | 5 | 10004 | completed | True | False | 26 | 100.7 | success | crafted_wooden_pickaxe |
| pc3-random-val | 6 | 10005 | completed | True | False | 12 | 38.5 | success | crafted_wooden_pickaxe |
| pc3-random-val | 7 | 10006 | completed | True | False | 11 | 19.6 | success | crafted_wooden_pickaxe |
| pc3-random-val | 8 | 10007 | completed | True | False | 11 | 18.9 | success | crafted_wooden_pickaxe |
| pc3-random-val | 9 | 10008 | completed | True | False | 20 | 76.7 | success | crafted_wooden_pickaxe |
| pc3-random-val | 10 | 10009 | completed | True | False | 25 | 98.1 | success | crafted_wooden_pickaxe |

## Reading these numbers

- Success means a wooden pickaxe was crafted from naturally gathered wood and confirmed held; it is not full Survival or an Ender Dragon result.
- Rates are computed over completed episodes only; invalid or interrupted episodes are counted separately and never treated as policy failures or successes.
- The Wilson 95% interval describes these episodes for one checkpoint or baseline under one step limit; it does not measure variation across independently trained agents.
- Returns from online training include exploratory actions and shaped rewards; compare them only with other training runs, not with frozen evaluation.
- The feasibility mask and scripted helpers supply substantial task knowledge; compare learned policies against the masked random baseline with the same helpers before claiming a benefit.
