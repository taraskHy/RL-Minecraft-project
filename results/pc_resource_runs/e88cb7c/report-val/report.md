# Natural-resource run summary

**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION**

| Group | Attempted | Completed | Invalid | Successes | Success rate | Wilson 95% | Deaths | Truncated | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_evaluation / q_learning / checkpoint 5dec49476ac1 / max_steps 60 | 10 | 10 | 0 | 10 | 1.00 | 0.72–1.00 | 0 | 0 | 18.2 / 18.2 | 55.5 / 55.5 | 5 |
| frozen_evaluation / random / masked_random / max_steps 60 | 10 | 10 | 0 | 8 | 0.80 | 0.49–0.94 | 0 | 2 | 28.1 / 20.1 | 93.5 / 32.5 | 7 |
| frozen_evaluation / sarsa / checkpoint a2c7f98456da / max_steps 60 | 10 | 10 | 0 | 8 | 0.80 | 0.49–0.94 | 0 | 2 | 23.9 / 14.9 | 62.8 / 29.9 | 7 |

## frozen_evaluation / q_learning / checkpoint 5dec49476ac1 / max_steps 60

Runs: pc-q-frozen-val. World seeds: 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009.
Evaluated checkpoint (policy.json SHA-256): 5dec49476ac17a3260ac5ebb94292734b8efb60abc181f829f3cee5e39cff302.
Mean frozen-policy shaped return: 9.376.
Leaf clearing: 5 canopy layers, 0 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: explored ×47, log_collected ×30, log_path_blocked ×17, crafted_oak_planks ×15, log_reached ×10, crafted_stick ×10, crafted_crafting_table ×10, table_placed ×10.

## frozen_evaluation / random / masked_random / max_steps 60

Runs: pc-random-val. World seeds: 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009.
Mean frozen-policy shaped return: 7.056.
Leaf clearing: 7 canopy layers, 0 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×1, pickaxe_not_crafted ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: explored ×115, no_placement_surface ×47, log_collected ×26, crafted_oak_planks ×15, table_reached ×12, crafted_crafting_table ×9, crafted_stick ×9, table_placed ×9.

## frozen_evaluation / sarsa / checkpoint a2c7f98456da / max_steps 60

Runs: pc-sarsa-frozen-val. World seeds: 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009.
Evaluated checkpoint (policy.json SHA-256): a2c7f98456da8ff407e77dab00804e5328280381300cc8ae3cbdfd807fa42d9c.
Mean frozen-policy shaped return: 7.462.
Leaf clearing: 7 canopy layers, 0 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×2.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: explored ×77, exploration_blocked ×60, log_collected ×24, crafted_oak_planks ×15, log_path_blocked ×10, crafted_stick ×8, crafted_crafting_table ×8, table_placed ×8.

## Episodes

| Run | Episode | Seed | Status | Success | Death | Steps | Task s | Failure cause | Final event |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pc-q-frozen-val | 1 | 10000 | completed | True | False | 11 | 38.6 | success | crafted_wooden_pickaxe |
| pc-q-frozen-val | 2 | 10001 | completed | True | False | 11 | 19.0 | success | crafted_wooden_pickaxe |
| pc-q-frozen-val | 3 | 10002 | completed | True | False | 10 | 13.7 | success | crafted_wooden_pickaxe |
| pc-q-frozen-val | 4 | 10003 | completed | True | False | 12 | 16.5 | success | crafted_wooden_pickaxe |
| pc-q-frozen-val | 5 | 10004 | completed | True | False | 40 | 174.1 | success | crafted_wooden_pickaxe |
| pc-q-frozen-val | 6 | 10005 | completed | True | False | 15 | 62.7 | success | crafted_wooden_pickaxe |
| pc-q-frozen-val | 7 | 10006 | completed | True | False | 11 | 40.1 | success | crafted_wooden_pickaxe |
| pc-q-frozen-val | 8 | 10007 | completed | True | False | 11 | 19.4 | success | crafted_wooden_pickaxe |
| pc-q-frozen-val | 9 | 10008 | completed | True | False | 35 | 71.4 | success | crafted_wooden_pickaxe |
| pc-q-frozen-val | 10 | 10009 | completed | True | False | 26 | 99.5 | success | crafted_wooden_pickaxe |
| pc-sarsa-frozen-val | 1 | 10000 | completed | True | False | 11 | 14.0 | success | crafted_wooden_pickaxe |
| pc-sarsa-frozen-val | 2 | 10001 | completed | True | False | 12 | 17.6 | success | crafted_wooden_pickaxe |
| pc-sarsa-frozen-val | 3 | 10002 | completed | True | False | 11 | 29.8 | success | crafted_wooden_pickaxe |
| pc-sarsa-frozen-val | 4 | 10003 | completed | True | False | 11 | 18.9 | success | crafted_wooden_pickaxe |
| pc-sarsa-frozen-val | 5 | 10004 | completed | False | False | 60 | 368.7 | no_wood_collected | explored |
| pc-sarsa-frozen-val | 6 | 10005 | completed | True | False | 11 | 17.8 | success | crafted_wooden_pickaxe |
| pc-sarsa-frozen-val | 7 | 10006 | completed | False | False | 60 | 20.5 | no_wood_collected | exploration_blocked |
| pc-sarsa-frozen-val | 8 | 10007 | completed | True | False | 13 | 16.5 | success | crafted_wooden_pickaxe |
| pc-sarsa-frozen-val | 9 | 10008 | completed | True | False | 34 | 80.6 | success | crafted_wooden_pickaxe |
| pc-sarsa-frozen-val | 10 | 10009 | completed | True | False | 16 | 43.8 | success | crafted_wooden_pickaxe |
| pc-random-val | 1 | 10000 | completed | True | False | 58 | 24.0 | success | crafted_wooden_pickaxe |
| pc-random-val | 2 | 10001 | completed | True | False | 11 | 26.4 | success | crafted_wooden_pickaxe |
| pc-random-val | 3 | 10002 | completed | True | False | 11 | 16.2 | success | crafted_wooden_pickaxe |
| pc-random-val | 4 | 10003 | completed | True | False | 11 | 21.7 | success | crafted_wooden_pickaxe |
| pc-random-val | 5 | 10004 | completed | False | False | 60 | 404.6 | no_wood_collected | explored |
| pc-random-val | 6 | 10005 | completed | True | False | 17 | 27.8 | success | crafted_wooden_pickaxe |
| pc-random-val | 7 | 10006 | completed | True | False | 11 | 16.9 | success | crafted_wooden_pickaxe |
| pc-random-val | 8 | 10007 | completed | True | False | 11 | 22.1 | success | crafted_wooden_pickaxe |
| pc-random-val | 9 | 10008 | completed | True | False | 31 | 105.0 | success | crafted_wooden_pickaxe |
| pc-random-val | 10 | 10009 | completed | False | False | 60 | 270.2 | pickaxe_not_crafted | explored |

## Reading these numbers

- Success means a wooden pickaxe was crafted from naturally gathered wood and confirmed held; it is not full Survival or an Ender Dragon result.
- Rates are computed over completed episodes only; invalid or interrupted episodes are counted separately and never treated as policy failures or successes.
- The Wilson 95% interval describes these episodes for one checkpoint or baseline under one step limit; it does not measure variation across independently trained agents.
- Returns from online training include exploratory actions and shaped rewards; compare them only with other training runs, not with frozen evaluation.
- The feasibility mask and scripted helpers supply substantial task knowledge; compare learned policies against the masked random baseline with the same helpers before claiming a benefit.
