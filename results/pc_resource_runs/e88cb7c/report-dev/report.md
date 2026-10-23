# Natural-resource run summary

**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION**

| Group | Attempted | Completed | Invalid | Successes | Success rate | Wilson 95% | Deaths | Truncated | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_evaluation / q_learning / checkpoint 5dec49476ac1 / max_steps 60 | 3 | 3 | 0 | 2 | 0.67 | 0.21–0.94 | 0 | 1 | 29 / 13.5 | 34.5 / 32.5 | 4 |
| frozen_evaluation / random / masked_random / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 13.3 / 13.3 | 24.6 / 24.6 | 13 |
| online_training / q_learning / max_steps 50 | 1 | 1 | 0 | 1 | 1.00 | 0.21–1.00 | 0 | 0 | 11 / 11 | 16.5 / 16.5 | 0 |
| online_training / q_learning / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 11.7 / 11.7 | 17.1 / 17.1 | 3 |
| online_training / sarsa / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 11 / 11 | 17.0 / 17.0 | 0 |

## frozen_evaluation / q_learning / checkpoint 5dec49476ac1 / max_steps 60

Runs: pc-q-frozen-dev. World seeds: 1, 2, 3.
Evaluated checkpoint (policy.json SHA-256): 5dec49476ac17a3260ac5ebb94292734b8efb60abc181f829f3cee5e39cff302.
Mean frozen-policy shaped return: 6.104.
Leaf clearing: 4 canopy layers, 0 wall blocks, 1 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_unreachable ×61, log_collected ×6, crafted_oak_planks ×5, cleared_2_leaf_layers ×2, descended_canopy ×2, log_already_reachable ×2, crafted_stick ×2, crafted_crafting_table ×2.

## frozen_evaluation / random / masked_random / max_steps 60

Runs: pc-random-dev. World seeds: 1, 2, 3.
Mean frozen-policy shaped return: 9.469.
Leaf clearing: 13 canopy layers, 0 wall blocks, 3 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_oak_planks ×5, crafted_birch_planks ×4, log_unreachable ×4, crafted_crafting_table ×3, crafted_stick ×3, table_placed ×3, crafted_wooden_pickaxe ×3.

## online_training / q_learning / max_steps 50

Runs: pc-canopy-seed1. World seeds: 1.
Mean exploratory shaped training return: 9.530.
Leaf clearing: 0 canopy layers, 0 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×3, crafted_birch_planks ×3, log_reached ×1, crafted_stick ×1, crafted_crafting_table ×1, table_placed ×1, crafted_wooden_pickaxe ×1.

## online_training / q_learning / max_steps 60

Runs: pc-q-dev. World seeds: 1, 2, 3.
Mean exploratory shaped training return: 9.505.
Leaf clearing: 3 canopy layers, 0 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_oak_planks ×6, log_reached ×3, crafted_stick ×3, crafted_crafting_table ×3, table_placed ×3, crafted_wooden_pickaxe ×3, crafted_birch_planks ×3.

## online_training / sarsa / max_steps 60

Runs: pc-sarsa-dev. World seeds: 1, 2, 3.
Mean exploratory shaped training return: 9.516.
Leaf clearing: 0 canopy layers, 0 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_birch_planks ×5, crafted_oak_planks ×4, crafted_stick ×3, crafted_crafting_table ×3, table_placed ×3, crafted_wooden_pickaxe ×3, log_reached ×2.

## Episodes

| Run | Episode | Seed | Status | Success | Death | Steps | Task s | Failure cause | Final event |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pc-canopy-seed1 | 1 | 1 | completed | True | False | 11 | 16.5 | success | crafted_wooden_pickaxe |
| pc-q-dev | 1 | 1 | completed | True | False | 12 | 16.0 | success | crafted_wooden_pickaxe |
| pc-q-dev | 2 | 2 | completed | True | False | 11 | 15.4 | success | crafted_wooden_pickaxe |
| pc-q-dev | 3 | 3 | completed | True | False | 12 | 19.8 | success | crafted_wooden_pickaxe |
| pc-sarsa-dev | 1 | 1 | completed | True | False | 11 | 15.1 | success | crafted_wooden_pickaxe |
| pc-sarsa-dev | 2 | 2 | completed | True | False | 10 | 16.6 | success | crafted_wooden_pickaxe |
| pc-sarsa-dev | 3 | 3 | completed | True | False | 12 | 19.5 | success | crafted_wooden_pickaxe |
| pc-random-dev | 1 | 1 | completed | True | False | 12 | 17.5 | success | crafted_wooden_pickaxe |
| pc-random-dev | 2 | 2 | completed | True | False | 13 | 27.1 | success | crafted_wooden_pickaxe |
| pc-random-dev | 3 | 3 | completed | True | False | 15 | 29.3 | success | crafted_wooden_pickaxe |
| pc-q-frozen-dev | 1 | 1 | completed | True | False | 12 | 17.0 | success | crafted_wooden_pickaxe |
| pc-q-frozen-dev | 2 | 2 | completed | True | False | 15 | 48.1 | success | crafted_wooden_pickaxe |
| pc-q-frozen-dev | 3 | 3 | completed | False | False | 60 | 38.3 | no_wood_collected | log_unreachable |

## Reading these numbers

- Success means a wooden pickaxe was crafted from naturally gathered wood and confirmed held; it is not full Survival or an Ender Dragon result.
- Rates are computed over completed episodes only; invalid or interrupted episodes are counted separately and never treated as policy failures or successes.
- The Wilson 95% interval describes these episodes for one checkpoint or baseline under one step limit; it does not measure variation across independently trained agents.
- Returns from online training include exploratory actions and shaped rewards; compare them only with other training runs, not with frozen evaluation.
- The feasibility mask and scripted helpers supply substantial task knowledge; compare learned policies against the masked random baseline with the same helpers before claiming a benefit.
