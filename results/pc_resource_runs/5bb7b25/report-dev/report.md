# Natural-resource run summary

**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION**

| Group | Attempted | Completed | Invalid | Successes | Success rate | Wilson 95% | Deaths | Truncated | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_evaluation / q_learning / checkpoint 9fc5c3696093 / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 10.3 / 10.3 | 16.8 / 16.8 | 2 |
| frozen_evaluation / random / masked_random / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 11.7 / 11.7 | 19.3 / 19.3 | 2 |
| frozen_evaluation / sarsa / checkpoint aabbc34453d5 / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 10.3 / 10.3 | 19.0 / 19.0 | 2 |
| online_training / q_learning / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 10.7 / 10.7 | 19.1 / 19.1 | 4 |
| online_training / sarsa / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 10.3 / 10.3 | 17.9 / 17.9 | 4 |

## frozen_evaluation / q_learning / checkpoint 9fc5c3696093 / max_steps 60

Runs: pc3-q-frozen-dev. World seeds: 1, 2, 3.
Evaluated checkpoint (policy.json SHA-256): 9fc5c3696093710481010d13f81ea78eecf9f0c93a92055366e8621d0560669d.
Mean frozen-policy shaped return: 9.537.
Leaf clearing: 2 canopy layers, 0 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_oak_planks ×6, crafted_stick ×3, crafted_crafting_table ×3, table_placed ×3, crafted_wooden_pickaxe ×3, crafted_birch_planks ×3, cleared_2_leaf_layers ×1.

## frozen_evaluation / random / masked_random / max_steps 60

Runs: pc3-random-dev. World seeds: 1, 2, 3.
Mean frozen-policy shaped return: 9.513.
Leaf clearing: 2 canopy layers, 0 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_oak_planks ×4, crafted_crafting_table ×3, crafted_stick ×3, table_placed ×3, crafted_wooden_pickaxe ×3, crafted_dark_oak_planks ×3, crafted_birch_planks ×2.

## frozen_evaluation / sarsa / checkpoint aabbc34453d5 / max_steps 60

Runs: pc3-sarsa-frozen-dev. World seeds: 1, 2, 3.
Evaluated checkpoint (policy.json SHA-256): aabbc34453d5b098a6140911dba65700fc50a1c46a913b5eaf6054b38b4c6b3f.
Mean frozen-policy shaped return: 9.537.
Leaf clearing: 2 canopy layers, 0 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_birch_planks ×7, crafted_stick ×3, crafted_crafting_table ×3, table_placed ×3, crafted_wooden_pickaxe ×3, crafted_oak_planks ×2, cleared_2_leaf_layers ×1.

## online_training / q_learning / max_steps 60

Runs: pc3-q-dev. World seeds: 1, 2, 3.
Mean exploratory shaped training return: 9.533.
Leaf clearing: 4 canopy layers, 0 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_oak_planks ×5, crafted_birch_planks ×4, crafted_stick ×3, crafted_crafting_table ×3, table_placed ×3, crafted_wooden_pickaxe ×3, cleared_2_leaf_layers ×2.

## online_training / sarsa / max_steps 60

Runs: pc3-sarsa-dev. World seeds: 1, 2, 3.
Mean exploratory shaped training return: 9.536.
Leaf clearing: 4 canopy layers, 0 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_birch_planks ×5, crafted_oak_planks ×4, crafted_stick ×3, crafted_crafting_table ×3, table_placed ×3, crafted_wooden_pickaxe ×3, cleared_4_leaf_layers ×1.

## Episodes

| Run | Episode | Seed | Status | Success | Death | Steps | Task s | Failure cause | Final event |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pc3-q-dev | 1 | 1 | completed | True | False | 11 | 17.5 | success | crafted_wooden_pickaxe |
| pc3-q-dev | 2 | 2 | completed | True | False | 11 | 22.5 | success | crafted_wooden_pickaxe |
| pc3-q-dev | 3 | 3 | completed | True | False | 10 | 17.3 | success | crafted_wooden_pickaxe |
| pc3-sarsa-dev | 1 | 1 | completed | True | False | 10 | 15.9 | success | crafted_wooden_pickaxe |
| pc3-sarsa-dev | 2 | 2 | completed | True | False | 11 | 20.8 | success | crafted_wooden_pickaxe |
| pc3-sarsa-dev | 3 | 3 | completed | True | False | 10 | 16.8 | success | crafted_wooden_pickaxe |
| pc3-random-dev | 1 | 1 | completed | True | False | 11 | 18.1 | success | crafted_wooden_pickaxe |
| pc3-random-dev | 2 | 2 | completed | True | False | 10 | 14.9 | success | crafted_wooden_pickaxe |
| pc3-random-dev | 3 | 3 | completed | True | False | 14 | 24.8 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-dev | 1 | 1 | completed | True | False | 10 | 17.7 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-dev | 2 | 2 | completed | True | False | 10 | 15.3 | success | crafted_wooden_pickaxe |
| pc3-q-frozen-dev | 3 | 3 | completed | True | False | 11 | 17.5 | success | crafted_wooden_pickaxe |
| pc3-sarsa-frozen-dev | 1 | 1 | completed | True | False | 10 | 15.5 | success | crafted_wooden_pickaxe |
| pc3-sarsa-frozen-dev | 2 | 2 | completed | True | False | 11 | 25.9 | success | crafted_wooden_pickaxe |
| pc3-sarsa-frozen-dev | 3 | 3 | completed | True | False | 10 | 15.5 | success | crafted_wooden_pickaxe |

## Reading these numbers

- Success means a wooden pickaxe was crafted from naturally gathered wood and confirmed held; it is not full Survival or an Ender Dragon result.
- Rates are computed over completed episodes only; invalid or interrupted episodes are counted separately and never treated as policy failures or successes.
- The Wilson 95% interval describes these episodes for one checkpoint or baseline under one step limit; it does not measure variation across independently trained agents.
- Returns from online training include exploratory actions and shaped rewards; compare them only with other training runs, not with frozen evaluation.
- The feasibility mask and scripted helpers supply substantial task knowledge; compare learned policies against the masked random baseline with the same helpers before claiming a benefit.
