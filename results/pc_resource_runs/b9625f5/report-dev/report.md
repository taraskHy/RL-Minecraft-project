# Natural-resource run summary

**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION**

| Group | Attempted | Completed | Invalid | Successes | Success rate | Wilson 95% | Deaths | Truncated | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_evaluation / q_learning / checkpoint a1f5984de1db / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 10.3 / 10.3 | 22.6 / 22.6 | 0 |
| frozen_evaluation / random / masked_random / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 10.7 / 10.7 | 17.4 / 17.4 | 6 |
| frozen_evaluation / sarsa / checkpoint c3b80be8eefd / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 11.7 / 11.7 | 19.9 / 19.9 | 6 |
| online_training / q_learning / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 11 / 11 | 17.9 / 17.9 | 2 |
| online_training / sarsa / max_steps 60 | 3 | 3 | 0 | 3 | 1.00 | 0.44–1.00 | 0 | 0 | 11 / 11 | 17.9 / 17.9 | 2 |

## frozen_evaluation / q_learning / checkpoint a1f5984de1db / max_steps 60

Runs: pc2-q-frozen-dev. World seeds: 1, 2, 3.
Evaluated checkpoint (policy.json SHA-256): a1f5984de1db88df6b8ec495256bdd02707a502ed1dcf750a02f17a4fa12bce5.
Mean frozen-policy shaped return: 9.528.
Leaf clearing: 0 canopy layers, 0 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×8, crafted_birch_planks ×6, crafted_oak_planks ×3, crafted_stick ×3, crafted_crafting_table ×3, table_placed ×3, crafted_wooden_pickaxe ×3, log_broken_without_pickup ×2.

## frozen_evaluation / random / masked_random / max_steps 60

Runs: pc2-random-dev. World seeds: 1, 2, 3.
Mean frozen-policy shaped return: 9.526.
Leaf clearing: 6 canopy layers, 0 wall blocks, 3 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_birch_planks ×6, crafted_oak_planks ×3, crafted_stick ×3, crafted_crafting_table ×3, table_placed ×3, crafted_wooden_pickaxe ×3, descended_canopy ×2.

## frozen_evaluation / sarsa / checkpoint c3b80be8eefd / max_steps 60

Runs: pc2-sarsa-frozen-dev. World seeds: 1, 2, 3.
Evaluated checkpoint (policy.json SHA-256): c3b80be8eefd1ebcef787a7a72423c30d56533f7bb58949fb94d35001ddd0f20.
Mean frozen-policy shaped return: 9.523.
Leaf clearing: 6 canopy layers, 0 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_birch_planks ×5, crafted_oak_planks ×4, crafted_stick ×3, crafted_crafting_table ×3, table_placed ×3, crafted_wooden_pickaxe ×3, log_reached ×2.

## online_training / q_learning / max_steps 60

Runs: pc2-q-dev. World seeds: 1, 2, 3.
Mean exploratory shaped training return: 9.512.
Leaf clearing: 2 canopy layers, 0 wall blocks, 1 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_oak_planks ×5, crafted_birch_planks ×4, crafted_stick ×3, crafted_crafting_table ×3, table_placed ×3, crafted_wooden_pickaxe ×3, log_reached ×2.

## online_training / sarsa / max_steps 60

Runs: pc2-sarsa-dev. World seeds: 1, 2, 3.
Mean exploratory shaped training return: 9.530.
Leaf clearing: 2 canopy layers, 0 wall blocks, 0 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×9, crafted_oak_planks ×6, crafted_stick ×3, crafted_crafting_table ×3, table_placed ×3, crafted_wooden_pickaxe ×3, crafted_birch_planks ×3, log_reached ×2.

## Episodes

| Run | Episode | Seed | Status | Success | Death | Steps | Task s | Failure cause | Final event |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pc2-q-dev | 1 | 1 | completed | True | False | 11 | 16.2 | success | crafted_wooden_pickaxe |
| pc2-q-dev | 2 | 2 | completed | True | False | 11 | 16.9 | success | crafted_wooden_pickaxe |
| pc2-q-dev | 3 | 3 | completed | True | False | 11 | 20.5 | success | crafted_wooden_pickaxe |
| pc2-sarsa-dev | 1 | 1 | completed | True | False | 11 | 19.5 | success | crafted_wooden_pickaxe |
| pc2-sarsa-dev | 2 | 2 | completed | True | False | 11 | 18.7 | success | crafted_wooden_pickaxe |
| pc2-sarsa-dev | 3 | 3 | completed | True | False | 11 | 15.7 | success | crafted_wooden_pickaxe |
| pc2-random-dev | 1 | 1 | completed | True | False | 10 | 17.4 | success | crafted_wooden_pickaxe |
| pc2-random-dev | 2 | 2 | completed | True | False | 11 | 16.7 | success | crafted_wooden_pickaxe |
| pc2-random-dev | 3 | 3 | completed | True | False | 11 | 18.2 | success | crafted_wooden_pickaxe |
| pc2-q-frozen-dev | 1 | 1 | completed | True | False | 10 | 13.2 | success | crafted_wooden_pickaxe |
| pc2-q-frozen-dev | 2 | 2 | completed | True | False | 10 | 17.1 | success | crafted_wooden_pickaxe |
| pc2-q-frozen-dev | 3 | 3 | completed | True | False | 11 | 37.4 | success | crafted_wooden_pickaxe |
| pc2-sarsa-frozen-dev | 1 | 1 | completed | True | False | 11 | 17.2 | success | crafted_wooden_pickaxe |
| pc2-sarsa-frozen-dev | 2 | 2 | completed | True | False | 11 | 21.4 | success | crafted_wooden_pickaxe |
| pc2-sarsa-frozen-dev | 3 | 3 | completed | True | False | 13 | 21.1 | success | crafted_wooden_pickaxe |

## Reading these numbers

- Success means a wooden pickaxe was crafted from naturally gathered wood and confirmed held; it is not full Survival or an Ender Dragon result.
- Rates are computed over completed episodes only; invalid or interrupted episodes are counted separately and never treated as policy failures or successes.
- The Wilson 95% interval describes these episodes for one checkpoint or baseline under one step limit; it does not measure variation across independently trained agents.
- Returns from online training include exploratory actions and shaped rewards; compare them only with other training runs, not with frozen evaluation.
- The feasibility mask and scripted helpers supply substantial task knowledge; compare learned policies against the masked random baseline with the same helpers before claiming a benefit.
