# Natural-resource run summary

**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION**

| Group | Attempted | Completed | Invalid | Successes | Success rate | Wilson 95% | Deaths | Truncated | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked / online_training / q_learning / max_steps 60 | 20 | 20 | 0 | 17 | 0.85 | 0.64–0.95 | 0 | 3 | 19.4 / 12.3 | 80.1 / 22.2 | 21 |
| masked / online_training / sarsa / max_steps 60 | 20 | 20 | 0 | 18 | 0.90 | 0.70–0.97 | 1 | 1 | 13.8 / 11.8 | 46.5 / 22.6 | 16 |
| unmasked / online_training / q_learning / max_steps 60 | 50 | 50 | 0 | 36 | 0.72 | 0.58–0.83 | 0 | 14 | 34.0 / 23.9 | 55.0 / 38.9 | 40 |
| unmasked / online_training / sarsa / max_steps 60 | 50 | 50 | 0 | 41 | 0.82 | 0.69–0.90 | 0 | 9 | 28.9 / 22.0 | 45.9 / 36.5 | 35 |
| unmasked_history / online_training / q_learning / max_steps 60 | 50 | 50 | 0 | 29 | 0.58 | 0.44–0.71 | 0 | 21 | 39.9 / 25.4 | 52.6 / 28.5 | 47 |
| unmasked_history / online_training / sarsa / max_steps 60 | 50 | 49 | 1 | 37 | 0.76 | 0.62–0.85 | 0 | 12 | 36.5 / 28.9 | 48.2 / 39.3 | 53 |

## Decisions

How much each decision was left to the policy: the share with a single allowed action, the share of choices the disclosed mask would have forbidden, and the share that changed none of the nine resource_v1 fields.

| Group | Decisions | Single allowed action | Choices outside the mask | No progress |
| --- | --- | --- | --- | --- |
| masked / online_training / q_learning / max_steps 60 | 389 | 74% | 0% | 49% |
| masked / online_training / sarsa / max_steps 60 | 277 | 61% | 0% | 29% |
| unmasked / online_training / q_learning / max_steps 60 | 1701 | 0% | 54% | 60% |
| unmasked / online_training / sarsa / max_steps 60 | 1444 | 0% | 53% | 54% |
| unmasked_history / online_training / q_learning / max_steps 60 | 1997 | 0% | 60% | 67% |
| unmasked_history / online_training / sarsa / max_steps 60 | 1811 | 0% | 60% | 65% |

## masked / online_training / q_learning / max_steps 60

Runs: mx-masked-q-train. World seeds: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119.
Mean exploratory shaped training return: 7.950.
Leaf clearing: 21 canopy layers, 4 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×2, pickaxe_not_crafted ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: explored ×94, log_collected ×54, table_path_blocked ×40, explored_locally ×33, crafted_oak_planks ×24, canopy_descent_refused:hazard_below ×22, crafted_stick ×18, crafted_crafting_table ×18.

## masked / online_training / sarsa / max_steps 60

Runs: mx-masked-sarsa-train. World seeds: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119.
Mean exploratory shaped training return: 8.012.
Leaf clearing: 16 canopy layers, 3 wall blocks, 1 fall damage in total.
Failure causes (completed episodes only): death ×1, no_wood_collected ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×54, explored ×50, crafted_oak_planks ×36, crafted_stick ×18, crafted_crafting_table ×18, table_placed ×18, crafted_wooden_pickaxe ×18, explored_locally ×16.

## unmasked / online_training / q_learning / max_steps 60

Runs: mx-unmasked-q-train. World seeds: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149.
Mean exploratory shaped training return: 6.090.
Leaf clearing: 40 canopy layers, 7 wall blocks, 8 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×5, pickaxe_not_crafted ×8, table_not_placed ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: missing_materials ×280, log_collected ×195, explored ×127, crafted_oak_planks ×102, crafted_stick ×92, log_already_reachable ×87, not_at_own_table ×85, crafted_crafting_table ×74.

## unmasked / online_training / sarsa / max_steps 60

Runs: mx-unmasked-sarsa-train. World seeds: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149.
Mean exploratory shaped training return: 7.195.
Leaf clearing: 35 canopy layers, 0 wall blocks, 11 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×3, pickaxe_not_crafted ×6.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: missing_materials ×225, log_collected ×189, explored ×103, log_already_reachable ×93, crafted_stick ×88, crafted_oak_planks ×82, not_at_own_table ×77, crafted_crafting_table ×65.

## unmasked_history / online_training / q_learning / max_steps 60

Runs: mx-history-q-train. World seeds: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149.
Mean exploratory shaped training return: 4.753.
Leaf clearing: 47 canopy layers, 8 wall blocks, 5 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×7, pickaxe_not_crafted ×13, table_not_placed ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: missing_materials ×450, log_collected ×182, explored ×117, no_own_table ×112, not_at_own_table ×112, log_already_reachable ×98, crafted_stick ×87, table_reached ×77.

## unmasked_history / online_training / sarsa / max_steps 60

Runs: mx-history-sarsa-train, mx-history-sarsa-train-part1. World seeds: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149.
Mean exploratory shaped training return: 6.406.
Leaf clearing: 53 canopy layers, 3 wall blocks, 6 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×6, pickaxe_not_crafted ×5, table_not_placed ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): invalid ×1.
Most frequent step events: missing_materials ×390, log_collected ×174, explored ×125, not_at_own_table ×111, log_already_reachable ×106, crafted_stick ×93, missing_table ×92, no_own_table ×89.

## Episodes

| Run | Episode | Seed | Status | Success | Death | Steps | Task s | Failure cause | Final event |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mx-masked-q-train | 1 | 100 | completed | True | False | 11 | 17.1 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 2 | 101 | completed | False | False | 60 | 428.8 | no_wood_collected | explored |
| mx-masked-q-train | 3 | 102 | completed | True | False | 10 | 17.5 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 4 | 103 | completed | True | False | 10 | 22.4 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 5 | 104 | completed | True | False | 11 | 18.3 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 6 | 105 | completed | True | False | 10 | 16.3 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 7 | 106 | completed | True | False | 11 | 17.6 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 8 | 107 | completed | True | False | 10 | 17.8 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 9 | 108 | completed | True | False | 12 | 38.9 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 10 | 109 | completed | True | False | 15 | 33.5 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 11 | 110 | completed | True | False | 12 | 19.4 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 12 | 111 | completed | False | False | 60 | 129.7 | pickaxe_not_crafted | table_path_blocked |
| mx-masked-q-train | 13 | 112 | completed | True | False | 10 | 13.6 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 14 | 113 | completed | True | False | 10 | 13.3 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 15 | 114 | completed | True | False | 11 | 29.1 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 16 | 115 | completed | True | False | 11 | 19.5 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 17 | 116 | completed | True | False | 12 | 29.8 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 18 | 117 | completed | True | False | 33 | 34.1 | success | crafted_wooden_pickaxe |
| mx-masked-q-train | 19 | 118 | completed | False | False | 60 | 667.3 | no_wood_collected | explored |
| mx-masked-q-train | 20 | 119 | completed | True | False | 10 | 18.6 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 1 | 100 | completed | True | False | 12 | 22.3 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 2 | 101 | completed | False | False | 60 | 477.1 | no_wood_collected | explored_locally |
| mx-masked-sarsa-train | 3 | 102 | completed | True | False | 12 | 17.2 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 4 | 103 | completed | True | False | 11 | 14.8 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 5 | 104 | completed | True | False | 11 | 18.8 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 6 | 105 | completed | True | False | 10 | 14.9 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 7 | 106 | completed | True | False | 10 | 13.2 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 8 | 107 | completed | True | False | 10 | 17.8 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 9 | 108 | completed | True | False | 11 | 15.8 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 10 | 109 | completed | True | False | 11 | 23.5 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 11 | 110 | completed | True | False | 11 | 16.1 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 12 | 111 | completed | True | False | 22 | 85.2 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 13 | 112 | completed | True | False | 11 | 15.9 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 14 | 113 | completed | True | False | 11 | 15.2 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 15 | 114 | completed | True | False | 12 | 16.1 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 16 | 115 | completed | True | False | 11 | 22.5 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 17 | 116 | completed | True | False | 14 | 31.9 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 18 | 117 | completed | True | False | 11 | 19.4 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-train | 19 | 118 | completed | False | True | 4 | 46.1 | death | exploration_time_budget;exploration_blocked |
| mx-masked-sarsa-train | 20 | 119 | completed | True | False | 12 | 25.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 1 | 100 | completed | False | False | 60 | 60.9 | table_not_placed | log_already_reachable |
| mx-unmasked-q-train | 2 | 101 | completed | False | False | 60 | 52.0 | no_wood_collected | missing_materials |
| mx-unmasked-q-train | 3 | 102 | completed | False | False | 60 | 32.9 | no_wood_collected | not_at_own_table |
| mx-unmasked-q-train | 4 | 103 | completed | True | False | 35 | 45.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 5 | 104 | completed | False | False | 60 | 51.9 | pickaxe_not_crafted | table_reached |
| mx-unmasked-q-train | 6 | 105 | completed | False | False | 60 | 42.5 | pickaxe_not_crafted | log_collected |
| mx-unmasked-q-train | 7 | 106 | completed | True | False | 30 | 24.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 8 | 107 | completed | False | False | 60 | 106.6 | pickaxe_not_crafted | table_already_placed |
| mx-unmasked-q-train | 9 | 108 | completed | True | False | 43 | 27.6 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 10 | 109 | completed | False | False | 60 | 74.2 | no_wood_collected | explored_locally |
| mx-unmasked-q-train | 11 | 110 | completed | True | False | 36 | 39.6 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 12 | 111 | completed | True | False | 14 | 20.6 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 13 | 112 | completed | True | False | 14 | 20.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 14 | 113 | completed | True | False | 36 | 41.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 15 | 114 | completed | True | False | 47 | 43.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 16 | 115 | completed | True | False | 30 | 34.4 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 17 | 116 | completed | True | False | 44 | 40.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 18 | 117 | completed | True | False | 18 | 38.3 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 19 | 118 | completed | False | False | 60 | 426.7 | no_wood_collected | not_at_own_table |
| mx-unmasked-q-train | 20 | 119 | completed | False | False | 60 | 58.5 | pickaxe_not_crafted | not_at_own_table |
| mx-unmasked-q-train | 21 | 120 | completed | True | False | 60 | 71.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 22 | 121 | completed | True | False | 13 | 17.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 23 | 122 | completed | True | False | 31 | 34.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 24 | 123 | completed | True | False | 19 | 27.6 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 25 | 124 | completed | True | False | 16 | 25.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 26 | 125 | completed | True | False | 16 | 80.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 27 | 126 | completed | False | False | 60 | 72.2 | pickaxe_not_crafted | crafted_stick |
| mx-unmasked-q-train | 28 | 127 | completed | True | False | 16 | 34.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 29 | 128 | completed | True | False | 25 | 132.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 30 | 129 | completed | True | False | 31 | 31.3 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 31 | 130 | completed | False | False | 60 | 38.8 | pickaxe_not_crafted | missing_materials |
| mx-unmasked-q-train | 32 | 131 | completed | True | False | 15 | 17.3 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 33 | 132 | completed | False | False | 60 | 25.9 | pickaxe_not_crafted | missing_materials |
| mx-unmasked-q-train | 34 | 133 | completed | True | False | 12 | 25.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 35 | 134 | completed | True | False | 32 | 85.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 36 | 135 | completed | True | False | 12 | 21.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 37 | 136 | completed | True | False | 17 | 21.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 38 | 137 | completed | True | False | 11 | 16.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 39 | 138 | completed | False | False | 60 | 190.8 | no_wood_collected | leaf_escape_refused:no_leaf_exit;exploration_blocked |
| mx-unmasked-q-train | 40 | 139 | completed | True | False | 30 | 34.3 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 41 | 140 | completed | True | False | 21 | 22.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 42 | 141 | completed | True | False | 13 | 21.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 43 | 142 | completed | True | False | 39 | 143.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 44 | 143 | completed | True | False | 13 | 18.3 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 45 | 144 | completed | True | False | 10 | 18.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 46 | 145 | completed | True | False | 12 | 24.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 47 | 146 | completed | True | False | 14 | 34.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 48 | 147 | completed | False | False | 60 | 113.5 | pickaxe_not_crafted | missing_materials |
| mx-unmasked-q-train | 49 | 148 | completed | True | False | 13 | 26.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-train | 50 | 149 | completed | True | False | 23 | 36.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 1 | 100 | completed | False | False | 60 | 45.6 | pickaxe_not_crafted | explored_locally |
| mx-unmasked-sarsa-train | 2 | 101 | completed | False | False | 60 | 28.2 | no_wood_collected | missing_materials |
| mx-unmasked-sarsa-train | 3 | 102 | completed | True | False | 24 | 42.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 4 | 103 | completed | False | False | 60 | 36.6 | pickaxe_not_crafted | missing_materials |
| mx-unmasked-sarsa-train | 5 | 104 | completed | True | False | 47 | 58.1 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 6 | 105 | completed | True | False | 30 | 32.6 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 7 | 106 | completed | True | False | 40 | 50.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 8 | 107 | completed | True | False | 56 | 38.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 9 | 108 | completed | False | False | 60 | 52.6 | pickaxe_not_crafted | log_already_reachable |
| mx-unmasked-sarsa-train | 10 | 109 | completed | True | False | 29 | 29.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 11 | 110 | completed | False | False | 60 | 74.1 | pickaxe_not_crafted | table_already_placed |
| mx-unmasked-sarsa-train | 12 | 111 | completed | True | False | 14 | 31.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 13 | 112 | completed | True | False | 32 | 25.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 14 | 113 | completed | True | False | 12 | 15.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 15 | 114 | completed | True | False | 12 | 19.1 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 16 | 115 | completed | True | False | 26 | 33.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 17 | 116 | completed | True | False | 58 | 62.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 18 | 117 | completed | True | False | 26 | 23.3 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 19 | 118 | completed | False | False | 60 | 238.4 | no_wood_collected | not_at_own_table |
| mx-unmasked-sarsa-train | 20 | 119 | completed | True | False | 42 | 30.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 21 | 120 | completed | True | False | 21 | 19.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 22 | 121 | completed | True | False | 17 | 20.3 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 23 | 122 | completed | True | False | 16 | 18.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 24 | 123 | completed | True | False | 20 | 19.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 25 | 124 | completed | False | False | 60 | 60.4 | pickaxe_not_crafted | log_already_reachable |
| mx-unmasked-sarsa-train | 26 | 125 | completed | True | False | 21 | 24.6 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 27 | 126 | completed | True | False | 16 | 22.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 28 | 127 | completed | True | False | 12 | 20.3 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 29 | 128 | completed | True | False | 35 | 252.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 30 | 129 | completed | True | False | 18 | 21.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 31 | 130 | completed | True | False | 17 | 54.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 32 | 131 | completed | True | False | 15 | 15.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 33 | 132 | completed | True | False | 11 | 16.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 34 | 133 | completed | True | False | 21 | 24.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 35 | 134 | completed | True | False | 37 | 147.6 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 36 | 135 | completed | True | False | 12 | 24.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 37 | 136 | completed | True | False | 10 | 17.4 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 38 | 137 | completed | True | False | 19 | 18.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 39 | 138 | completed | False | False | 60 | 144.8 | no_wood_collected | not_at_own_table |
| mx-unmasked-sarsa-train | 40 | 139 | completed | True | False | 13 | 21.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 41 | 140 | completed | True | False | 16 | 29.1 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 42 | 141 | completed | True | False | 14 | 21.4 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 43 | 142 | completed | True | False | 19 | 60.6 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 44 | 143 | completed | True | False | 12 | 18.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 45 | 144 | completed | True | False | 12 | 33.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 46 | 145 | completed | True | False | 13 | 18.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 47 | 146 | completed | True | False | 12 | 20.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 48 | 147 | completed | False | False | 60 | 121.3 | pickaxe_not_crafted | table_already_placed |
| mx-unmasked-sarsa-train | 49 | 148 | completed | True | False | 14 | 22.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-train | 50 | 149 | completed | True | False | 13 | 15.6 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 1 | 100 | completed | False | False | 60 | 40.8 | pickaxe_not_crafted | log_collected |
| mx-history-q-train | 2 | 101 | completed | False | False | 60 | 33.2 | no_wood_collected | no_own_table |
| mx-history-q-train | 3 | 102 | completed | True | False | 48 | 40.3 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 4 | 103 | completed | True | False | 26 | 19.2 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 5 | 104 | completed | False | False | 60 | 66.5 | pickaxe_not_crafted | crafted_birch_planks |
| mx-history-q-train | 6 | 105 | completed | True | False | 34 | 34.2 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 7 | 106 | completed | False | False | 60 | 44.2 | pickaxe_not_crafted | table_already_placed |
| mx-history-q-train | 8 | 107 | completed | True | False | 29 | 18.2 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 9 | 108 | completed | True | False | 29 | 27.9 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 10 | 109 | completed | False | False | 60 | 94.4 | no_wood_collected | no_own_table |
| mx-history-q-train | 11 | 110 | completed | False | False | 60 | 123.6 | pickaxe_not_crafted | missing_materials |
| mx-history-q-train | 12 | 111 | completed | True | False | 15 | 20.8 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 13 | 112 | completed | True | False | 31 | 24.1 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 14 | 113 | completed | True | False | 14 | 20.7 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 15 | 114 | completed | True | False | 33 | 42.7 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 16 | 115 | completed | False | False | 60 | 42.8 | pickaxe_not_crafted | log_reached |
| mx-history-q-train | 17 | 116 | completed | True | False | 27 | 30.1 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 18 | 117 | completed | False | False | 60 | 28.5 | no_wood_collected | missing_materials |
| mx-history-q-train | 19 | 118 | completed | False | False | 60 | 68.2 | no_wood_collected | missing_materials |
| mx-history-q-train | 20 | 119 | completed | True | False | 29 | 25.2 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 21 | 120 | completed | True | False | 19 | 30.6 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 22 | 121 | completed | True | False | 29 | 34.5 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 23 | 122 | completed | False | False | 60 | 55.5 | pickaxe_not_crafted | missing_materials |
| mx-history-q-train | 24 | 123 | completed | True | False | 24 | 24.8 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 25 | 124 | completed | False | False | 60 | 95.5 | pickaxe_not_crafted | log_reached |
| mx-history-q-train | 26 | 125 | completed | True | False | 20 | 22.2 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 27 | 126 | completed | False | False | 60 | 70.0 | pickaxe_not_crafted | table_reached |
| mx-history-q-train | 28 | 127 | completed | True | False | 55 | 58.8 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 29 | 128 | completed | False | False | 60 | 198.9 | pickaxe_not_crafted | table_already_placed |
| mx-history-q-train | 30 | 129 | completed | True | False | 17 | 28.8 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 31 | 130 | completed | True | False | 21 | 25.7 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 32 | 131 | completed | True | False | 25 | 25.5 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 33 | 132 | completed | False | False | 60 | 32.1 | no_wood_collected | no_own_table |
| mx-history-q-train | 34 | 133 | completed | False | False | 60 | 53.8 | pickaxe_not_crafted | log_already_reachable |
| mx-history-q-train | 35 | 134 | completed | False | False | 60 | 74.6 | table_not_placed | missing_materials |
| mx-history-q-train | 36 | 135 | completed | True | False | 14 | 16.1 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 37 | 136 | completed | False | False | 60 | 40.8 | pickaxe_not_crafted | missing_materials |
| mx-history-q-train | 38 | 137 | completed | False | False | 60 | 74.5 | pickaxe_not_crafted | missing_materials |
| mx-history-q-train | 39 | 138 | completed | False | False | 60 | 385.0 | no_wood_collected | explored |
| mx-history-q-train | 40 | 139 | completed | True | False | 14 | 20.6 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 41 | 140 | completed | True | False | 18 | 24.0 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 42 | 141 | completed | True | False | 29 | 30.4 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 43 | 142 | completed | False | False | 60 | 104.8 | no_wood_collected | missing_materials |
| mx-history-q-train | 44 | 143 | completed | True | False | 29 | 28.3 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 45 | 144 | completed | False | False | 60 | 73.4 | pickaxe_not_crafted | missing_materials |
| mx-history-q-train | 46 | 145 | completed | True | False | 15 | 23.0 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 47 | 146 | completed | True | False | 15 | 17.4 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 48 | 147 | completed | True | False | 32 | 40.8 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 49 | 148 | completed | True | False | 22 | 22.7 | success | crafted_wooden_pickaxe |
| mx-history-q-train | 50 | 149 | completed | True | False | 24 | 49.0 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 1 | 100 | completed | False | False | 60 | 40.0 | table_not_placed | log_collected |
| mx-history-sarsa-train | 2 | 101 | completed | False | False | 60 | 36.2 | no_wood_collected | not_at_own_table |
| mx-history-sarsa-train | 3 | 102 | completed | True | False | 27 | 22.3 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 4 | 103 | completed | True | False | 31 | 24.0 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 5 | 104 | completed | True | False | 40 | 28.3 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 6 | 105 | completed | True | False | 40 | 43.6 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 7 | 106 | completed | True | False | 47 | 45.8 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 8 | 107 | completed | True | False | 52 | 44.4 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 9 | 108 | completed | False | False | 60 | 26.2 | pickaxe_not_crafted | missing_materials |
| mx-history-sarsa-train | 10 | 109 | completed | False | False | 60 | 33.9 | pickaxe_not_crafted | missing_materials |
| mx-history-sarsa-train | 11 | 110 | completed | False | False | 60 | 35.2 | pickaxe_not_crafted | log_already_reachable |
| mx-history-sarsa-train | 12 | 111 | completed | True | False | 60 | 102.1 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 13 | 112 | completed | False | False | 60 | 52.3 | no_wood_collected | missing_materials |
| mx-history-sarsa-train | 14 | 113 | completed | True | False | 14 | 14.7 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 15 | 114 | completed | True | False | 27 | 20.4 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 16 | 115 | completed | True | False | 19 | 22.3 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 17 | 116 | completed | True | False | 59 | 59.2 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 18 | 117 | completed | True | False | 19 | 18.2 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 19 | 118 | completed | False | False | 60 | 172.8 | no_wood_collected | explored |
| mx-history-sarsa-train | 20 | 119 | completed | True | False | 26 | 43.6 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 21 | 120 | completed | True | False | 38 | 34.1 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 22 | 121 | completed | True | False | 20 | 17.0 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 23 | 122 | completed | True | False | 32 | 26.7 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train | 24 | 123 | invalid | None | None | 22 | n/a | invalid_run | table_already_placed |
| mx-history-sarsa-train-part1 | 1 | 124 | completed | True | False | 14 | 18.9 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 2 | 125 | completed | False | False | 60 | 80.7 | pickaxe_not_crafted | crafted_oak_planks |
| mx-history-sarsa-train-part1 | 3 | 126 | completed | True | False | 14 | 20.7 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 4 | 127 | completed | True | False | 28 | 32.7 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 5 | 128 | completed | True | False | 50 | 153.5 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 6 | 129 | completed | True | False | 18 | 18.8 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 7 | 130 | completed | True | False | 46 | 39.5 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 8 | 131 | completed | True | False | 16 | 33.2 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 9 | 132 | completed | True | False | 52 | 54.8 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 10 | 133 | completed | True | False | 16 | 17.1 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 11 | 134 | completed | True | False | 41 | 117.5 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 12 | 135 | completed | True | False | 16 | 21.0 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 13 | 136 | completed | True | False | 18 | 24.4 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 14 | 137 | completed | False | False | 60 | 45.6 | pickaxe_not_crafted | table_already_placed |
| mx-history-sarsa-train-part1 | 15 | 138 | completed | False | False | 60 | 270.7 | no_wood_collected | missing_materials |
| mx-history-sarsa-train-part1 | 16 | 139 | completed | True | False | 16 | 21.1 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 17 | 140 | completed | True | False | 13 | 20.9 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 18 | 141 | completed | True | False | 15 | 17.0 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 19 | 142 | completed | False | False | 60 | 101.9 | no_wood_collected | missing_materials |
| mx-history-sarsa-train-part1 | 20 | 143 | completed | False | False | 60 | 13.6 | no_wood_collected | canopy_descent_refused:fall_too_far;exploration_blocked |
| mx-history-sarsa-train-part1 | 21 | 144 | completed | True | False | 32 | 38.5 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 22 | 145 | completed | True | False | 13 | 19.4 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 23 | 146 | completed | True | False | 32 | 30.1 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 24 | 147 | completed | True | False | 38 | 154.0 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 25 | 148 | completed | True | False | 17 | 18.4 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-train-part1 | 26 | 149 | completed | True | False | 13 | 14.0 | success | crafted_wooden_pickaxe |

## Reading these numbers

- Success means a wooden pickaxe was crafted from naturally gathered wood and confirmed held; it is not full Survival or an Ender Dragon result.
- Rates are computed over completed episodes only; invalid or interrupted episodes are counted separately and never treated as policy failures or successes.
- The Wilson 95% interval describes these episodes for one checkpoint or baseline under one step limit; it does not measure variation across independently trained agents.
- Returns from online training include exploratory actions and shaped rewards; compare them only with other training runs, not with frozen evaluation.
- The feasibility mask and scripted helpers supply substantial task knowledge; compare learned policies against the random baseline of the same condition (masked random with the mask, uniform random over all eight actions without it) before claiming a benefit.
- Conditions are never pooled. A benefit is claimed only when a learned policy's Wilson interval lies entirely above its baseline's; overlapping intervals mean no benefit was shown. Fisher's exact test is reported as a supplement, not as the decision rule.
- Runs recorded before experiment conditions existed are reported as the masked condition, which is how they ran.
