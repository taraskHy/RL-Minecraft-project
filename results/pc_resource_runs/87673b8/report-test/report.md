# Natural-resource run summary

**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION**

| Group | Attempted | Completed | Invalid | Successes | Success rate | Wilson 95% | Deaths | Truncated | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked / frozen_evaluation / q_learning / checkpoint f5e09d4c2155 / max_steps 60 | 20 | 20 | 0 | 16 | 0.80 | 0.58–0.92 | 0 | 4 | 22.1 / 12.6 | 37.3 / 29.0 | 32 |
| masked / frozen_evaluation / random / masked_random / max_steps 60 | 20 | 20 | 0 | 19 | 0.95 | 0.76–0.99 | 0 | 1 | 14.6 / 12.2 | 31.2 / 29.1 | 21 |
| masked / frozen_evaluation / sarsa / checkpoint 4af5e623199e / max_steps 60 | 20 | 20 | 0 | 18 | 0.90 | 0.70–0.97 | 0 | 2 | 19.2 / 14.7 | 35.0 / 32.6 | 39 |
| unmasked / frozen_evaluation / q_learning / checkpoint 7d0fcce8653d / max_steps 60 | 20 | 20 | 0 | 20 | 1.00 | 0.84–1.00 | 0 | 0 | 14.1 / 14.1 | 39.1 / 39.1 | 26 |
| unmasked / frozen_evaluation / random / uniform_random / max_steps 60 | 20 | 20 | 0 | 4 | 0.20 | 0.08–0.42 | 0 | 16 | 58.6 / 53 | 41.9 / 35.0 | 24 |
| unmasked / frozen_evaluation / sarsa / checkpoint 8d17136157fd / max_steps 60 | 20 | 20 | 0 | 14 | 0.70 | 0.48–0.85 | 0 | 6 | 26.1 / 11.6 | 39.8 / 22.3 | 9 |
| unmasked_history / frozen_evaluation / q_learning / checkpoint 3362dd3fab15 / max_steps 60 | 20 | 20 | 0 | 16 | 0.80 | 0.58–0.92 | 0 | 4 | 25.1 / 16.3 | 29.5 / 24.8 | 24 |
| unmasked_history / frozen_evaluation / sarsa / checkpoint 36c2cd4005f1 / max_steps 60 | 20 | 20 | 0 | 17 | 0.85 | 0.64–0.95 | 0 | 3 | 21.8 / 15.1 | 28.4 / 23.7 | 17 |

## Decisions

How much each decision was left to the policy: the share with a single allowed action, the share of choices the disclosed mask would have forbidden, and the share that changed none of the nine resource_v1 fields.

| Group | Decisions | Single allowed action | Choices outside the mask | No progress |
| --- | --- | --- | --- | --- |
| masked / frozen_evaluation / q_learning / checkpoint f5e09d4c2155 / max_steps 60 | 442 | 79% | 0% | 56% |
| masked / frozen_evaluation / random / masked_random / max_steps 60 | 292 | 54% | 0% | 26% |
| masked / frozen_evaluation / sarsa / checkpoint 4af5e623199e / max_steps 60 | 385 | 70% | 0% | 46% |
| unmasked / frozen_evaluation / q_learning / checkpoint 7d0fcce8653d / max_steps 60 | 281 | 0% | 2% | 19% |
| unmasked / frozen_evaluation / random / uniform_random / max_steps 60 | 1172 | 0% | 81% | 76% |
| unmasked / frozen_evaluation / sarsa / checkpoint 8d17136157fd / max_steps 60 | 522 | 0% | 41% | 60% |
| unmasked_history / frozen_evaluation / q_learning / checkpoint 3362dd3fab15 / max_steps 60 | 501 | 0% | 29% | 58% |
| unmasked_history / frozen_evaluation / sarsa / checkpoint 36c2cd4005f1 / max_steps 60 | 436 | 0% | 32% | 52% |

## Comparison with the random baseline

Each frozen learned policy against the random baseline with the same action mask, step limit, world seeds, pause, and helper revision, counted on the seeds both completed. A benefit is claimed only when the learned interval lies entirely above the baseline's. Fisher's exact p is two-sided and supplementary.

| Condition | Policy | Baseline | Seeds compared | Learned successes (Wilson 95%) | Baseline successes (Wilson 95%) | Fisher exact p | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| masked | q_learning | masked_random (masked) | 20 | 16/20 (0.58–0.92) | 19/20 (0.76–0.99) | 0.342 | no benefit shown: the intervals overlap |
| masked | sarsa | masked_random (masked) | 20 | 18/20 (0.70–0.97) | 19/20 (0.76–0.99) | 1 | no benefit shown: the intervals overlap |
| unmasked | q_learning | uniform_random (unmasked) | 20 | 20/20 (0.84–1.00) | 4/20 (0.08–0.42) | 1.54e-07 | benefit shown: the learned interval lies entirely above the baseline's |
| unmasked | sarsa | uniform_random (unmasked) | 20 | 14/20 (0.48–0.85) | 4/20 (0.08–0.42) | 0.00364 | benefit shown: the learned interval lies entirely above the baseline's |
| unmasked_history | q_learning | uniform_random (unmasked) | 20 | 16/20 (0.58–0.92) | 4/20 (0.08–0.42) | 0.00036 | benefit shown: the learned interval lies entirely above the baseline's |
| unmasked_history | sarsa | uniform_random (unmasked) | 20 | 17/20 (0.64–0.95) | 4/20 (0.08–0.42) | 8.75e-05 | benefit shown: the learned interval lies entirely above the baseline's |


## Effect of the history bit

Each unmasked_history policy against the unmasked policy of the same algorithm on the seeds both completed. The bit is credited with an effect only when the history interval lies entirely above the unmasked one; with one trained agent per condition this compares two checkpoints, not two state designs.

| Policy | Seeds compared | With the bit (Wilson 95%) | Without (Wilson 95%) | Fisher exact p | Verdict |
| --- | --- | --- | --- | --- | --- |
| q_learning | 20 | 16/20 (0.58–0.92) | 20/20 (0.84–1.00) | 0.106 | no effect of the bit shown: the intervals overlap |
| sarsa | 20 | 17/20 (0.64–0.95) | 14/20 (0.48–0.85) | 0.451 | no effect of the bit shown: the intervals overlap |

## masked / frozen_evaluation / q_learning / checkpoint f5e09d4c2155 / max_steps 60

Runs: mx-masked-q-test. World seeds: 20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019.
Evaluated checkpoint (policy.json SHA-256): f5e09d4c21556975fb02ef80fdbf0e23ba393b753ff028da36255d13ca2303c8.
Mean frozen-policy shaped return: 7.393.
Leaf clearing: 32 canopy layers, 11 wall blocks, 3 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×2, pickaxe_not_crafted ×1, table_not_placed ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: exploration_blocked ×154, canopy_descent_refused:fall_too_far ×147, canopy_descent_refused:hazard_below ×59, log_path_blocked ×55, log_collected ×50, crafted_oak_planks ×26, log_broken_without_pickup ×26, crafted_stick ×18.

## masked / frozen_evaluation / random / masked_random / max_steps 60

Runs: mx-masked-random-test. World seeds: 20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019.
Mean frozen-policy shaped return: 8.954.
Leaf clearing: 21 canopy layers, 2 wall blocks, 3 fall damage in total.
Failure causes (completed episodes only): pickaxe_not_crafted ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×59, canopy_descent_refused:fall_too_far ×52, crafted_oak_planks ×35, exploration_blocked ×35, crafted_stick ×20, crafted_crafting_table ×20, table_placed ×20, log_path_blocked ×20.

## masked / frozen_evaluation / sarsa / checkpoint 4af5e623199e / max_steps 60

Runs: mx-masked-sarsa-test. World seeds: 20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019.
Evaluated checkpoint (policy.json SHA-256): 4af5e623199e2c763bd0944e69734da22a6cd762ea969f61aa02308e774fdc37.
Mean frozen-policy shaped return: 8.376.
Leaf clearing: 39 canopy layers, 11 wall blocks, 3 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×2.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: canopy_descent_refused:fall_too_far ×112, exploration_blocked ×102, log_collected ×53, leaf_escape_refused:no_leaf_exit ×33, no_placement_surface ×32, crafted_oak_planks ×29, explored ×29, crafted_stick ×18.

## unmasked / frozen_evaluation / q_learning / checkpoint 7d0fcce8653d / max_steps 60

Runs: mx-unmasked-q-test. World seeds: 20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019.
Evaluated checkpoint (policy.json SHA-256): 7d0fcce8653df8d4afc0867b321fd9670a8c64ab6ef03a876c64b9cd1c9e0349.
Mean frozen-policy shaped return: 9.451.
Leaf clearing: 26 canopy layers, 3 wall blocks, 3 fall damage in total.
Failure causes (completed episodes only): none.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: log_collected ×58, explored ×34, crafted_oak_planks ×33, crafted_stick ×22, crafted_crafting_table ×20, table_placed ×20, crafted_wooden_pickaxe ×20, crafted_spruce_planks ×15.

## unmasked / frozen_evaluation / random / uniform_random / max_steps 60

Runs: mx-unmasked-random-test. World seeds: 20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019.
Mean frozen-policy shaped return: 1.113.
Leaf clearing: 24 canopy layers, 2 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): pickaxe_not_crafted ×10, table_not_placed ×6.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: missing_materials ×311, log_already_reachable ×127, log_collected ×109, not_at_own_table ×107, no_own_table ×104, missing_table ×86, table_already_placed ×53, table_reached ×41.

## unmasked / frozen_evaluation / sarsa / checkpoint 8d17136157fd / max_steps 60

Runs: mx-unmasked-sarsa-test. World seeds: 20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019.
Evaluated checkpoint (policy.json SHA-256): 8d17136157fd51379fab4e2a4fef94d47f3a0d3a8258a8f29b5d1b1a3dcac2d5.
Mean frozen-policy shaped return: 6.309.
Leaf clearing: 9 canopy layers, 0 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×1, pickaxe_not_crafted ×5.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: missing_materials ×115, log_collected ×58, explored_locally ×50, table_reached ×39, table_already_placed ×32, explored ×30, crafted_oak_planks ×26, crafted_stick ×21.

## unmasked_history / frozen_evaluation / q_learning / checkpoint 3362dd3fab15 / max_steps 60

Runs: mx-history-q-test. World seeds: 20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019.
Evaluated checkpoint (policy.json SHA-256): 3362dd3fab15f4ab8e0f5b0442b1a6eac2ac65b7bc0fee83bf7b4d71a674da51.
Mean frozen-policy shaped return: 7.259.
Leaf clearing: 24 canopy layers, 5 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×2, pickaxe_not_crafted ×2.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: exploration_blocked ×115, canopy_descent_refused:fall_too_far ×110, missing_materials ×63, log_collected ×54, leaf_escape_refused:no_leaf_exit ×50, crafted_oak_planks ×29, no_own_table ×22, crafted_stick ×22.

## unmasked_history / frozen_evaluation / sarsa / checkpoint 36c2cd4005f1 / max_steps 60

Runs: mx-history-sarsa-test. World seeds: 20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019.
Evaluated checkpoint (policy.json SHA-256): 36c2cd4005f1650dba4b75e5de24615ec341bba8b166ec0b50a929c0b5fcb602.
Mean frozen-policy shaped return: 7.759.
Leaf clearing: 17 canopy layers, 5 wall blocks, 2 fall damage in total.
Failure causes (completed episodes only): no_wood_collected ×1, pickaxe_not_crafted ×1, table_not_placed ×1.
Not measured (invalid, interrupted, or unfinished; excluded from every rate): none.
Most frequent step events: canopy_descent_refused:fall_too_far ×69, missing_materials ×65, exploration_blocked ×62, log_collected ×54, not_at_own_table ×32, crafted_oak_planks ×30, crafted_stick ×23, table_already_placed ×23.

## Episodes

| Run | Episode | Seed | Status | Success | Death | Steps | Task s | Failure cause | Final event |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mx-masked-q-test | 1 | 20000 | completed | False | False | 60 | 38.1 | no_wood_collected | canopy_descent_refused:hazard_below;exploration_blocked |
| mx-masked-q-test | 2 | 20001 | completed | True | False | 12 | 27.1 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 3 | 20002 | completed | True | False | 14 | 54.1 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 4 | 20003 | completed | True | False | 11 | 27.9 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 5 | 20004 | completed | True | False | 14 | 18.4 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 6 | 20005 | completed | True | False | 11 | 17.0 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 7 | 20006 | completed | True | False | 12 | 29.5 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 8 | 20007 | completed | True | False | 12 | 19.3 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 9 | 20008 | completed | True | False | 11 | 15.3 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 10 | 20009 | completed | True | False | 13 | 46.2 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 11 | 20010 | completed | True | False | 24 | 63.8 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 12 | 20011 | completed | True | False | 15 | 37.8 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 13 | 20012 | completed | False | False | 60 | 90.0 | pickaxe_not_crafted | canopy_descent_refused:fall_too_far;exploration_blocked |
| mx-masked-q-test | 14 | 20013 | completed | False | False | 60 | 30.3 | no_wood_collected | canopy_descent_refused:fall_too_far;exploration_blocked |
| mx-masked-q-test | 15 | 20014 | completed | True | False | 10 | 30.2 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 16 | 20015 | completed | True | False | 10 | 13.8 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 17 | 20016 | completed | False | False | 60 | 123.8 | table_not_placed | canopy_descent_refused:fall_too_far;exploration_blocked |
| mx-masked-q-test | 18 | 20017 | completed | True | False | 11 | 17.7 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 19 | 20018 | completed | True | False | 11 | 20.9 | success | crafted_wooden_pickaxe |
| mx-masked-q-test | 20 | 20019 | completed | True | False | 11 | 24.4 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 1 | 20000 | completed | True | False | 10 | 27.2 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 2 | 20001 | completed | True | False | 11 | 26.2 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 3 | 20002 | completed | True | False | 11 | 16.2 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 4 | 20003 | completed | True | False | 11 | 16.9 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 5 | 20004 | completed | True | False | 11 | 22.1 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 6 | 20005 | completed | True | False | 12 | 19.9 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 7 | 20006 | completed | True | False | 14 | 46.5 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 8 | 20007 | completed | True | False | 11 | 25.8 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 9 | 20008 | completed | True | False | 12 | 30.5 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 10 | 20009 | completed | True | False | 12 | 37.6 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 11 | 20010 | completed | True | False | 25 | 76.0 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 12 | 20011 | completed | True | False | 12 | 24.4 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 13 | 20012 | completed | False | False | 60 | 43.1 | no_wood_collected | canopy_descent_refused:fall_too_far;exploration_blocked |
| mx-masked-sarsa-test | 14 | 20013 | completed | True | False | 11 | 22.5 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 15 | 20014 | completed | True | False | 13 | 45.9 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 16 | 20015 | completed | False | False | 60 | 69.7 | no_wood_collected | canopy_descent_refused:fall_too_far;exploration_blocked |
| mx-masked-sarsa-test | 17 | 20016 | completed | True | False | 20 | 61.3 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 18 | 20017 | completed | True | False | 45 | 29.3 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 19 | 20018 | completed | True | False | 11 | 29.6 | success | crafted_wooden_pickaxe |
| mx-masked-sarsa-test | 20 | 20019 | completed | True | False | 13 | 28.7 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 1 | 20000 | completed | True | False | 10 | 25.3 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 2 | 20001 | completed | True | False | 12 | 46.7 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 3 | 20002 | completed | True | False | 11 | 19.6 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 4 | 20003 | completed | True | False | 11 | 23.0 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 5 | 20004 | completed | True | False | 10 | 27.9 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 6 | 20005 | completed | True | False | 11 | 21.1 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 7 | 20006 | completed | True | False | 15 | 39.5 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 8 | 20007 | completed | True | False | 12 | 23.4 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 9 | 20008 | completed | True | False | 11 | 24.3 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 10 | 20009 | completed | False | False | 60 | 70.7 | pickaxe_not_crafted | canopy_descent_refused:fall_too_far;exploration_blocked |
| mx-masked-random-test | 11 | 20010 | completed | True | False | 12 | 33.9 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 12 | 20011 | completed | True | False | 15 | 36.8 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 13 | 20012 | completed | True | False | 13 | 25.7 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 14 | 20013 | completed | True | False | 11 | 24.9 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 15 | 20014 | completed | True | False | 12 | 27.8 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 16 | 20015 | completed | True | False | 10 | 22.4 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 17 | 20016 | completed | True | False | 13 | 26.2 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 18 | 20017 | completed | True | False | 11 | 24.3 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 19 | 20018 | completed | True | False | 11 | 21.0 | success | crafted_wooden_pickaxe |
| mx-masked-random-test | 20 | 20019 | completed | True | False | 21 | 58.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 1 | 20000 | completed | True | False | 17 | 43.4 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 2 | 20001 | completed | True | False | 14 | 45.1 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 3 | 20002 | completed | True | False | 12 | 18.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 4 | 20003 | completed | True | False | 10 | 21.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 5 | 20004 | completed | True | False | 11 | 22.4 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 6 | 20005 | completed | True | False | 11 | 21.6 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 7 | 20006 | completed | True | False | 18 | 54.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 8 | 20007 | completed | True | False | 12 | 26.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 9 | 20008 | completed | True | False | 11 | 18.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 10 | 20009 | completed | True | False | 12 | 24.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 11 | 20010 | completed | True | False | 33 | 133.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 12 | 20011 | completed | True | False | 12 | 25.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 13 | 20012 | completed | True | False | 14 | 30.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 14 | 20013 | completed | True | False | 10 | 23.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 15 | 20014 | completed | True | False | 20 | 124.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 16 | 20015 | completed | True | False | 11 | 21.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 17 | 20016 | completed | True | False | 10 | 19.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 18 | 20017 | completed | True | False | 11 | 20.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 19 | 20018 | completed | True | False | 11 | 23.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-q-test | 20 | 20019 | completed | True | False | 21 | 64.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 1 | 20000 | completed | True | False | 11 | 23.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 2 | 20001 | completed | False | False | 60 | 88.4 | pickaxe_not_crafted | table_reached |
| mx-unmasked-sarsa-test | 3 | 20002 | completed | False | False | 60 | 49.9 | pickaxe_not_crafted | missing_materials |
| mx-unmasked-sarsa-test | 4 | 20003 | completed | True | False | 11 | 22.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 5 | 20004 | completed | True | False | 11 | 23.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 6 | 20005 | completed | True | False | 11 | 18.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 7 | 20006 | completed | True | False | 16 | 29.5 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 8 | 20007 | completed | False | False | 60 | 121.3 | no_wood_collected | explored_locally |
| mx-unmasked-sarsa-test | 9 | 20008 | completed | True | False | 12 | 20.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 10 | 20009 | completed | True | False | 12 | 23.8 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 11 | 20010 | completed | False | False | 60 | 181.7 | pickaxe_not_crafted | table_reached |
| mx-unmasked-sarsa-test | 12 | 20011 | completed | True | False | 15 | 42.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 13 | 20012 | completed | True | False | 11 | 25.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 14 | 20013 | completed | True | False | 10 | 14.2 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 15 | 20014 | completed | True | False | 10 | 17.6 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 16 | 20015 | completed | False | False | 60 | 22.0 | pickaxe_not_crafted | missing_materials |
| mx-unmasked-sarsa-test | 17 | 20016 | completed | True | False | 10 | 16.4 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 18 | 20017 | completed | False | False | 60 | 21.2 | pickaxe_not_crafted | missing_materials |
| mx-unmasked-sarsa-test | 19 | 20018 | completed | True | False | 11 | 14.7 | success | crafted_wooden_pickaxe |
| mx-unmasked-sarsa-test | 20 | 20019 | completed | True | False | 11 | 20.6 | success | crafted_wooden_pickaxe |
| mx-unmasked-random-test | 1 | 20000 | completed | False | False | 60 | 42.3 | pickaxe_not_crafted | missing_materials |
| mx-unmasked-random-test | 2 | 20001 | completed | False | False | 60 | 80.4 | pickaxe_not_crafted | table_reached |
| mx-unmasked-random-test | 3 | 20002 | completed | True | False | 44 | 23.0 | success | crafted_wooden_pickaxe |
| mx-unmasked-random-test | 4 | 20003 | completed | True | False | 56 | 35.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-random-test | 5 | 20004 | completed | False | False | 60 | 41.9 | pickaxe_not_crafted | log_collected |
| mx-unmasked-random-test | 6 | 20005 | completed | False | False | 60 | 60.4 | table_not_placed | crafted_birch_planks |
| mx-unmasked-random-test | 7 | 20006 | completed | False | False | 60 | 43.4 | pickaxe_not_crafted | crafted_spruce_planks |
| mx-unmasked-random-test | 8 | 20007 | completed | False | False | 60 | 48.0 | pickaxe_not_crafted | crafted_oak_planks |
| mx-unmasked-random-test | 9 | 20008 | completed | False | False | 60 | 52.6 | pickaxe_not_crafted | crafted_oak_planks |
| mx-unmasked-random-test | 10 | 20009 | completed | False | False | 60 | 29.4 | pickaxe_not_crafted | missing_materials |
| mx-unmasked-random-test | 11 | 20010 | completed | True | False | 59 | 50.1 | success | crafted_wooden_pickaxe |
| mx-unmasked-random-test | 12 | 20011 | completed | True | False | 53 | 30.9 | success | crafted_wooden_pickaxe |
| mx-unmasked-random-test | 13 | 20012 | completed | False | False | 60 | 47.9 | table_not_placed | log_already_reachable |
| mx-unmasked-random-test | 14 | 20013 | completed | False | False | 60 | 62.0 | pickaxe_not_crafted | log_collected |
| mx-unmasked-random-test | 15 | 20014 | completed | False | False | 60 | 41.5 | table_not_placed | crafted_spruce_planks |
| mx-unmasked-random-test | 16 | 20015 | completed | False | False | 60 | 26.9 | table_not_placed | crafted_stick |
| mx-unmasked-random-test | 17 | 20016 | completed | False | False | 60 | 28.1 | pickaxe_not_crafted | table_already_placed |
| mx-unmasked-random-test | 18 | 20017 | completed | False | False | 60 | 24.4 | pickaxe_not_crafted | missing_materials |
| mx-unmasked-random-test | 19 | 20018 | completed | False | False | 60 | 21.7 | table_not_placed | crafted_crafting_table |
| mx-unmasked-random-test | 20 | 20019 | completed | False | False | 60 | 46.2 | table_not_placed | crafted_crafting_table |
| mx-history-q-test | 1 | 20000 | completed | True | False | 14 | 18.8 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 2 | 20001 | completed | True | False | 20 | 26.0 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 3 | 20002 | completed | True | False | 14 | 18.7 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 4 | 20003 | completed | True | False | 14 | 19.5 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 5 | 20004 | completed | True | False | 14 | 19.8 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 6 | 20005 | completed | False | False | 60 | 56.7 | pickaxe_not_crafted | no_reachable_log |
| mx-history-q-test | 7 | 20006 | completed | True | False | 14 | 15.2 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 8 | 20007 | completed | True | False | 15 | 24.5 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 9 | 20008 | completed | True | False | 15 | 24.3 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 10 | 20009 | completed | False | False | 60 | 35.2 | pickaxe_not_crafted | no_reachable_log |
| mx-history-q-test | 11 | 20010 | completed | True | False | 27 | 62.9 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 12 | 20011 | completed | True | False | 18 | 24.1 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 13 | 20012 | completed | False | False | 60 | 69.0 | no_wood_collected | canopy_descent_refused:fall_too_far;leaf_escape_refused:no_leaf_exit;exploration_blocked |
| mx-history-q-test | 14 | 20013 | completed | True | False | 14 | 18.6 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 15 | 20014 | completed | True | False | 14 | 16.4 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 16 | 20015 | completed | True | False | 14 | 14.5 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 17 | 20016 | completed | False | False | 60 | 31.9 | no_wood_collected | canopy_descent_refused:fall_too_far;exploration_blocked |
| mx-history-q-test | 18 | 20017 | completed | True | False | 24 | 48.1 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 19 | 20018 | completed | True | False | 14 | 17.9 | success | crafted_wooden_pickaxe |
| mx-history-q-test | 20 | 20019 | completed | True | False | 16 | 27.5 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 1 | 20000 | completed | True | False | 12 | 15.0 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 2 | 20001 | completed | True | False | 12 | 13.9 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 3 | 20002 | completed | True | False | 13 | 17.0 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 4 | 20003 | completed | True | False | 14 | 27.0 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 5 | 20004 | completed | True | False | 13 | 20.3 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 6 | 20005 | completed | True | False | 12 | 15.5 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 7 | 20006 | completed | False | False | 60 | 14.2 | table_not_placed | missing_materials |
| mx-history-sarsa-test | 8 | 20007 | completed | True | False | 13 | 21.8 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 9 | 20008 | completed | True | False | 12 | 17.3 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 10 | 20009 | completed | True | False | 37 | 63.0 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 11 | 20010 | completed | True | False | 14 | 25.8 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 12 | 20011 | completed | True | False | 17 | 28.2 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 13 | 20012 | completed | True | False | 13 | 17.2 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 14 | 20013 | completed | False | False | 60 | 41.5 | no_wood_collected | canopy_descent_refused:fall_too_far;exploration_blocked |
| mx-history-sarsa-test | 15 | 20014 | completed | True | False | 13 | 21.1 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 16 | 20015 | completed | True | False | 12 | 22.8 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 17 | 20016 | completed | False | False | 60 | 110.1 | pickaxe_not_crafted | canopy_descent_refused:fall_too_far;exploration_blocked |
| mx-history-sarsa-test | 18 | 20017 | completed | True | False | 16 | 22.4 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 19 | 20018 | completed | True | False | 15 | 17.3 | success | crafted_wooden_pickaxe |
| mx-history-sarsa-test | 20 | 20019 | completed | True | False | 18 | 37.8 | success | crafted_wooden_pickaxe |

## Reading these numbers

- Success means a wooden pickaxe was crafted from naturally gathered wood and confirmed held; it is not full Survival or an Ender Dragon result.
- Rates are computed over completed episodes only; invalid or interrupted episodes are counted separately and never treated as policy failures or successes.
- The Wilson 95% interval describes these episodes for one checkpoint or baseline under one step limit; it does not measure variation across independently trained agents.
- Returns from online training include exploratory actions and shaped rewards; compare them only with other training runs, not with frozen evaluation.
- The feasibility mask and scripted helpers supply substantial task knowledge; compare learned policies against the random baseline of the same condition (masked random with the mask, uniform random over all eight actions without it) before claiming a benefit.
- Conditions are never pooled. A benefit is claimed only when a learned policy's Wilson interval lies entirely above its baseline's; overlapping intervals mean no benefit was shown. Fisher's exact test is reported as a supplement, not as the decision rule.
- Runs recorded before experiment conditions existed are reported as the masked condition, which is how they ran.
