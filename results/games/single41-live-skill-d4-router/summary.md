# live (Minecraft 1.21.4 server)

Checkpoint `results/final_policy.pt`, 64 courses (from results/games/single41-first64-courses.json).

Neo skill `results/neo_skill.pt`: it acts instead of the policy from each router decision whose route crosses a wall in its next jump (for two decisions) or in its second (three), here 2994 of 9947 ticks; at the simulator's checkpoints on 63 of 64 courses.

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 44 | 84% | 84% |
| 7 | 20 | 85% | 90% |

Overall: 84% here, 86% in the simulator; outcomes agree on 98% of courses.
Replaying the recorded actions in the simulator: largest position difference 2.91e+00 blocks; 63 of 64 episodes within 1e-6 blocks on every tick.
