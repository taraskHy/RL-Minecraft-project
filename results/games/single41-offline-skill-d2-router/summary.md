# offline (prismarine-physics, no server)

Checkpoint `results/final_policy.pt`, 64 courses (from results/games/single41-first64-courses.json).

Neo skill `results/neo_skill.pt`: it acts instead of the policy from each router decision whose route crosses a wall in its next jump (for two decisions) or in its second (three), here 2901 of 10302 ticks; at the simulator's checkpoints on 64 of 64 courses.

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 44 | 86% | 86% |
| 7 | 20 | 85% | 85% |

Overall: 86% here, 86% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 7.11e-15 blocks; 64 of 64 episodes within 1e-6 blocks on every tick.
