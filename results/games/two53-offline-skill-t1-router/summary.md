# offline (prismarine-physics, no server)

Checkpoint `results/final_policy.pt`, 64 courses (from results/games/two53-first64-courses.json).

Neo skill `results/neo_skill.pt`: it acts instead of the policy from each router decision whose route crosses a wall in its next jump (for two decisions) or in its second (three), here 5204 of 9998 ticks; at the simulator's checkpoints on 64 of 64 courses.

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 35 | 80% | 80% |
| 7 | 29 | 76% | 76% |

Overall: 78% here, 78% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 3.55e-15 blocks; 64 of 64 episodes within 1e-6 blocks on every tick.
