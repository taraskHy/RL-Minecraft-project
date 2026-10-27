# offline (prismarine-physics, no server)

Checkpoint `results/final_policy.pt`, 4 courses (from configs/showcase_courses.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 4 | 75% | 75% |

Overall: 75% here, 75% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 1.78e-15 blocks; 4 of 4 episodes within 1e-6 blocks on every tick.
