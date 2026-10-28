# offline (prismarine-physics, no server)

Checkpoint `results/final_policy.pt`, 31 courses (from configs/neo_live_courses.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 1 | 0% | 0% |
| 6 | 10 | 70% | 70% |
| 7 | 10 | 80% | 80% |
| 8 | 10 | 100% | 100% |

Overall: 81% here, 81% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 7.11e-15 blocks; 31 of 31 episodes within 1e-6 blocks on every tick.
