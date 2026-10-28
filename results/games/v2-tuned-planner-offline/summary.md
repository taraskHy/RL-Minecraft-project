# offline (prismarine-physics, no server)

Checkpoint `results/final_policy.pt`, 60 courses (from results/games/v2-planner-offline/courses.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 10 | 100% | 100% |
| 1 | 10 | 100% | 100% |
| 2 | 10 | 100% | 100% |
| 3 | 10 | 90% | 90% |
| 4 | 10 | 60% | 60% |
| 5 | 10 | 60% | 60% |

Overall: 85% here, 85% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 7.11e-15 blocks; 60 of 60 episodes within 1e-6 blocks on every tick.
