# offline (prismarine-physics, no server)

Checkpoint `results/combined_policy.pt`, 60 courses (from results/games/v2-live-obstacles/courses.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 10 | 100% | 100% |
| 1 | 10 | 100% | 100% |
| 2 | 10 | 100% | 100% |
| 3 | 10 | 100% | 100% |
| 4 | 10 | 100% | 100% |
| 5 | 10 | 90% | 90% |

Overall: 98% here, 98% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 1.42e-14 blocks; 60 of 60 episodes within 1e-6 blocks on every tick.
