# offline (prismarine-physics, no server)

Checkpoint `results/final_policy.pt`, 60 courses (held-out with obstacles, 10 jumps each).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 10 | 100% | 100% |
| 1 | 10 | 80% | 80% |
| 2 | 10 | 90% | 90% |
| 3 | 10 | 100% | 100% |
| 4 | 10 | 90% | 90% |
| 5 | 10 | 90% | 90% |

Overall: 92% here, 92% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 7.11e-15 blocks; 60 of 60 episodes within 1e-6 blocks on every tick.
