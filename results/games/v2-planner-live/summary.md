# live (Minecraft 1.21.4 server)

Checkpoint `results/final_policy.pt`, 60 courses (held-out, 10 jumps each).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 10 | 100% | 100% |
| 1 | 10 | 100% | 100% |
| 2 | 10 | 100% | 100% |
| 3 | 10 | 90% | 90% |
| 4 | 10 | 60% | 60% |
| 5 | 10 | 50% | 50% |

Overall: 83% here, 83% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 2.13e-14 blocks; 60 of 60 episodes within 1e-6 blocks on every tick.
