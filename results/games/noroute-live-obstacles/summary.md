# live (Minecraft 1.21.4 server)

Checkpoint `runs/noroute-v2/checkpoints/best.pt`, 60 courses (held-out with obstacles, 10 jumps each).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 10 | 50% | 50% |
| 1 | 10 | 60% | 60% |
| 2 | 10 | 90% | 90% |
| 3 | 10 | 60% | 60% |
| 4 | 10 | 70% | 70% |
| 5 | 10 | 40% | 40% |

Overall: 62% here, 62% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 2.13e-14 blocks; 60 of 60 episodes within 1e-6 blocks on every tick.
