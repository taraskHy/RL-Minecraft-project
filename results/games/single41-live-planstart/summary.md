# live (Minecraft 1.21.4 server)

Checkpoint `results/final_policy.pt`, 64 courses (from results/games/single41-first64-courses.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 44 | 86% | 86% |
| 7 | 20 | 80% | 85% |

Overall: 84% here, 86% in the simulator; outcomes agree on 98% of courses.
Replaying the recorded actions in the simulator: largest position difference 2.73e+02 blocks; 62 of 64 episodes within 1e-6 blocks on every tick.
