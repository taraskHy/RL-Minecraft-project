# live (Minecraft 1.21.4 server)

Checkpoint `results/combined_policy.pt`, 16 courses (from results/games/skill-replays-courses-neo.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 11 | 36% | 36% |
| 7 | 5 | 60% | 60% |

Overall: 44% here, 44% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 2.13e-14 blocks; 16 of 16 episodes within 1e-6 blocks on every tick.
