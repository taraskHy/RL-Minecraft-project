# offline (prismarine-physics, no server)

Checkpoint `results/combined_policy.pt`, 65 courses (from results/games/skill-replays-courses-65.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 3 | 67% | 67% |
| 7 | 3 | 33% | 33% |
| 8 | 3 | 67% | 67% |
| 9 | 4 | 100% | 100% |
| 10 | 4 | 100% | 100% |
| 11 | 4 | 100% | 100% |
| 12 | 24 | 96% | 96% |
| 13 | 4 | 75% | 75% |
| 14 | 4 | 100% | 100% |
| 15 | 2 | 100% | 100% |
| 16 | 2 | 100% | 100% |
| 17 | 2 | 100% | 100% |
| 18 | 2 | 100% | 100% |
| 19 | 2 | 100% | 100% |
| 20 | 2 | 100% | 100% |

Overall: 91% here, 91% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 8.88e-16 blocks; 65 of 65 episodes within 1e-6 blocks on every tick.
