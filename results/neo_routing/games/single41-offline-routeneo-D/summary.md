# offline (prismarine-physics, no server)

Checkpoint `results/final_policy.pt`, 64 courses (from results/games/single41-first64-courses.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 44 | 91% | 91% |
| 7 | 20 | 95% | 95% |

Overall: 92% here, 92% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 7.11e-15 blocks; 64 of 64 episodes within 1e-6 blocks on every tick.
