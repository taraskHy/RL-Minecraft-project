# offline (prismarine-physics, no server)

Checkpoint `results/final_policy.pt`, 16 courses (from results/games/skill-replays-courses-neo.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 11 | 91% | 91% |
| 7 | 5 | 60% | 60% |

Overall: 81% here, 81% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 3.55e-15 blocks; 16 of 16 episodes within 1e-6 blocks on every tick.
