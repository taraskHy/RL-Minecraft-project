# offline (prismarine-physics, no server)

Checkpoint `runs/neo-dagger3/checkpoints/best.pt`, 31 courses (from configs/neo_live_courses.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 1 | 0% | 0% |
| 6 | 10 | 50% | 50% |
| 7 | 10 | 30% | 30% |
| 8 | 10 | 30% | 30% |

Overall: 35% here, 35% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 7.11e-15 blocks; 31 of 31 episodes within 1e-6 blocks on every tick.
