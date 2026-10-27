# offline (prismarine-physics, no server)

Checkpoint `runs/main-s0/checkpoints/best.pt`, 30 held-out courses (10 jumps each).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 5 | 100% | 100% |
| 1 | 5 | 100% | 100% |
| 2 | 5 | 100% | 100% |
| 3 | 5 | 100% | 100% |
| 4 | 5 | 100% | 100% |
| 5 | 5 | 60% | 60% |

Overall: 93% here, 93% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 7.11e-15 blocks; 30 of 30 episodes within 1e-6 blocks on every tick.
