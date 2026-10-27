# offline (prismarine-physics, no server)

Checkpoint `runs/obstacles-ft/checkpoints/best.pt`, 3 courses (from configs/showcase_courses.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 3 | 100% | 100% |

Overall: 100% here, 100% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 1.78e-15 blocks; 3 of 3 episodes within 1e-6 blocks on every tick.
