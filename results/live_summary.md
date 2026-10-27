# Real Minecraft 1.21.4 server (final policy: obstacles-ft)

Mineflayer bot on a local void-world server; every episode played in real time. Server corrections (position resets): 0 in all 123 episodes.


Checkpoint `results/final_policy.pt`, 60 courses (held-out, 10 jumps each).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 10 | 90% | 90% |
| 1 | 10 | 100% | 100% |
| 2 | 10 | 100% | 100% |
| 3 | 10 | 100% | 100% |
| 4 | 10 | 100% | 100% |
| 5 | 10 | 100% | 100% |

Overall: 98% here, 98% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 2.13e-14 blocks; 60 of 60 episodes within 1e-6 blocks on every tick.


Checkpoint `results/final_policy.pt`, 60 courses (held-out with obstacles, 10 jumps each).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 10 | 100% | 100% |
| 1 | 10 | 100% | 100% |
| 2 | 10 | 100% | 100% |
| 3 | 10 | 100% | 100% |
| 4 | 10 | 100% | 100% |
| 5 | 10 | 80% | 80% |

Overall: 97% here, 97% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 2.13e-14 blocks; 60 of 60 episodes within 1e-6 blocks on every tick.


Checkpoint `results/final_policy.pt`, 3 courses (from configs/showcase_courses.json).

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 0 | 3 | 100% | 100% |

Overall: 100% here, 100% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 2.13e-14 blocks; 3 of 3 episodes within 1e-6 blocks on every tick.

