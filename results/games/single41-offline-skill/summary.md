# offline (prismarine-physics, no server)

Checkpoint `results/final_policy.pt`, 64 courses (from results/games/single41-first64-courses.json).

Neo skill `results/neo_skill.pt`: it acts instead of the policy in each neo's block (from the platform before the takeoff to the one after the target), here 3749 of 10816 ticks.

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 44 | 82% | 82% |
| 7 | 20 | 75% | 75% |

Overall: 80% here, 80% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 7.11e-15 blocks; 64 of 64 episodes within 1e-6 blocks on every tick.
