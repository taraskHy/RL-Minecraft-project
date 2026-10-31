# live (Minecraft 1.21.4 server)

Checkpoint `results/final_policy.pt`, 64 courses (from results/games/single41-first64-courses.json).

Neo skill `results/neo_skill.pt`: it acts instead of the policy in each neo's block (from the platform before the takeoff to the one after the target), here 3107 of 10529 ticks.

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 44 | 93% | 93% |
| 7 | 20 | 90% | 90% |

Overall: 92% here, 92% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 2.13e-14 blocks; 64 of 64 episodes within 1e-6 blocks on every tick.
