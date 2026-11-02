# live (Minecraft 1.21.4 server)

Checkpoint `results/final_policy.pt`, 64 courses (from results/games/slime56-first64-courses.json).

Neo skill `results/slime_skill.pt`: it acts instead of the policy in each slime pad's and shaped platform's block (from the takeoff onto it to its exit), here 7803 of 11305 ticks.

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 64 | 75% | 75% |

Overall: 75% here, 75% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 2.13e-14 blocks; 64 of 64 episodes within 1e-6 blocks on every tick.
