# live (Minecraft 1.21.4 server)

Checkpoint `results/final_policy.pt`, 64 courses (from results/games/water101-first64-courses.json).

Neo skill `results/water_skill.pt`: it acts instead of the policy in each water jump's block (from the platform before the takeoff to the exit), here 8786 of 13170 ticks.

| Level | Courses | Success here | Success in simulator (same courses) |
| --- | --- | --- | --- |
| 6 | 64 | 94% | 94% |

Overall: 94% here, 94% in the simulator; outcomes agree on 100% of courses.
Replaying the recorded actions in the simulator: largest position difference 5.86e-14 blocks; 64 of 64 episodes within 1e-6 blocks on every tick.
