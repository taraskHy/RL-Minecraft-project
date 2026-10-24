# Report and presentation plan

This is a writing plan, not a finished report and not evidence of a completed Minecraft run. Use actual outputs and observed failures to fill it in after the experiments.

The handout requires 8–20 pages excluding references, Arial 12, 1.5 line spacing, one-inch margins, and a 15-minute presentation. A target of roughly 12–15 pages leaves room for explanations the entire team understands.

| Report section | Suggested space | Evidence to include |
| --- | --- | --- |
| Overview and scope | 1 page | Full Survival goal, why it is difficult, exact current achievement |
| Problem formulation | 2 pages | State/action/reward/dynamics, terminal and time-limit rules, learned/scripted boundary |
| Methodology | 2–3 pages | Q-learning and SARSA equations, exploration, shaping, implementation and optional justified extensions |
| Experimental design | 1–2 pages | Baselines, grid, budgets, seed splits, metrics, compute/time, selection rule |
| Results | 2–3 pages | Actual curves/tables, seed dispersion, failure examples, simulator/game/full-run results separately |
| Discussion and related work | 2 pages | What caused successes/failures, modeling mismatch, reproducibility and limits, relevant research |
| Conclusion and contributions | 1 page | Claims supported by results; actual contributions and future work |
| References | Excluded from page count | Supplied lectures, algorithm/reward sources, software versions and research cited |

Do not fill an absent experiment with hypothetical numbers. Label the larger sweep and game experiments as planned until run. A result on the supplied-table simulator must not be presented as natural-world crafting. A prepared dragon fight must not be presented as fresh-world Survival.

## Related work

### Plan4MC (Yuan et al., 2023)

*Skill Reinforcement Learning and Planning for Open-World Long-Horizon Tasks*, arXiv [2303.16563](https://arxiv.org/abs/2303.16563) (v2, 4 December 2023; NeurIPS 2023 Foundation Models for Decision Making workshop). Code: [PKU-RL/Plan4MC](https://github.com/PKU-RL/Plan4MC), MIT licence. Every number below was checked against v2 (Sec. 3–5, Tables 2, 5 and 7–9).

- **Method.** MineDojo programmatic tasks. The agent sees RGB images plus compass, location and biome, and acts through 12×3 discrete actions. Skills come in three types:
  - *Finding*: one hierarchical policy. The high level is PPO with a state-count reward (+1 for every unvisited grid cell); the low level is pre-trained with DQN for goal navigation.
  - *Manipulation*: PPO with self-imitation, using MineCLIP or distance, attack and depth rewards.
  - *Crafting*: a single scripted action, not learned.

  ChatGPT (GPT-3.5) generated a skill dependency graph, which was checked against the Minecraft database (6 mistakes among 55 skills). A depth-first planner over that graph alternates planning and execution.
- **Budget and protocol.** 7M environment steps in total for all skills; 40 tasks in four groups of 10; 30 test episodes per task. Each task gives its starting tools and biome.
- **Results** (Table 2, success per task group). Without the Finding skill, Mine-Stones drops from 0.293 to 0.097. Nothing beyond iron tools is evaluated.

  | Task group | Plan4MC | MineAgent | Plan4MC without Finding |
  | --- | --- | --- | --- |
  | Cut-Trees | 0.417 | 0.003 | 0.187 |
  | Mine-Stones | 0.293 | 0.026 | 0.097 |
  | Mine-Ores | 0.267 | 0.000 | 0.243 |
  | Interact-Mobs | 0.320 | 0.171 | 0.170 |

  Single tasks (Table 9):

  | Task | Plan4MC success | Starting inventory |
  | --- | --- | --- |
  | wooden_pickaxe | 0.53 | none |
  | stone_pickaxe | 0.17 | a wooden pickaxe |
  | iron_pickaxe | 0.17 | five stone pickaxes and 64 dirt |
- **Why it matters to us.** Its structure is close to ours: learned decisions over skills, with crafting scripted. Its ablation also points at the same bottleneck we see: finding things (our SEARCH helper and the canopy failures). It differs in three ways:
  - pixels and neural policies, against our compact symbolic state and tabular methods;
  - given starting tools, against fresh worlds with an empty inventory;
  - MineDojo, against a real Java 1.21.4 server through Mineflayer.
- **Not comparable.** Its success rates cannot be compared with ours: the tasks, simulator, starting inventories, step limits, observation and action spaces, and test protocols all differ. Cite it as related work, not as a baseline. We do not use its code.

## Fifteen-minute presentation

| Time | Topic |
| --- | --- |
| 0–2 minutes | Goal, motivation, achieved scope |
| 2–5 minutes | Environment and learned decisions |
| 5–8 minutes | One numerical TD update; Q-learning versus SARSA |
| 8–11 minutes | Actual results, comparison, and one informative failure |
| 11–14 minutes | Verified demo evidence and learned/scripted contribution |
| 14–15 minutes | Limitations, conclusions, and contributions |

Rehearse transitions between speakers. Retain a recorded successful run and a short failure example so the presentation does not depend entirely on live game setup. Final success footage must meet the fresh-world protocol in [full_survival_plan.md](full_survival_plan.md).
