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
