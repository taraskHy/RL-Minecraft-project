# Plan for fresh-world Survival through the Ender Dragon

The final requested behavior is a fresh-world Survival start followed by an actual Ender Dragon defeat. The present repository does not implement that complete behavior. This plan preserves the target while defining the substantial work left for 3–4 students before 31 December 2026.

## Approach that keeps the RL explainable

Use structured observations and small discrete decisions first. A learned policy can decide which locally available action to take, while ordinary code implements movement, mining a selected block, and executing a selected recipe. Keep the boundary explicit. A finite-state coordinator is an understandable way to sequence learned skills, but its stage transitions are programmed, not learned. Do not describe a mostly scripted full-run controller as an end-to-end learned model.

Tabular Q-learning and SARSA are appropriate for the current compact state space. They cannot represent arbitrary Minecraft worlds with this six-integer observation. Introduce small task-specific representations first. If tables fail to generalize over measured continuous/richer observations, add one small DQN with replay and a target network, which are covered in `7.pdf`. Do not add PPO, transformers, pretrained language-model planning, and imitation learning all at once.

This is a proposed hybrid architecture, not a promise of full completion. Confirm the scientific contribution and permitted API observations with the instructor early. Full Survival with low-level actions is far harder than the starter task; for context, [the VPT project](https://openai.com/index/vpt/) used extensive video pretraining before RL fine-tuning for diamond-tool crafting. That work is related research, not a dependency or a method implemented here.

The user also identified Netherite, a potentially useful faster training environment. The [version-pinned assessment](netherite_assessment.md) records its current capabilities, external training dependencies, version mismatch with our adapter, and a bounded integration experiment. It is a candidate backend; no build or transfer result has been verified for this project.

## Stage gates

| Stage | Required behavior in actual Minecraft | Evidence needed before integration | Status |
| --- | --- | --- | --- |
| Learning foundation | Compare RL updates and reward designs on a controlled resource task | Reproducible simulator experiment with multiple seeds | Implemented; baseline runs recorded separately |
| Resource skill | Find natural trees; gather wood; craft/place a table; craft tools | Repeated successes from empty inventory on documented starts | Not implemented; the development arena supplies its table |
| Survival/equipment | Obtain food, stone/iron equipment and defend/recover | Metrics for survival, deaths, acquisition time, and failure causes | Not implemented |
| Nether access | Construct/use a portal with naturally obtained resources | Complete uninterrupted Overworld-to-Nether transitions | Not implemented |
| End resources | Obtain blaze rods and ender pearls; craft Eyes of Ender | Repeated in-game acquisitions with no command-generated inventory | Not implemented |
| Stronghold | Locate the stronghold, find and activate its End portal | Navigation and activation logs across starts | Not implemented |
| Dragon combat | Handle crystals, avoid attacks, damage the dragon appropriately | Repeated prepared-fight results, including failures; verified actual damage and defeat | Not implemented |
| Full run | Compose all stages from a fresh Survival world | Uninterrupted video/logs showing the complete run and defeat | Not implemented |

The arena bridge is only a development aid for the resource skill. It cannot be used unchanged as the final controller. Before writing the dragon policy, verify Mineflayer's interaction with dragon multipart hitboxes and actual health changes; do not assume an `attack` call against the parent entity proves damage. Similarly, a stage should advance from observed achievements, not from elapsed time or a command being sent.

## Work division and decision gates

Everyone first learns and reproduces the initial Q-learning/SARSA experiment. Then one member owns game interfaces and action reliability, one owns RL/state/reward experiments, and one owns integration/evaluation. A fourth can own combat and report/reproducibility, with reviews shared. Record real contributions, not these proposed assignments as if completed.

Use the dated schedule in [course_requirements.md](course_requirements.md). By 5 November, verify the first actual-game milestone. By 19 November, assess natural-world gathering and basic survival. By 3 December, assess Nether/stronghold progress and whether the combat interface works. By 10 December, attempt an uninterrupted fresh-world run. Reserve the remaining weeks for repeat evaluation, writing, and rehearsal.

These dates are gates for honest assessment, not guaranteed delivery dates. If a gate fails, identify the bottleneck and discuss the consequence immediately; do not silently change the final requirement to a simulated or prepared fight. The GPU helps only if a neural network becomes justified; it cannot fix incorrect observations, an unreliable action interface, or missing progression skills.

## Final-run protocol to freeze before evaluation

Record Java version, Minecraft version, server configuration, difficulty, permitted observations, world seed selection, start inventory, death/restart rule, and a time limit. A fresh-world run must not use teleports, creative items, prebuilt portals, direct damage commands, manual input, or the development-arena reset. Prepared states remain valid training aids when disclosed and kept outside final-run evidence.

Save the model/checkpoints, all settings, world seed, every intervention and failure, stage transitions, and the actual defeat event. Confirm dragon death in the game rather than equating a missing boss bar with a kill. Report both the learned contribution and what the coordinator/action helpers do. The full objective is achieved only after this evidence exists.
