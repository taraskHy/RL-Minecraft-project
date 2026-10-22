# Netherite assessment for this project

Reviewed on 23 October 2026 after the user shared [Elliot Arledge's announcement](https://x.com/elliotarledge/status/2082366172222439879). **Decision: investigate Netherite as an optional faster training environment. Adoption and actual-game transfer are not yet verified.** The target remains fresh-world Minecraft Survival through an Ender Dragon defeat.

## What the announcement and current code establish

The announcement is dated 29 July 2026. It describes a C/CUDA implementation of Minecraft 1.11.2 and a demonstration of 7,200 parallel worlds. I recovered the post text through X's public syndication endpoint after the normal page returned 403. This assessment does not treat its video as a measured benchmark on our hardware.

The current repository is substantially different. All current-code findings below refer to commit **`3b1a2551da2434d290079d083e9edecdf6186193`**, checked through the GitHub API. Its [README](https://github.com/Infatoshi/netherite/blob/3b1a2551da2434d290079d083e9edecdf6186193/README.md) targets **Minecraft 1.7.10**. Search results still show older 1.11.2 documentation, which should not be combined with the current installation instructions or performance claims.

The current README reports a 181-million-parameter policy, imitation learning followed by GRPO, and a four-GPU training benchmark. Its headline learned result concerns wooden-pickaxe acquisition. These are upstream reports, not results reproduced here, and they do not establish equivalent performance on one RTX 5080.

The dragon claim has a different meaning. [`csrc/play/chain.sh`](https://github.com/Infatoshi/netherite/blob/3b1a2551da2434d290079d083e9edecdf6186193/csrc/play/chain.sh) replays recorded action streams in checkpointed segments and compares their results against Java. This is useful evidence for the simulator's mechanics along those trajectories. It is not an evaluation of an RL policy independently solving fresh-world Survival. Nor does agreement on tested trajectories prove every possible game state is identical.

## Useful parts and integration work

| Finding | Implication for our project |
| --- | --- |
| The [C interface](https://github.com/Infatoshi/netherite/blob/3b1a2551da2434d290079d083e9edecdf6186193/csrc/runtime/rlbind.h) exposes reset, step, result polling, image buffers, and semantic observations | A Python adapter is technically plausible; we do not need to write our own game engine |
| The [semantic configuration](https://github.com/Infatoshi/netherite/blob/3b1a2551da2434d290079d083e9edecdf6186193/configs/speedrun-semantic.yaml) provides block identity, distance, and surface information without rendering RGB frames | We can investigate compact structured features before considering a large visual policy; the observation privilege must be disclosed |
| The [RL build](https://github.com/Infatoshi/netherite/blob/3b1a2551da2434d290079d083e9edecdf6186193/csrc/runtime/rlbind.mk) targets Linux/CUDA and includes CUDA runtime/toolchain assumptions | This Windows project cannot use it as an ordinary pip dependency. A Linux or WSL2 feasibility check is needed; WSL2 compatibility has not been established here |
| The [training configuration](https://github.com/Infatoshi/netherite/blob/3b1a2551da2434d290079d083e9edecdf6186193/configs/speedrun-grpo.yaml) invokes a separate `mcsr-speedrun` checkout and a checkpoint outside this repository | Cloning the public engine alone does not supply the documented complete trainer or checkpoint; availability or replacement work must be resolved |
| Our adapter targets Minecraft 1.21.4; current Netherite targets 1.7.10 | The versions and action semantics need deliberate alignment. The [pinned Mineflayer version](https://github.com/PrismarineJS/mineflayer/blob/4.39.0/lib/version.js) supports versions starting at 1.8.8, so our adapter cannot simply be pointed at a 1.7.10 server |

The interface also exposes privileged information intended for reward calculation. Keep that separate from policy observations, or explicitly redefine and document the task. Access to an API field does not automatically make it an appropriate learned-policy input.

## Bounded feasibility experiment

Before changing the main project backend:

1. Pin the revision and verify build prerequisites, dependency availability, and reuse terms. Use the user's owned Minecraft installation; do not distribute generated game assets.
2. Build and verify one environment on the available Linux/GPU setup. Record actual memory use and steps per second; increase the batch only after the small case works.
3. Implement a minimal Python adapter against the public C API. Verify reset, a few actions, actual inventory changes, episode boundaries, and reproducibility. Do not copy the external trainer's results into our own evaluation.
4. Run a small resource task with compact features and a course-level learner. Keep Q-learning/SARSA as explanatory baselines; add a small neural method only when the representation requires it. Reimplementing the upstream C/CUDA engine, large policy, GRPO, and DAgger is outside the intended learning scope.
5. Demonstrate the learned behavior in matching real Java gameplay and measure transfer failures before relying on the simulator for later Survival stages.

Stop or defer adoption if toolchain/binding work becomes the main project. The existing simulator remains useful for teaching and algorithm tests, while real-game integration remains necessary for the final demonstration. Netherite may improve training throughput and simulation fidelity; it does not remove the need to learn, integrate, and evaluate the complete progression.

This review downloaded source for inspection only. No Netherite build, GPU training, game installation, or project-backend replacement was performed.
