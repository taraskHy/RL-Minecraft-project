# Minecraft reinforcement learning course project

**Goal: start in a fresh Minecraft Java Survival world and defeat the Ender Dragon.**

**Current status: working first-milestone RL foundation. Full Survival and the dragon fight are not implemented or demonstrated.** This repository is the start of the project, not a completed Minecraft-beating model. Development arenas and simulated results do not satisfy the final goal.

The design follows the supplied course material: MDPs, epsilon-greedy exploration, tabular Q-learning, SARSA, reward design, tuning, and evaluation. Start with these understandable methods. A neural network is a later option if measured state-space or generalization problems justify it.

## Windows launcher

Double-click **`setup.cmd`**, then **`run.cmd`**. Setup installs portable Java/Node and the pinned Minecraft server inside this folder. The menu offers simulator training, experiment comparisons, a real Minecraft development-arena run, a separate Survival connection check, and diagnostics. Python 3.12 or newer is the only prerequisite. Read the [Windows setup guide](docs/windows_setup.md) for commands, world locations, and joining with your Minecraft client.

The Minecraft server requires your explicit EULA decision through `run.cmd accept-eula`. Simulator training works without it. The Survival profile provides an ordinary world for development; a full Survival policy still needs to be built.

## Run the first experiment

From this folder in PowerShell, using Python 3.12 or newer:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m minecraft_rl train --episodes 15000 --out runs/my_first_training
.\.venv\Scripts\python.exe -m minecraft_rl demo --policy runs/my_first_training/policy.json
```

The Python core uses only the standard library. It needs no GPU, Minecraft installation, downloaded training data, or Python packages. A local `.venv` has already been created during project setup; recreating it is unnecessary on this machine. Optional installation as a command is `python -m pip install -e .`.

Training saves a readable JSON Q table, CSV episode records, validation metrics, configuration/provenance, and an SVG learning curve. Choose a new output directory for each run; existing results are never silently overwritten. `demo` prints the state/action trace of the **simulator**.

Compare algorithms, rewards, and two learning rates across three independent training seeds:

```powershell
.\.venv\Scripts\python.exe -m minecraft_rl experiment --config configs/smoke.json --out runs/my_comparison
```

The larger planned sweep is:

```powershell
.\.venv\Scripts\python.exe -m minecraft_rl experiment --config configs/course_experiment.json --out runs/course_experiment
```

It varies learning rate, discount, and exploration decay across five training seeds. The smaller run establishes that the pipeline works; it does not by itself fulfill the entire course's experimental requirement. See [the experiment protocol](docs/experiment_protocol.md) before interpreting results or changing test seeds.

The [first measured comparison](results/baseline/README.md) is included: SARSA achieved 99.83% simulator task success across 600 evaluation episodes, versus 0.17% for random actions. All four learned conditions and their training-seed variation are retained. Try the included simulator policy immediately:

```powershell
.\.venv\Scripts\python.exe -m minecraft_rl demo --policy results/baseline/demo_policy.json
```

## What is implemented

| Component | Current behavior |
| --- | --- |
| Crafting simulator | A 5x5 grid with two collectible logs, a lava hazard, a supplied workbench, randomized safe starts, and stochastic movement |
| Learned decisions | Four movement choices, chopping, crafting planks, crafting sticks, and crafting a wooden pickaxe; no hardcoded solution path or action mask |
| RL algorithms | Q-learning and SARSA implemented directly, with readable one-step updates and random tie-breaking |
| Comparisons | Random baseline, sparse task reward versus potential-based shaping, validation selection, held-out episode seeds, multiple training seeds |
| Tests | Resource accounting, transition rules, reward shaping, TD targets, reproducibility, frozen evaluation, and serialization |
| Optional game bridge | Mineflayer development-arena adapter with a verified live wooden-pickaxe run, managed local servers, and a working Survival connection probe |
| Full Survival system | Planned; natural-world resource discovery, workbench creation, equipment/survival, Nether progression, stronghold search, and dragon combat remain to build |

The optional adapter is a **development tool**. It builds/resets an arena using server commands and a provided crafting table. Those conveniences must not appear in the final fresh-world Survival run. See [Minecraft bridge setup](minecraft_bridge/README.md). Runtime installation and live-game verification are separate: setup downloads the server but never accepts its EULA or configures your game account.

[Real-game integration evidence](results/game_smoke/README.md) is now retained: a 15-decision wooden-pickaxe run, a successful repeat with a 2 GB server heap, blocked-movement/lava/reset checks, and an ordinary Survival connection. These are milestone checks, not full Survival or Ender Dragon results.

## Read and learn in this order

1. [Course requirements and schedule](docs/course_requirements.md): the rubric, every supplied source, and the plan through **31 December 2026**.
2. [Learning guide](docs/learning_guide.md): the concepts, worked updates, lecture pages, and exercises for everyone in the group.
3. [Environment specification](docs/environment.md): exactly what the first agent sees, chooses, and earns.
4. [Full Survival development plan](docs/full_survival_plan.md): the remaining engineering, learned/scripted boundaries, and evidence required for the final demo.
5. [Report and presentation plan](docs/report_plan.md): how to turn actual results into the required submissions.

The course requires an **8–20 page report** (Arial 12, 1.5 spacing, one-inch margins; references excluded), a **15-minute presentation**, and contributions from every team member. The user supplied the 90% project / 10% presentation grade split.

## Code map

```text
minecraft_rl/
  environment.py    State, actions, recipes, transitions, reward
  agents.py         Q-learning / SARSA update and policy files
  training.py       Exploration, training, frozen evaluation, metrics
  experiments.py    Validation-based configuration selection and testing
  artifacts.py      CSV, JSON, provenance, and learning curves
  live.py           Optional Python-to-Minecraft adapter
  __main__.py       Commands
minecraft_bridge/   Node/Mineflayer bridge, setup, protocol checks
infrastructure/     Portable runtime setup, local server lifecycle, PC launcher
setup.cmd / run.cmd Windows setup and menu / command launchers
configs/            Small and larger experiment specifications
tests/              Automated checks
docs/               Requirements, learning material, plan, and methods
results/            Small retained baseline summaries and example policy
runs/               Local generated runs (ignored by Git)
```

The project intentionally keeps the starting algorithm small. Development currently runs on a laptop; the user plans to move to a 64 GB desktop and later an RTX 5080 machine. These resources do not remove Minecraft's exploration and integration challenges. The team should verify the code, explain the learning updates, keep real contribution records, and disclose AI assistance according to its course's policy. Original course PDFs/DOCX and temporary extractions are not copied into version control.
