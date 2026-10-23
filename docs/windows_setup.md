# Run the project on a Windows PC

The launcher manages local dependencies, simulator experiments, a Minecraft development arena, and a separate ordinary Survival server. **It does not yet implement an agent that completes full Survival or defeats the Ender Dragon.** The Survival connection check only observes a real bot; it is not a trained policy.

## First setup

1. Keep the complete project folder on your PC. Install **64-bit Python 3.12 or newer**, including the Python launcher, if it is missing: [official Windows downloads](https://www.python.org/downloads/windows/).
2. Double-click **`setup.cmd`**. It creates `.venv` if needed, downloads project-local Node 24, Java 21, pnpm, and the official Minecraft **1.21.4** server, verifies the pinned download hashes, and installs the locked bridge dependencies. Internet access is required the first time. Nothing is installed globally; no administrator access is needed.
3. Double-click **`run.cmd`** for the menu. Start with **2: saved simulator policy** or **3: train a simulator policy**. You can use these without downloading Java/Node or accepting Minecraft's server EULA.
4. Before starting Minecraft, choose **5: read / accept the Minecraft server EULA**. Read the linked agreement and enter `I AGREE` only if you agree. Setup leaves `eula=false` until you make this choice. The decision is saved inside this checkout, not in a global setting.
5. Choose **6: development arena** to start its dedicated server, run the included policy through the real game bridge, save the observed trajectory, and stop the server. This is a wooden-pickaxe development task and resets its small test region.

Windows may ask about downloaded files. The launchers use PowerShell's execution-policy override for their own process only; they do not change your system policy. A managed corporate policy can still prevent execution.

## Install the graphical Minecraft client and connect

The tested client version is **Minecraft: Java Edition 1.21.4**, using the ordinary unmodified release. The project setup has installed its own server; the Minecraft Launcher downloads the separate graphical client.

1. In Minecraft Launcher, select **Minecraft: Java Edition**.
2. Open **Installations**, then **New installation**. Name it `RL Project`, select **release 1.21.4**, and create it.
3. Select `RL Project` on the Play tab and click **Play** to download and launch it. Leave the launcher's Java settings at their default. No Forge, Fabric, or other mods are required.
4. On this laptop, project setup and the EULA decision are already complete. Open a terminal in the project folder and run `run.cmd server survival`. On a new PC, run `setup.cmd` first and record the EULA decision there.
5. Once the terminal reports that the server is ready, choose **Multiplayer → Direct Connection** in Minecraft and enter **`127.0.0.1:25566`**. Keep the server terminal open while playing; **Ctrl+C** saves and stops it.

This connects your client to the project's ordinary Survival world. The current learned behavior is the separate crafting arena: `run.cmd arena --dev-arena` runs that bot episode and saves its trajectory, using port **25565**. That short automatic test stops its server when finished; it is not a persistent Survival bot. The [Minecraft profile table](#minecraft-profiles) explains the distinction.

Launcher installation steps are described in [Minecraft's installation help](https://help.minecraft.net/hc/en-us/articles/23431114561037-Troubleshooting-Launching-Minecraft-from-the-Minecraft-Launcher); version details are in the [official 1.21.4 release notes](https://www.minecraft.net/en-us/article/minecraft-java-edition-1-21-4).

## Watch the crafting bot

The ordinary Survival server has no autonomous progression agent yet. To watch the implemented crafting policy in its separate development arena:

1. Disconnect from the Survival world. In its server terminal, press **Ctrl+C** and wait for it to save and stop.
2. In the project terminal, run the following, replacing `YourMinecraftName` with your Java player name:

   ```powershell
   .\run.cmd watch --dev-arena --viewer YourMinecraftName
   ```

3. Wait for the join instructions, then select **Multiplayer → Direct Connection → `127.0.0.1:25565`** in Minecraft 1.21.4.
4. The watcher places your player in **Spectator mode**, adds night vision, and moves your camera above the arena. After a short countdown, the frozen policy collects logs, crafts a wooden pickaxe, and equips it in its hand. The bot keeps holding the tool while the terminal waits for your replay choice; you can fly closer to see it. Spectator mode prevents your player from moving the bot or collecting its dropped items.
5. The terminal reports the outcome and saved result folder. Press **Enter** there to replay, or enter **q** to save and stop. **Ctrl+C** also stops the managed arena server.

You can also choose **11** in the `run.cmd` menu and enter your player name. The watch command waits up to five minutes for you to join; `--join-timeout` changes that limit up to ten minutes. `--once` stops the server after one episode instead of offering replay.

Viewer controls apply only to the dedicated development server. They do not grant operator privileges or change the ordinary Survival world. The arena is reset between episodes; the provided crafting table and reset commands remain development aids, not part of the final fresh-world Survival evaluation.

## Commands in PowerShell

For the next stage, including visible online learning in freshly generated worlds,
see [watch natural-resource training](resource_training.md). Menu **12** runs it
on **127.0.0.1:25567**; menu 11 remains the frozen arena demonstration.

Run these from the project folder. The same commands also work in a normal Windows terminal; activation of `.venv` is unnecessary.

```powershell
.\run.cmd doctor
.\run.cmd demo
.\run.cmd train --algorithm sarsa --episodes 15000
.\run.cmd experiment --config configs/smoke.json
.\run.cmd test
.\run.cmd accept-eula
.\run.cmd arena --dev-arena
.\run.cmd survival-check --seconds 10
.\run.cmd server survival
```

`setup.cmd` pauses when finished so double-click users can read the result. For unattended setup, use `run.cmd setup`. Relative paths in command options are resolved from the project root. Output folders get unique timestamps by default, and the launcher prints the destination. Use `--out runs/my_run` to choose a new directory yourself.

The saved simulator policy is `results/baseline/demo_policy.json`. Training prints the path to a new `policy.json`; pass that file to a later demo or arena run:

```powershell
.\run.cmd demo --policy runs/YOUR_TRAINING_FOLDER/policy.json
.\run.cmd arena --dev-arena --policy runs/YOUR_TRAINING_FOLDER/policy.json
```

The simulator uses a readable Q table, Q-learning/SARSA, and the CPU. The RTX 5080 is not used by these algorithms. CUDA, Docker, WSL, neural-network packages, and Netherite are not prerequisites. The [Netherite assessment](netherite_assessment.md) explains the separate research work needed for that optional backend.

## Laptop now, desktop later

The current plan is to develop on the laptop, later use the 64 GB desktop (GPU model still to be confirmed), and eventually use the RTX 5080. The current training code and headless Minecraft server use the CPU; no NVIDIA card or CUDA installation is required for these stages.

The server starts with a **512 MB Java heap and a 2 GB maximum**. This is a heap limit, not a total-process memory limit; Java, Node, Python, and an optional graphical Minecraft client also use memory. Start only one server at a time while developing on the laptop. `run.cmd doctor` shows the configured heap.

If more server heap is needed on a later machine, create `.local/machine.json` with:

```json
{"server_memory_mb": 4096}
```

The setting applies on the next server start. Values must be integers from 1024 to 32768 MB. It is local to each checkout and ignored by Git; there is no need to allocate most of a 64 GB machine's memory to this small server.

When moving machines, copy or clone the source project, run `setup.cmd` there, and bring any trained policy folders from `runs/` that you want to keep. Do not copy `.venv`, `node_modules`, or portable runtimes. Saved Q tables are ordinary JSON and are independent of the GPU. If continuing an existing world, stop its server before backing it up; otherwise the destination setup creates separate new worlds. We will detect the exact NVIDIA model on that machine before choosing any later neural-network dependencies.

## Minecraft profiles

| Profile | Local address | World and behavior |
| --- | --- | --- |
| `arena` | `127.0.0.1:25565` | Dedicated development world. `RLCourseBot` is an operator so the explicit arena reset can build its test region. |
| `survival` | `127.0.0.1:25566` | Separate ordinary Survival world, random seed on first creation, empty operator list. No supplied resources, arena, or progression commands. |

Both profiles bind only to the local computer and use offline authentication for local bots. They are not multiplayer hosting configurations. The Survival world is fresh **on its first start** and preserved on later starts; restarting a server does not reset a world. A future evaluation harness must create independently seeded fresh worlds for its final trials.

`arena --dev-arena` owns the server for one episode and stops it afterward. `survival-check` similarly starts the Survival server, connects `SurvivalProbe` for a short observation log, disconnects, and stops the server. It sends no gameplay commands and makes no learning updates. A death or connection error fails the check. It preserves the bot's actual game state on disk.

Use `server arena` or `server survival` to leave a server running for inspection. **Ctrl+C saves and stops it.** Wait for the shutdown to finish. Do not close the console forcibly while it is saving. An occupied port is reported rather than killing another process. Do not run `arena` while a manually launched arena server is already using port 25565.

To see a running world, launch your owned **Minecraft Java Edition 1.21.4** client, choose Multiplayer → Direct Connection, and enter the profile's address. The scripts install the server, not the graphical game or your account. A connection-only check may finish before you join; use `server survival` for a persistent session. The automatic arena run is brief. A game client is optional for headless bot experiments.

## Local files and recovery

| Path | Contents |
| --- | --- |
| `.venv/` | Project Python environment |
| `.local/runtime/` | Portable Node, Java, and pnpm |
| `.local/downloads/` | Verified download cache |
| `.local/server/` | Pinned Minecraft server jar |
| `.local/servers/arena/` | Development server properties, world, logs |
| `.local/servers/survival/` | Survival server properties, world, logs |
| `.local/minecraft-eula-consent.json` | Your explicit local EULA decision |
| `.local/machine.json` | Optional per-machine Java heap override |
| `runs/` | Timestamped training, arena trajectories, Survival connection observations |

All of these local runtime/world/run folders are ignored by Git. Share the source, lockfile, configs, and selected results with teammates; each teammate runs setup on their machine. Do not copy `.venv` between machines.

Setup can be repeated: verified downloads are reused and worlds are retained. A corrupted cached download is rejected. The server manager refuses unexpected server folders or incompatible configuration rather than adopting somebody else's world. Keep backups before intentionally editing or replacing worlds. The launcher never deletes a world.

If setup fails, read the named download/runtime error and rerun `run.cmd setup` after fixing network access or disk space. If Python is missing, install it from the official link above. If a server cannot start, check the log path printed by the launcher. `run.cmd doctor` checks runtime availability and bridge imports; `run.cmd test` runs automated checks. Neither proves that a game episode succeeds. Live `outcome.json` and the observed trajectory provide that separate evidence.

## Verification on 23 October 2026

The actual portable runtimes and server jar were downloaded and their pinned hashes checked. Repeated setup, all **63 automated tests**, a 15,000-episode simulator training run, the saved-policy demo, and a new virtual environment in a folder containing spaces passed. Launching from outside the project folder and running diagnostics with only Windows directories on PATH also passed. The Survival probe correctly failed with a clear connection error against an unused port. Server lifecycle tests used controlled test subprocesses to check readiness, occupied ports, startup failure, and graceful shutdown.

After the user explicitly accepted the EULA, the real arena policy crafted a wooden pickaxe in **15 decisions**. A repeat with the **2 GB heap** also passed, together with blocked movement, actual lava death, and reset-after-death checks. The ordinary Survival probe connected with an empty inventory and logged complete health, food, and position observations. Each server saved and stopped normally. The evidence is retained in [results/game_smoke](../results/game_smoke/README.md).

Acceptance is already recorded on this laptop. New checkouts still require an explicit local decision. These results verify the development arena and the ordinary-world connection; **full Survival progression and the Ender Dragon agent remain unimplemented**.
