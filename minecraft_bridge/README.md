# Real Minecraft development bridge

This optional bridge runs the **first wooden-pickaxe experiment inside Minecraft Java**. It is infrastructure for a milestone, not an implementation of fresh-world survival or an Ender Dragon victory. The supplied crafting table and controlled map are development aids. The final fresh-survival evaluation must start with an empty inventory and ordinary world, and must never invoke this arena reset.

**Verification status:** a frozen policy completed a live Minecraft 1.21.4 development-arena run on 23 October 2026: two logs were collected and a wooden pickaxe was observed in inventory after 15 decisions. The separate Survival probe connected to an ordinary world with an empty inventory. Automated dependency, protocol, and observation tests also pass. These are integration checks, not a Minecraft success-rate benchmark or evidence of full Survival completion. See [Windows verification details](../docs/windows_setup.md#verification-on-23-october-2026).

## Requirements

For automatic Windows setup, use **`setup.cmd`** in the repository root, then **`run.cmd arena --dev-arena`** after recording your own EULA decision with **`run.cmd accept-eula`**. The [Windows guide](../docs/windows_setup.md) describes the managed profiles. The manual requirements below are useful if running the bridge independently.

- Node.js **22 or later** and **pnpm 11.25.0**; `mineflayer` is pinned to **4.39.0** and `vec3` to **0.1.10**. The included `pnpm-lock.yaml` pins transitive packages.
- Minecraft **Java Edition 1.21.4**, running separately on the same computer, with Java **21**.
- A **dedicated disposable Overworld server**, not a world containing builds you care about.
- For this local development adapter, the server uses offline authentication and the bot is `RLCourseBot`. Bind the server to `127.0.0.1` with `server-ip=127.0.0.1`; use `online-mode=false`, `spawn-protection=0`, and `allow-flight=true`. Do not expose this test server to the network. The bridge only accepts localhost destinations.
- In the dedicated server console, `op RLCourseBot` authorizes arena resets. The adapter itself opens no HTTP port.

For manual setup, download and configure the server using the [official Minecraft server instructions](https://www.minecraft.net/en-us/download/server). The low-level bridge does not download or start a server; the separate Windows launcher handles those steps. Neither setup nor the bridge accepts the EULA automatically. Use version **1.21.4**, not whichever version the download page currently advertises.

From the repository root in PowerShell:

```powershell
cd minecraft_bridge
pnpm install --frozen-lockfile --ignore-scripts
pnpm test
pnpm smoke
node --check bridge.js
cd ..
python -m unittest discover -s minecraft_bridge -p "test_live.py"
```

Install pnpm 11.25.0 if it is not already available, then use the lockfile command above on each teammate's machine. Dependency lifecycle scripts are disabled; the bridge's required imports were checked with this installation mode. Node alone is not sufficient for installing dependencies. The Python simulator has no Node dependency.

## Explicit development-world changes

Only `--dev-arena` / `dev_arena=True` enables reset. At the default origin `(0,64,0)`, reset replaces blocks in **x=-1..5, y=62..68, z=-1..5**. It builds a two-layer stone floor that contains lava from below, barrier walls/roof, two oak logs, one lava cell, and a crafting table. Every block in this region is checked against the expected arena before reset succeeds. It removes item drops within the arena, clears the bot's inventory/effects, restores initial health/food, briefly uses creative mode during construction, then switches the bot to survival and teleports it to a sampled start cell. It also disables `doFireTick` and `doMobSpawning` **for the entire test world**; those settings are not restored automatically.

These commands belong only to episode setup in the dedicated development world. During a step, movement uses normal controls, logs are actually dug and picked up, and crafting uses actual inventory recipes. No movement teleport, artificial inventory increment, or success command is used. A failed move, pickup, recipe, connection, or reset raises an error and invalidates the episode; it is never replaced with simulated progress.

After crafting, the bridge requests a server inventory synchronization through Mineflayer 4.39.0's private `_syncWindow` helper before checking results. This avoids treating the library's optimistic local crafting updates as confirmed inventory. The helper is version-specific; changing Mineflayer requires rechecking that contract.

A successfully crafted wooden pickaxe is automatically equipped in the bot's main hand so spectators can see it. This uses ordinary inventory controls and adds no learned action or reward. Observations include `held_item` (the item name, or `null`); success still depends on the actual inventory.

## Python API

```python
from minecraft_rl.environment import Action, EnvConfig
from minecraft_rl.live import MinecraftEnv

# Run only while the dedicated local test server is running.
with MinecraftEnv(EnvConfig(max_steps=100), dev_arena=True) as env:
    state = env.reset(seed=42)
    next_state, reward, terminated, truncated, info = env.step(Action.NORTH)
    print(next_state, info["position"], info["inventory"])
```

The constructor accepts `host`, `port`, `username`, `node`, `origin=(0,64,0)`, and `timeout=45`. It connects to the existing server. `reset(seed)` returns the same six-field state used by the simulator; `step(action)` returns `(state, reward, terminated, truncated, info)`. `close()` disconnects the bot; context-manager use is preferred.

`observe()` returns a validated current arena observation without taking an action. `players()` returns connected names and their server-reported numeric gamemodes (`3` is spectator); invisible spectator entities are not required. The launcher uses this read-only player list to wait for a viewer in its optional watch mode.

State: `(x,z,remaining_log_mask,logs,planks,sticks)`. Actions: `NORTH=0`, `SOUTH=1`, `EAST=2`, `WEST=3`, `CHOP=4`, `CRAFT_PLANKS=5`, `CRAFT_STICKS=6`, `CRAFT_PICKAXE=7`. North reduces z. Chop targets an adjacent remaining log, collects its real item, and returns to the original cell. The physical crafting table lies just outside the grid beside station `(4,4)`.

Rewards reuse the simulator's base reward and potential function. `slip_probability` is **not** injected into Minecraft; the real movement controller and server have different dynamics. Individual live actions take different wall-clock durations, while this first experiment discounts per decision. Death can destroy inventory, which also differs from the simplified simulator's terminal observation. Report live evaluation separately and do not assume a transferred table will work without testing.

## JSON-lines protocol

For direct inspection, with an existing server:

```powershell
node minecraft_bridge/bridge.js --dev-arena --host 127.0.0.1 --port 25565 --username RLCourseBot --origin 0 64 0
```

Send one request per line:

```json
{"id":1,"op":"status"}
{"id":2,"op":"reset","start":[0,0]}
{"id":3,"op":"step","action":2}
{"id":4,"op":"observe"}
{"id":5,"op":"close"}
```

Responses contain the same `id`, `ok`, and either `result` or `error`. Observations include actual `position`, `inventory`, `health`, `success`, `death`, and `backend="minecraft-java-development-arena"`. Diagnostic messages go to stderr. Without `--dev-arena`, `status` is available but reset is rejected. Requests are sequential and bounded; a 40-second operation timeout closes the session.

## First game verification

1. Check `status` and an explicit reset on the disposable server. Confirm the observed world, inventory, gamemode, and sampled start match the response.
2. Execute one movement at a time and compare the physical coordinates. Verify an occupied log cell blocks movement.
3. From beside each log, call chop and confirm actual inventory increments and the bot returns to its prior cell.
4. Craft planks twice, sticks once, navigate to `(4,4)`, and craft the wooden pickaxe. Save the complete response log and a game recording.
5. Check a lava episode really ends in death. Then evaluate the learned policy with a fixed episode budget, reporting errors separately from valid failures and successes.

Only after these pass should this backend be used for learning curves. A successful game smoke test would verify the **wooden-pickaxe arena milestone only**. Subsequent fresh-world skills still need development and end-to-end evaluation.

## Primary sources checked

- [Mineflayer 4.39.0 package requirements](https://github.com/PrismarineJS/mineflayer/blob/4.39.0/package.json): Node >=22 (the README's older >=18 statement is not the package requirement).
- [Mineflayer tested Minecraft versions](https://github.com/PrismarineJS/mineflayer/blob/4.39.0/lib/version.js): includes 1.21.4.
- [Mineflayer API](https://github.com/PrismarineJS/mineflayer/blob/4.39.0/docs/api.md): physical control, digging, inventory, crafting, and game state.
- [Minecraft Java 21 requirement](https://www.minecraft.net/en-us/article/minecraft-java-edition-1-20-5): introduced in 1.20.5.
