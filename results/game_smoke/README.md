# Real Minecraft integration checks — 23 October 2026

These runs used the actual Minecraft Java 1.21.4 server and Mineflayer on the user's Windows laptop, after the user explicitly accepted the server EULA. They validate the local infrastructure and the small development arena. **They do not demonstrate full Survival, an Ender Dragon victory, or a measured generalization success rate.**

The included frozen SARSA policy selected 15 actions and produced one real wooden pickaxe in the development arena. Two logs were physically dug and collected. A repeat with the new **2 GB maximum Java heap** also succeeded in 15 decisions. Both runs used the same start seed and saved policy, so they are integration repetitions, not independent training seeds.

Separate scripted diagnostics confirmed that an unbroken log blocks movement, entering the lava cell produces observed health zero and terminal death, and a development reset works after death. These diagnostics deliberately control the test start and are not learned behavior. The provided workbench, arena construction, and reset commands are development aids.

The ordinary Survival server also passed its connection check. The retained final probe contains three complete observations with an empty inventory, Survival gamemode, and health/food 20. It issued no movement or progression commands. The world was fresh on its first check and reused on the final check. An early probe emitted its initial snapshot before health/food arrived; the code now waits for complete data, and the final retained probe was collected after that fix.

| File | Evidence |
| --- | --- |
| [summary.json](summary.json) | Conditions, scope, source run IDs, and retained-file checksums |
| [arena_trajectory.csv](arena_trajectory.csv) | First 15-decision frozen-policy trajectory |
| [arena_outcome.json](arena_outcome.json) | Actual final inventory, position, health, and success observation |
| [arena_integration_observations.json](arena_integration_observations.json) | Blocked movement, lava death, reset, and repeated policy observations with a 2 GB heap |
| [survival_observations.jsonl](survival_observations.jsonl) | Complete ordinary-world observations and connection completion event |
| [survival_result.json](survival_result.json) | Probe exit status and its limited evidence statement |

All servers saved their worlds and stopped normally. The automated suite also passed **63 tests**. Simulator results remain separate in [the baseline directory](../baseline/README.md).

To repeat the normal live milestone, run `run.cmd arena --dev-arena`. To repeat the diagnostic checks, run `.venv\Scripts\python.exe -m infrastructure.check_arena --dev-arena` from the project root. The diagnostic uses the same dedicated arena server and saves a new report in `runs/`. See [Windows setup](../../docs/windows_setup.md) before running any arena command.
