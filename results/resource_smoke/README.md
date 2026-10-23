# Natural-resource development diagnostics — 23 October 2026

These are genuine Minecraft development runs. **None completed the natural
wood/table/pickaxe task.** They verify real observations, online Q updates,
fresh-world isolation, and spectator learning messages, while exposing the
navigation work still required.

| Run | World seed | Recorded result | Interpretation |
| --- | --- | --- | --- |
| resource-smoke-online | 42 | 40 decisions, no wood, truncated | Early exploration/pathfinding failure; valid failure for this development revision |
| resource-smoke-watched | 1 | 40 decisions, no wood, truncated | Spectator received actual Q-update messages; bot remained on a tree canopy |
| resource-smoke-descent | 1 | Four decisions, then transport timeout | Invalid run; do not count as success or policy failure. Server log jumps from 02:30 to 03:44 during a laptop offline/sleep interval |

The final run's start was `(160.5, 76, 163.5)` on `oak_leaves`. Its last observed
position was approximately `(157.51, 75, 164.70)`, still on leaves. An adjacent
column had grass five blocks below the bot's feet. The corrected pathfinder
setting permits three-block descents; it therefore does not solve this canopy
escape. Inspect `adjacent_drops`, visible log positions, and before/after
observations in the retained JSONL.

Policies started empty, with zero simulator pretraining. Their Q-values changed
from actual game transitions. They mostly selected the only feasible action,
SEARCH; these runs do **not** show learned crafting proficiency or an improvement
over a baseline. The spectator report from the last interrupted run confirms
message reception and Spectator mode, and explicitly reports no held pickaxe.

These runs used evolving development code. `metadata.json` records source hashes
at command start. The first run was launched while the bridge was still being
edited; its recorded bridge hash is not proof of the exact bridge loaded later
at server readiness. Treat that run as diagnostic evidence only. Subsequent
runs froze bridge edits during execution. Reproduce validation on a committed
revision before reporting a benchmark.

Original JSON/CSV files were copied unchanged; `retention.json` records their
SHA-256 digests. The server log tails and final spectator report are supplementary
diagnostics. Server properties and empty operator lists are retained. Full worlds,
runtimes, consent files, and dependencies remain local under ignored directories.
No inventory, terrain, success, or failure result was fabricated.
