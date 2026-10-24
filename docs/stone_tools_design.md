# Stone-tools milestone: design

**Status.** Design accepted: on 24 October 2026 the user accepted all 12 recommendations in section 11. Offline step S1 (section 9) is under way on branch `codex/stone-tools`; no stone-stage episode has been run. Numbers marked *computed* were computed on 24 October 2026 from retained files under `results/pc_resource_runs/`. Numbers marked *estimate* are guesses; measurements replace them. Line numbers refer to commit `2618695` on `codex/pc-canopy-descent`; the initial-Q fixes in `cb63ef2` moved a few lines in `minecraft_rl/resources.py` and `infrastructure/resource_session.py`. Two of the computed figures were re-checked independently from the retained files before this was committed: 114 of the 124 held-out successes ended with 0 logs, 3 planks and 2 sticks, and a stone-tool material was under or beside the bot in 34 of the 400 `87673b8` episodes. `minecraft_bridge/resource-bridge.js` has not changed since `529bf7d`.

This document merges three independent designs: execution helpers and safety, the learning problem, and evidence and schedule. Section 12 lists where they disagreed and what was decided.

**Assumptions.** These are team defaults that the user has not confirmed.
- **A1.** Learning is reported only without a feasibility mask. At `87673b8`, masked random play succeeded 19/20, so a masked comparison cannot show learning (`results/pc_resource_runs/README.md`).
- **A2.** The state is plain: current observations only, with no `last_action_failed` bit.
- **A3.** Later stages follow a recipe dependency graph from `minecraft-data`. Each stage has its own small tabular learner, and the stage transitions are programmed (`docs/full_survival_plan.md:7`, `:28`).
- **A4.** The stone learner's initial Q is chosen after tonight's initial-Q experiment (`configs/initial_q_experiment.json`, 0 against −1), by the rule in section 3.5.

---

## 1. Goal and definition of done

**Goal.** The bot starts in a fresh Survival world with an empty inventory and ends holding a stone pickaxe. It crafts the pickaxe at its own crafting table from 3 stone-tool materials and 2 sticks. It mines the materials itself with its wooden pickaxe. No server commands, supplied items, teleports or creative mode are allowed, in training or in testing.

**Facts checked in the pinned `minecraft-data` 3.117.0 (`pc/1.21.4`).** These come from the data files only, not from a running game.

| Fact | Value |
| --- | --- |
| Stone pickaxe recipes (item 856) | Exactly three, all 3x3, so a table is required. The top row is 3 cobblestone, 3 cobbled deepslate or 3 blackstone; 2 sticks go down the middle. |
| Blocks that give a recipe material with a wooden pickaxe | `stone` (hardness 1.5) and `cobblestone` (2.0) drop cobblestone. `deepslate` (3.0) and `cobbled_deepslate` (3.5) drop cobbled deepslate. `blackstone` (1.5) drops itself. |
| Pickaxe blocks that drop themselves and do not count | granite, andesite, diorite, tuff, calcite, sandstone, terracotta, mossy cobblestone |
| `infested_stone` | Hardness 0.75. It drops nothing and spawns a silverfish. |
| Wooden pickaxe | 59 durability; mining speed 2 on `mineable/pickaxe` |
| Stone pickaxe | 131 durability |
| Iron and copper ore | Material `incorrect_for_wooden_tool`; they need a stone pickaxe or better |
| Furnace (next stage) | 8 of one stone-tool material |

**Not verified.** Each of these must be tested, not assumed.
- **Mixed materials.** It is not known whether the server accepts a grid that mixes materials, for example 2 cobblestone and 1 cobbled deepslate. I expect it does, because vanilla matches a tag in each slot. But `minecraft-data` lists only single-material variants, so `bot.recipesFor` will not find a mixed recipe.
- **Dig times.** These come from the vanilla formula; check them against `bot.digTime`.
  - stone: about 1.15 s with the wooden pickaxe, or 7.5 s by hand with no drop;
  - cobblestone: about 1.5 s;
  - deepslate: about 2.25 s;
  - dirt: about 0.75 s by hand; grass block: about 0.9 s by hand;
  - about 5 times slower when the bot is off the ground.
- **Durability.** It is not known whether a tool loses durability on dirt, grass and leaves. I believe it loses 1 for every block with non-zero hardness.
- **Reading durability.** It is not known whether `prismarine-item` 1.18.0 reports `durabilityUsed` for 1.21.4 items. The getter reads the `damage` component (`index.js:369-386`).
- **Depth to stone.** How deep the dirt is above stone, per biome, is unknown.
- **Time of day.** I expect morning at the first spawn, with dusk about 10 real minutes later.

**Definition of done.** The milestone is done when all of the following hold.
1. The stone stage is implemented and every offline test in section 7 passes. This includes the updated static test in `resource-bridge.test.js`.
2. The scripted reference passed the helper gate on the validation worlds (section 7.6).
3. The pre-registered experiment in section 8 ran on one committed revision. Its runs, reports and README are retained under `results/stone_runs/<commit>/`, with digests.
4. The three claims below are reported, whatever the result.
   - **C1. The milestone is verified live.** For at least one algorithm, at least 2 of its 3 frozen checkpoints succeed end to end on at least 15 of the 20 held-out worlds. The Wilson 95% interval for 15/20 is 0.53–0.89 (computed). If only the scripted reference reaches 15/20, the claim reads "milestone verified with a scripted stone stage".
   - **C2. Learning beats random play at the stone stage.** For each checkpoint, the Wilson interval of its stage-2 success must lie entirely above the uniform random arm's. An algorithm-level claim needs 2 of its 3 checkpoints to pass.
   - **C3. The scripted reference.** It is reported descriptively, with no hypothesis test.
5. `TEAM_STATUS.md` and the stage table in `docs/full_survival_plan.md` are updated, committed and pushed.

**Why C1 needs 2 of 3 checkpoints.** Picking the best of six checkpoints after seeing the test results inflates the chance of passing. Computed from the binomial: if every checkpoint's true end-to-end rate were 0.6, one checkpoint would reach 15/20 with probability 0.13, and at least one of six with probability 0.55. The 2-of-3 rule gives 0.04.

---

## 2. What is learned and what is scripted

| Part | Kind | Notes |
| --- | --- | --- |
| Wood stage (actions 0–7) | Learned earlier, frozen | The `87673b8` unmasked Q-learning checkpoint, run at epsilon 0 and never updated (section 5) |
| Switch to the stone stage | Programmed | Happens on an observed, held wooden pickaxe; the recipe preconditions are checked |
| Which stone macro action to take | **Learned** | Tabular Q-learning or SARSA over 5 actions and at most 48 live states |
| Finding visible stone and walking to it | Scripted | `SEARCH_STONE` |
| Digging a staircase through dirt and through rock that is not a recipe material | Scripted (new) | `DIG_DOWN` |
| Mining one stone block and picking up the drop | Scripted | `MINE_STONE` |
| Walking back to the table | Scripted | `RETURN_TO_TABLE` |
| Recipe, crafting, equipping | Scripted | `CRAFT_STONE_PICKAXE` |
| Safety checks that read blocks the bot cannot see | Scripted | They only ever refuse; disclosed (section 4.3) |

**What the report may claim.** Learning chose among five stone macro actions, after a frozen learned wood policy. It must not claim learned motor control, digging, navigation or crafting. The learner sees decisions, not seconds (`docs/resource_training.md:487-491`). Plan4MC's 0.17 on `stone_pickaxe`, which started from a given wooden pickaxe (`docs/report_plan.md:47`), is related work and not comparable.

---

## 3. State, actions and reward

### 3.1 Actions

The stone learner has five actions, all allowed in every state, with no mask.
- Bridge indices 8 to 12 are appended to `ACTIONS` (`resource-bridge.js:8`), and `validateAction` (`:367-370`) is updated to match.
- Indices 0 to 7 keep their meaning, so the frozen wood checkpoint still loads; `ResourceAgent.load` checks the action names (`resources.py:323`).
- A Python `StoneAction` enum, 0 to 4, maps to bridge indices 8 to 12.

| Stone action | Bridge | Useful when (state) | What the helper does | Budget |
| --- | --- | --- | --- | --- |
| `SEARCH_STONE` | 8 | `nearby_stone=0` | Walks to visible stone; if there is none, explores (section 4.6) | 36 s |
| `DIG_DOWN` | 9 | `nearby_stone=0` | Digs a staircase until stone is in reach, or until its budget runs out (section 4.4) | 30 s, at most 6 steps |
| `MINE_STONE` | 10 | `nearby_stone=1` and `stone_items<3` | Mines one stone block in reach and picks up the drop (section 4.5) | dig time + 3 s, then 9 s for the pickup |
| `RETURN_TO_TABLE` | 11 | `nearby_table=0` | The `APPROACH_TABLE` procedure (`:1014-1028`) with a 36 s deadline and a first navigation of up to 24 s; never digs. Action 6 stays unchanged for the wood stage. | 36 s |
| `CRAFT_STONE_PICKAXE` | 12 | `stone_items=3` and `nearby_table=1` | Crafts, syncs the inventory, equips the pickaxe and checks that it is held (section 4.8) | about 5 s (estimate) |

**Events** (the step's `event` string):
- `SEARCH_STONE`: `stone_already_reachable`, `stone_reached`, `stone_path_blocked`, `explored`, `explored_locally`, `exploration_blocked`, `exploration_time_budget`, `exploration_boundary`, plus the existing canopy and foliage notes.
- `DIG_DOWN`:
  - `stair_<n>_steps;stone_exposed` or `stair_<n>_steps;time_budget`;
  - `stair_refused:<reason>`, with every heading's reason in `action_detail`;
  - `stair_stopped:<cave_below|liquid_revealed|took_damage|hostile_nearby|depth_limit|dig_unconfirmed|move_failed>`;
  - `stone_already_reachable`, `stepped_off_table`, `moved_along_surface`.
- `MINE_STONE`: `cobblestone_collected`, `cobbled_deepslate_collected`, `blackstone_collected`, `stone_broken_without_pickup`, `no_reachable_stone`, `stone_unreachable`, `no_pickaxe`, `action_failed: …`.
- `RETURN_TO_TABLE`: `table_reached`, `table_not_visible`, `table_path_blocked`, `no_own_table`.
- `CRAFT_STONE_PICKAXE`: `crafted_stone_pickaxe`, `not_at_own_table`, `missing_materials`, `craft_not_confirmed`. If the equip fails, it raises an `ObservationError` (an invalid episode), as the wooden pickaxe equip does (`:952`).

**Refusals.** An action taken where it is not useful is an ordinary observed transition: it takes its real time, costs the step penalty, and changes nothing.

**Kept out of the action set:**
- **The 8 wood actions.** Including them would inflate random play's waste, so the learning gap would be artificial.
- **A second table.** It needs 4 planks, and no successful hand-off had more than 3 planks, apart from two random-arm runs (computed).
- **Carrying the table.** Left for v2.
- **Mining 3 blocks per decision.** That would hide the stop decision.
- **A learned dig heading.** Without X-ray information, no state field tells the headings apart.

### 3.2 State `stone_v1`

| Field | Values | Definition |
| --- | --- | --- |
| `stone_items` | 0–3 | Cobblestone + cobbled deepslate + blackstone in the inventory, capped at 3 |
| `nearby_stone` | 0/1 | `MINE_STONE` has at least one candidate; the observation and the action use the same function |
| `stone_visible` | 0/1 | `SEARCH_STONE` has a target it has not recently failed to reach; set to 1 whenever `nearby_stone` is 1 |
| `nearby_table` | 0/1 | The own table is in reach and visible (`tableInReach`, `:550-553`) |
| `health_bin` | 0/1/2 | 0, 1–10 or more than 10 points, as in `resources.py:153`; 0 is terminal |

**Size.** 4 × 2 × 2 × 2 × 3 = 96 nominal states, 64 of them with health above 0. Excluding `nearby_stone=1` with `stone_visible=0` leaves 48 consistent live states, so the table holds at most 240 Q values. For comparison, the unmasked wood learner visited 180 states (computed from its `policy.json`).

**Why these fields.** Each bit is the precondition of the helper that uses it, computed by the same function, as `nearby_log` is for `CHOP` (`:695`, `:894`). An unmasked learner can therefore learn exactly which actions are feasible from the state; only failures that depend on the terrain stay random. `stone_items=3` is the "can I afford the next craft" field from open problem 4 in `TEAM_STATUS.md`. The wood-shortage loop cannot recur here, because nothing uses up stone except the final craft.

**Left out:**
- **Logs, planks and sticks.** No stone action uses them, and they are almost constant at hand-off: 114 of 124 successes ended with 0 logs, 3 planks and 2 sticks (computed).
- **`table_placed`.** It is always 1, because no stone action removes the table.
- **Depth.** `DIG_DOWN` runs until stone is in reach and keeps its own memory.
- **Pickaxe uses left.** It is recorded and is a failure cause, but it is not a state field (section 4.9).
- **History bits.** Excluded by assumption A2.

**Disclosure.** `stone_visible` depends on helper memory. Failed targets are blocked for a few attempts, as in `SearchMemory` (`:40-54`). It is the helper's precondition, not a pure perception bit.

### 3.3 Reward and potential

**Base reward.**
- −0.01 per stone decision;
- +10 when the bridge confirms a **held** stone pickaxe;
- −10 for death.

**Settings.** gamma 0.97, alpha 0.2, epsilon falling linearly from 0.6 to 0.1 over the training episodes. Terminal potential is 0. Hitting the step limit truncates the episode but does not terminate it, so updates still bootstrap, as in `resources.py:211-229` and `infrastructure/resource_session.py:358-365`.

**Potential** (range 0 to 1.1):

`Phi_stone(s) = 0.3·min(3, stone_items) + 0.1·nearby_stone·[stone_items < 3] + 0.2·nearby_table·[stone_items ≥ 3]`

Shaped reward is −0.01 + 0.97·Phi(next) − Phi(current), computed:

| Transition | Shaped reward |
| --- | --- |
| Reaching stone with 0 items (`SEARCH_STONE` or `DIG_DOWN`) | +0.087 |
| Mining from 0 to 1 item, stone still in reach | +0.278 |
| Mining from 1 to 2 items, stone still in reach | +0.269 |
| Mining from 2 to 3 items, away from the table | +0.163 |
| Returning to the table with 3 items | +0.157 |
| Crafting the stone pickaxe (terminal) | +8.89 |
| Leaving the table with 3 items | −0.237 |
| Leaving stone with 1 item | −0.119 |
| A step that changes nothing | −0.01 − 0.03·Phi (−0.01 to −0.043) |
| Death | −10.01 − Phi |

**Why three terms.** Every step forward has a positive shaped reward. At `87673b8`, a useful action whose first update gave a negative target fell below actions never tried in that state (initial value 0), and frozen policies looped. That was 7 of the 19 learned-policy test failures.
- `SEARCH` was potential-neutral in the wood stage, so it fell below the untried actions.
- With only a material term, `RETURN_TO_TABLE` with 3 items would earn −0.037 and repeat that mechanism. The table term fixes it.

With initial Q 0, the first update of a useful action whose next state has no learned values yet gives 0.2 × the shaped reward, which is above 0. This is a property of the design, not a measured result.

**Rejected:**
- a depth or visibility term, because seeing stone is not progress;
- a per-second cost, which would make the stone and wood rewards incomparable;
- one potential shared across stages, because the stages are separate MDPs.

### 3.4 Episode limits

- **Stage 1:** 60 decisions, as at `87673b8`.
- **Stage 2:** 30 decisions.

**Why 30.**
- **Optimal path.** The shortest path is 6 decisions: `DIG_DOWN` or `SEARCH_STONE`, then 3 × `MINE_STONE`, then `RETURN_TO_TABLE` and `CRAFT_STONE_PICKAXE`. 30 is 5 times that, close to the wood stage's ratio of 60 to about 11.
- **Daylight.** At an estimated 5–10 s per decision, most episodes stay well inside the first game day.

**Declared check.** In the central parameter corner of the simulator study (section 6), if either algorithm's predicted mean frozen success is below 0.9 at 30 decisions but at least 0.9 at 40, use 40. The size of the gap between learning and random play is never the reason for the choice.

### 3.5 Initial Q

**Theory.** Under potential-based shaping, a constant initial value Q0 is equivalent to an unshaped start of Q0 + Phi(s) (Wiewiora 2003). An action refused in place converges to q*(s) = −0.333 − Phi(s), which lies between −0.333 and −1.433 here. A start of −1 is below q*(s) only where Phi < 0.667, that is with 0 or 1 items, or with 2 items away from stone.

**Rule, declared now, before tonight's result.** Use 0, unless both of these hold, in which case use −1:
- (a) the initial-Q experiment's primary rule shows an effect in favour of −1 for at least one algorithm, and against −1 for neither;
- (b) in the stone simulator study, −1 meets the budget rule of section 6 for both algorithms.

**Why the evidence must be live.** The wood simulator never produced the live loops (`configs/initial_q_experiment.json`, `pessimistic_value_basis`). So a simulator cannot choose this value on the property that matters; it can only show that a value does no harm.

**Loop counts.** Both loop measures are reported:
- untried choices, as in `scripts/initial_q_analysis.py`;
- repeated no-progress choices of actions already tried in that state.

### 3.6 Observation fields added (recorded; only those in 3.2 enter the state)

- **Inventory counts:** `inventory_counts.stone_materials` and `stone_pickaxes`, in `inventoryCounts` (`:372-383`).
- **State bits and targets:** `nearby_stone` and `stone_visible`, plus `stone_targets`, the up to 3 nearest seen targets.
- **Durability:**
  - `pickaxe_uses_left`: 59 minus `durabilityUsed`, or null if it cannot be read;
  - `pickaxe_digs`: the bridge's own count of blocks broken while the wooden pickaxe was held.
- **Hostile mobs:** `hostile_nearby` lists hostile entities within 12 blocks with line of sight; `entities.json` marks zombie, skeleton, spider, creeper and silverfish as `hostile`.
- **Time and depth:** `time_of_day` (`bot.time.timeOfDay`; the bridge does not read it today) and `depth_below_stage_start`.
- **Records:** `staircase` and `stone_mining`, each published as soon as it is created, as `recovery` is in `descendCanopy` (`:775`).
- **Survey only:** `diagnostic_only.*` (section 7.6).

The stone scans run only once a wooden pickaxe is in the inventory, so the wood stage's observation cost stays as it was at `87673b8`.

---

## 4. Bridge helpers and safety rules

### 4.1 Where the stone stage starts (computed)

- **Final support block** in the 124 successful held-out test episodes at `87673b8` (all eight arms):

  | Support block | Episodes |
  | --- | --- |
  | grass block | 72 |
  | dirt | 43 |
  | podzol | 3 |
  | its own crafting table | 3 |
  | oak log | 1 |
  | mossy cobblestone | 1 |
  | spruce leaves | 1 |
  | stone | 0 |

- **Biomes at those successes:** forest 45, savanna 22, taiga 16, dark forest 8, snowy taiga 7, old-growth spruce taiga 7, sunflower plains 7, sparse jungle 6, swamp 4, old-growth pine taiga 2.
- **Stone near the bot.** A stone-tool material block was under the bot or in a neighbouring column (`ground_block` and `adjacent_drops`, `:682-689`):
  - in 34 of the 400 `87673b8` episodes (36 if mossy cobblestone counts);
  - in 43 of all 528 retained episodes.

  This is a lower bound on what is visible within 32 blocks. By biome at `87673b8`: windswept hills 6/6, stony shore 3/3, plains 8/65, forest 5/122.
- **Hard biomes.** Spawns in desert, beach, snowy beach and swamp, where digging down is likely to be refused, were 32 of 400.
- **Consequence.** Digging down is the main route to stone. Exposed stone is cheaper when it is there, so choosing between the two is a real decision for the learner.

### 4.2 Stone perception

**Stone sources.** stone, cobblestone, deepslate and cobbled deepslate. Blackstone counts in the inventory but is not searched for in the Overworld. Lookalikes (granite and the others in section 1) are never targets.

**Discovery (`stone_visible`, the targets of `SEARCH_STONE`).**
- **Ray fan.** Rays are cast from the eyes at 32 headings by 13 elevations, from −60° to +60°, each up to 32 blocks.
- **Occlusion.** Each ray uses `bot.world.raycast` with the same per-shape occlusion as `canSeeFace` (`:223-242`).
- **Targets.** A first hit counts if it is a stone source, lies inside `TARGET_MARGIN` (`:14`, `withinRadius` `:180-182`), passes `digVerdict`, and is not blocked in a `StoneMemory` copied from `SearchMemory`.
- **Cost (estimate).** About 10^4 block reads per scan; measure it offline before the survey.
- **Gaps.** The rays are about 6 blocks apart at 32 blocks, so `stone_visible` understates what is there.
- **Rejected:** `findBlocks` with `count: 64`, the pattern in `visibleLogs` (`:515`). It would return the 64 nearest stone blocks, almost all of them buried.

**Reach (`nearby_stone`, `MINE_STONE`).** A block is in reach when all of these hold:
- `findBlocks` finds it within 6 blocks, with an exposed-face prefilter (a face next to an empty, non-liquid cell);
- `canSeeFace` passes;
- the block centre is at most 4.5 from the eyes. This is the bridge's existing limit; `canDigBlock` allows 5.1 (`mineflayer/lib/plugins/digging.js:220-226`);
- the block is not in the bot's hitbox columns, either below or above. This extends `underHitbox` (`:174-178`) upward, because a block dug above the head can let gravel fall onto the bot;
- the drop comes to rest at a height the bot can collect (section 4.5);
- `digVerdict` passes.

`observe()` and `MINE_STONE` call the same function. This is the same rule that fixed the CHOP ping-pong loop at `e88cb7c`.

**Limits to disclose.**
- Sight covers 360°, as it does for logs.
- Rays pass through water. The liquid rule removes stone seen only through water.
- Leaves block rays.
- Infested stone is visible to the bot by name, while a player sees ordinary stone. This is a perception advantage.

### 4.3 One safety check for every dig: `digVerdict`

`digVerdict(world, cell, ctx)` is a pure function that can be tested offline. It returns `{ ok, reason, block, neighbour }`, is modelled on `planCanopyDescent` (`:82-105`), and both `MINE_STONE` and `DIG_DOWN` use it. It refuses when any of the following holds:

1. **`liquid_adjacent`.** Water, lava, bubble column, seagrass, kelp or any waterlogged block is on one of the cell's 6 faces. The pathfinder's own check, `dontCreateFlow`, covers 5 faces (`mineflayer-pathfinder/lib/movements.js:256-264`).
2. **`gravity_above`.** The scan goes up through a chain of sand, red sand, gravel, suspicious sand or gravel, concrete powder or anvils, for at most 6 blocks. The pathfinder checks only one block (`movements.js:266-271`).
3. **`dripstone_below`.** Pointed dripstone hangs from the cell and would fall.
4. **`melts_to_water`.** The cell is any kind of ice.
5. **`infested`.** The block name starts with `infested_`.
6. **`spawner_nearby`.** A spawner is within 8 blocks.
7. **`own_column`.** The cell is in the bot's hitbox columns, above or below.
8. **`wrong_tool`.** The block's harvest tools exclude the wooden pickaxe (ores, obsidian), or its dig time with the chosen tool is over 4 s.
9. **`outside_margin`.** The cell's centre is outside `MAX_RADIUS - TARGET_MARGIN`.
10. **`own_table`.** The cell is the bot's table or the block under it (`ownTable`, `:456`).
11. **`hostile_nearby`.** A hostile mob is within 12 blocks with line of sight.

`DIG_DOWN` also refuses logs, leaves, stone sources, ores and containers or interactive blocks.

**Disclosure.** Checks 1 to 6 read loaded blocks that a player could not see. They only ever refuse: targets come only from visible blocks, and the staircase heading uses no knowledge of where stone is. The canopy descent already reads hidden blocks in the same way, scanning 12 blocks down (`:571-580`). The user accepted this on 24 October 2026; whether the instructor accepts it is still to be asked (section 11, item 1).

### 4.4 `DIG_DOWN`: a staircase

**Geometry.** The bot stands with its feet at (x, y, z) and picks a heading d of +x, −x, +z or −z. One step digs three cells in the next column, top first:
- H at (x+d, y+1);
- M at (x+d, y);
- L at (x+d, y−1).

The bot then moves to (x+d, y−1) with `navigate(GoalBlock)` (`:728-744`) and lands on the floor at (x+d, y−2): one block lower, with no fall damage.

**Why three cells.** These are exactly the cells the pinned pathfinder needs in both directions, with `canDig = false` and no placing (verified in the code).
- **Going down** uses the drop-down move. It needs B, C and D of the next column clear (`movements.js:476-501`); those are H, M and L.
- **Going back up** uses the jump-up move (`movements.js:300-355`). From column k it needs:
  - the cell two above its own feet, which is H of column k;
  - M and L of column k−1, which are clear;
  - a solid floor under column k−1, one block higher.
- **Consequence.** The bot can walk back up the staircase, and `RETURN_TO_TABLE` works without digging.

**Light (expected, not verified).** The first three columns are open to the sky; after that, the tunnel is roofed. Sky light should drop by about 1 per block from the opening, which leaves no light-0 cell within 6 steps.

**Before the first step:**
- If the bot stands on its own table or on a log (4 of 124 hand-offs), it first steps onto natural ground next to it, in one move of at most 4 s (`stepped_off_table`).
- If it stands on leaves (1 of 124), it first runs the existing canopy descent, as `SEARCH` does (`:848-858`).
- It centres on its cell, because `underHitbox` refuses cells under an off-centre hitbox.

**Checks on each step**, made by a pure `planStairStep(world, feet, heading, ctx)` against the current world, before every dig:
- `digVerdict` passes for every non-empty cell among H, M and L;
- the floor at (x+d, y−2) is a `walkableFloor` and not a `FALL_HAZARDS` block (`:314-320`, `:31-32`). Otherwise the step stops with `cave_below`;
- the landing feet and head cells pass `passableCell` and have no liquid beside them;
- the step's total dig time is at most 8 s (estimate of what fits the budget);
- the column is inside the margin;
- the depth is at most 12 below the stage-2 start;
- after the step, the movement rules still offer a move back up. This is the `getNeighbors` check that `enclosed()` uses (`:622-630`).

**Choosing the heading.** Among the headings whose first step passes:
1. drop headings within one column of the bot's own table;
2. prefer the first column with the most solid cells at H and M, because digging into a slope reaches stone sooner;
3. break ties in a fixed order rotated on each call, like `explorationIndex`.

**Stop rules**, checked after every dig and before every move:
- `stone_exposed`: a valid `MINE_STONE` target now exists. The floor the bot is about to land on is not a target;
- `time_budget`: 30 s have passed, or fewer than 8 s remain for a new step (estimate);
- `step_limit`: 6 steps;
- `dig_unconfirmed`, using the existing check that the cell became empty (`:661-662`);
- `move_failed`;
- `took_damage`: health dropped;
- `hostile_nearby`: a hostile mob within 6 blocks;
- `depth_limit`;
- `liquid_revealed`: the bot steps back one step.

**What it digs.**
- **By hand**, with a hotbar slot that holds no tool: grass block, dirt, coarse and rooted dirt, podzol, mycelium, clay, mud, sand, red sand, gravel and snow.
- **With the pickaxe:** granite, diorite, andesite, tuff, calcite, sandstone, red sandstone and terracotta.
- **Never:** stone sources, logs, leaves, ores, containers, ice, its own support or its own table. A heading that needs one of these is refused for that step.

**Memory.** It remembers refused headings per position. When all four are refused, it moves up to 4 blocks along the surface, inside the margin (`moved_along_surface`), so a repeated `DIG_DOWN` tries something new, as `SEARCH` does. If the bot already stands in its staircase, the next call continues it downward.

**Record.** The `staircase` record holds `heading`, `steps[{cells:[{name, pos, tool, seconds}], floor, planned_y, actual_y, damage}]`, `refused{heading: reason}`, `items_picked_up`, `start_y`, `end_y` and `stopped`. Dirt and sand picked up on the way go into the inventory like any natural item, and the record lists them.

**Worked example (estimate; the real dirt depth is unknown).** Take flat ground with grass, then 3 dirt, then stone 5 blocks below the feet.
- The five steps dig 1, 2, 3, 3 and 2 cells; the last step exposes the top of a stone block level with the floor.
- In total that is 11 hand digs and about 15–20 s, in one decision.
- Then 3 × `MINE_STONE` gives 3 cobblestone for 3 pickaxe uses:
  1. the exposed block (picked up after one reversible step down);
  2. the stone ahead at feet level;
  3. a side wall.

### 4.5 `MINE_STONE`

- **No walking before the dig.** Like `CHOP` (`:888-928`), it re-checks up to 3 candidates from the reach list.
- **The dig.** It equips the wooden pickaxe (`no_pickaxe` if there is none), then digs with `digBlock`. The dig is bounded by dig time + 3 s and by death (`:745-766`).
- **Drop band.** A target is allowed only if its drop comes to rest between feet −1 and feet +2. The vanilla pickup box covers that range (`nodeCollectsItem`, `:110-113`), and a drop at feet −1 needs only a reversible one-block step down.
- **Pickup.** It waits up to 1 s for a dropped item that is a stone material. Dirt drops from `DIG_DOWN` lie around, so the type must be checked; `droppedItemNear` (`:542-545`) does not check it. It then walks with `GoalPickup` (`:443-447`), using `CHOP`'s 9 s deadline, its 2 attempts and its rule for items out of bounds (`:912-925`).
- **Stricter movement.** Pickups and staircase moves use a second Movements object, with `maxDropDown = 2` (a feet drop of at most 1; see the comment at `:424-427`) and with `pointed_dripstone` added to `blocksToAvoid`. The wood stage's `safeMovements` (`:416-436`) stays unchanged, so stage 1 behaves as it did at `87673b8`.

### 4.6 `SEARCH_STONE`

The structure follows `search()` (`:844-887`):
1. canopy descent if the bot is on leaves;
2. `stone_already_reachable` if a stone is in reach;
3. foliage escape if the bot is enclosed;
4. otherwise, walk to `StoneMemory.next(targets)` with a `GoalMineStone(block)`, for at most 18 s. `GoalMineStone` is modelled on `GoalPickup`; its `isEnd(node)` requires the eyes within 4.5 of the block centre, a node outside the block's column, and a block at most 2 above the node;
5. after arriving, re-check against the reach list, then report `stone_reached` or `stone_path_blocked` (the target is blocked in memory);
6. with no target, take up to 4 moves on the spawn-relative exploration rings (`explorationTarget`, `:145-150`), with their own index, rescanning after each move.

The budget is 36 s, as for `SEARCH_BUDGET_MS` (`:19`). Long walks use the ordinary movement settings.

### 4.7 `RETURN_TO_TABLE`

This is the same procedure as action 6, with a 36 s deadline, a first navigation of up to 24 s, and then the approach cells (`approachCells`, `:324-334`). Action 6 keeps its 12 s navigation and 18 s deadline (`:1021-1022`), because the frozen wood policy uses it. A separate action is needed: after `SEARCH_STONE` has explored up to 48 blocks from spawn, a 12 s navigation may not reach the table (estimate).

### 4.8 `CRAFT_STONE_PICKAXE`

It needs, in order:
1. the own table in reach (`tableInReach`);
2. at least 3 stone materials and 2 sticks.

It then:
1. builds the recipe with `craftingRecipe`, extended to fill a standard stone pickaxe recipe from the inventory's actual stone-material items, as it already does for planks (`:393-414`);
2. calls `bot.craft`, then `syncInventory` (`:958-963`);
3. confirms that the count went up;
4. equips the stone pickaxe and checks that it is held, as at `:945-953`.

Success is the bridge's `success` field under the stone goal (section 5).

### 4.9 Tools and durability

- **Which item to hold.** The bridge holds the pickaxe only for `mineable/pickaxe` blocks. For everything else it selects a hotbar slot that holds no tool, which digs at hand speed and costs no durability. It checks `heldItem` after each switch.
- **Uses per episode (estimate).** The worked example costs 3 pickaxe uses; holding the pickaxe for every dig would cost about 14. An episode should use well under 59.
- **A broken pickaxe.** If the pickaxe breaks, the stage cannot finish: stone dug by hand drops nothing. `MINE_STONE` then returns `no_pickaxe`, and the failure cause `pickaxe_broken` is recorded.
- **Wood stage.** Its `CHOP` stays unchanged.

### 4.10 Night, mobs and health

- **Night.** Each episode starts in a fresh world, presumably in the morning. A successful wood-plus-stone episode should take about 2 to 2.5 minutes (estimate). The bridge records `time_of_day` in every observation, and episodes that pass tick 12000 are flagged. No time cap is declared; revisit this if the probes show episodes reaching dusk.
- **Mobs.** The server runs at difficulty normal with monsters on (`infrastructure/resource_server.py:180`, `:185`). New digs are refused while a hostile mob is within 12 blocks with line of sight, and a running macro stops at 6 blocks. There is no combat or flee skill; `hostile_nearby` and `took_damage` are observed events. Deaths do happen: a cactus at `87673b8` and a spider in the benchmark. The cause of every death is reported from the server log.
- **Health and food.** The health bin is in the state. Food is recorded; a full food bar should last a 10-minute episode (estimate).

### 4.11 The 64-block radius and wall-clock budgets

- **Radius.** It is horizontal only (`:180-182`), and leaving it invalidates the episode (`:709`). Every planned staircase column, mining target and search target stays inside `TARGET_MARGIN`. Every move goes through the pathfinder with `stepExclusion` (`:341-350`, `:433`).
- **Budgets.** Every budget in section 3.1 fits inside the bridge's 44 s request deadline (`:1070`), with room for two observations, and inside Python's 65 s response timeout (`minecraft_rl/resource_live.py:54`).
- **Load.** Under 2 slots, digs and moves take longer in real time. The stone contention check (section 7.6) measures this, because the wood-task benchmark does not cover these helpers (`configs/parallel_contention_benchmark.json`, `limits`).

### 4.12 Code changes, blocking ones first

1. **Success is wired to the wooden pickaxe** (blocking). `observe()` sets `success: counts.pickaxes > 0` (`:705`), and `step()` returns `already_completed` before any action when that holds (`:1006`). Add a `--goal` option to `parseArgs` (`:352-365`):
   - `wooden_pickaxe` is the default, which leaves wood runs unchanged; under it, actions 8–12 are refused;
   - `stone_pickaxe` sets `success` to a held stone pickaxe and allows actions 0–12.
2. **`ACTIONS` and `validateAction`** (`:8`, `:367-370`): append the new actions; **`inventoryCounts`** (`:372-383`): add the new counts.
3. **The stone recipe fill** in `craftingRecipe` (`:399`).
4. **The new pure functions**, preferably in a new module, `minecraft_bridge/stone-helpers.js`, and the five helpers built on `digBlock` and `navigate`.
5. **`ResourceAgent` is tied to 8 actions** (blocking). `N_ACTIONS` is global (`resources.py:44`) and is used at `:266`, `:299` and `:339`, and `load` compares the action names (`:323`). Generalize it over a stage specification (actions, fields, caps, environment), keeping `resource_v1` checkpoints loadable.
6. **The session coordinator** in `infrastructure/resource_session.py`, which today runs one stage per episode (section 5).
7. **The disclosure text.** `scripted_support` and `recovery_helpers` (`resource_session.py:272-290`) end with "Apart from these leaf blocks, no block is dug outside CHOP". That is false for stone runs and must be rewritten for them.
8. **Digested sources.** Add the new modules to the digested sources (`resource_session.py:298-301`; checked before every episode, `:321-323`).
9. **Step log.** Add `staircase` and `stone_mining` summaries to `recovery_summary` (`resource_session.py:133-154`).
10. **Report.** Stage-aware failure causes, replacing `FAILURE_CAUSES` (`minecraft_rl/resource_report.py:28`) and `classify_failure` (`:93`); the stone condition (`:190`); the stage-1 checkpoint in the group key (`:267`).
11. **Scheduler.** Stage keys in `SHARED_KEYS` and `JOB_KEYS` (`infrastructure/parallel_runs.py:62-68`); a `scripted_baseline` evaluation source; the stone condition in the random-baseline check (`:222-224`).
12. **CLI.** `resource-train` and `resource-demo` options for the stage, the wood policy and the stage-2 step limit (`infrastructure/__main__.py`).
13. **The static test.** It counts 3 `await digBlock(` call sites (`resource-bridge.test.js:485`). Update the count, and keep the checks for no chat, no creative mode and `canDig = false` (`:486-490`).

---

## 5. How the stone stage connects to the wood stage

**One episode** uses one fresh world, one bridge process started with `--goal stone_pickaxe`, and one bot.
1. **Start.** The existing empty-inventory checks apply (`resource-bridge.js:719`, `resource_session.py:340-341`).
2. **Stage 1.** The frozen wood checkpoint acts on `resource_v1` with actions 0–7 and epsilon 0, for at most 60 decisions. It has its own random stream, so its length does not change the stone learner's exploration. It is never updated.
3. **Hand-off.** It happens when an observation shows a wooden pickaxe after `crafted_wooden_pickaxe`. The coordinator checks the two-node recipe requirement (wooden pickaxe, then stone pickaxe):
   - sticks at least 2;
   - table placed and in reach (`CRAFT_PICKAXE` refuses unless it is, `:934-935`);
   - 3 stone materials missing.

   It records the check. If a requirement is unmet, the episode ends as `precondition_unmet`; this has not been observed (124 of 124 retained successes had at least 2 sticks, computed). The requirement is written in code, with a test that checks it against the pinned `minecraft-data` recipes and harvest tools. The general planner waits for a third node (furnace or iron).
4. **Stage 2.** The stone learner runs in the same world and bridge, for at most 30 decisions. The bridge keeps its spawn centre, `ownTable` and memories (`:455-458`). A new bridge would lose these and fail the empty-inventory check.
5. **End.** The outcome records `stage1_success`, `stage2_started`, `stage2_success`, and decisions and seconds per stage. Every transition carries `stage`. A stage-1 failure is reported as "stone stage not reached" and is never replayed.

**Stage-1 checkpoint (accepted 24 October 2026).** `results/pc_resource_runs/87673b8/mx-unmasked-q-train/policy.json`:
- SHA-256 `7d0fcce8653df8d4afc0867b321fd9670a8c64ab6ef03a876c64b9cd1c9e0349` (computed);
- condition unmasked, 180 states, 20/20 on 20000–20019.

The bridge revision changes, so its stage-1 success is measured again in every stone run. Its successes ended with 0 logs, 3 planks and 2 sticks 18 times, and with 0 logs, 1 plank and 6 sticks twice (computed).

**Rejected:**
- **One flat learner over 13 actions.** It would have 11,520 × 16 = 184,320 nominal states, relearn the wood stage, and put the +10 about 25 decisions back. Flat random play would almost never pass the wood stage, so beating it would say nothing about stone. Kept only as an optional simulator ablation.
- **Training stage 2 from prepared starts with a given wooden pickaxe.** That breaks the hard rule for any result.
- **Training the wood policy further.** That would confound the comparison.

---

## 6. Simulator extension and budget study

**`StoneSimulator`.** It lives next to `ResourceSimulator` (`resources.py:372-479`), is labelled SIMULATOR ONLY, and uses the same agent class, reward function and linear epsilon schedule as live runs. It decides budgets; it is never evidence.

**Reset.** Each episode starts from the hand-off state: 0 items, `nearby_table=1`, health above 10. It draws hidden terrain:
- `surface_stone`, with probability q_s;
- `visible_at_start`, with probability p_v, given surface stone;
- `dig_good`, with probability q_d.

**Dynamics.**
- **`SEARCH_STONE`:**
  - with stone in reach, it does nothing;
  - if stone is visible, it reaches it with probability p_nav, giving a face of U{1..6} minable blocks; a failure may block the target;
  - if nothing is visible, it reveals stone with probability p_rev, only in worlds with surface stone;
  - it always leaves the table.
- **`DIG_DOWN`:** it reaches stone with probability 0.8 in good worlds and 0.1 in bad ones; otherwise it is refused and nothing changes. Damage has a chance of 0.03 and death 0.003 (all estimates). It leaves the table.
- **`MINE_STONE`:** with probability p_mine, items go up by 1 and the face down by 1. An empty face sets `nearby_stone=0`.
- **`RETURN_TO_TABLE`:** succeeds with probability p_ret and redraws visibility.
- **`CRAFT_STONE_PICKAXE`:** with 3 items at the table, succeeds with probability 0.95.

**Priors.**
- **q_s:** corners of 0.2, 0.5 and 0.8 (pessimistic, central, optimistic). Only 34 of 400 trails touched stone (computed), which is a lower bound.
- **q_d:** 0.6, 0.75 and 0.9. Desert, beach, snowy beach and swamp spawns were 32 of 400 (computed).
- **p_nav, p_rev, p_mine and p_ret:** a grid, until the survey and the probes calibrate them from logged event rates.

**Grid.** The study crosses:
- 2 algorithms;
- initial Q of 0, −0.5 and −1;
- budgets of 30, 40 and 50 training episodes;
- step limits of 30 and 40;
- 3 corners;
- 30 agent seeds.

Each cell gets 200 frozen evaluation episodes. Uniform random and the scripted reference are run at each step limit and corner. The outputs are:
- success (mean and 10th percentile);
- both loop fractions;
- decisions and states visited;
- the exact binomial power of the Wilson rule at 20 test episodes.

It runs on the CPU; tabular work needs no GPU.

**Rules, declared before the study runs.**
- **Budget.** The smallest of 30, 40 and 50 at which both algorithms reach a mean frozen success of at least 0.9, with at least 27 of 30 agent seeds at 0.8 or above, in the pessimistic corner. If none qualifies, use 50 and declare the predicted shortfall. The minimum is 30, because live play is harder than the simulator.
- **Step limit.** As in section 3.4.
- **Initial Q.** As in section 3.5.
- **Power.** If the predicted power against random is below 0.5, this is declared before any live run.

**Output and order.**
- The study writes `results/stone_budget_study/`, laid out like `results/resource_budget_study/`.
- It runs once on the priors, then again after calibration.
- It is committed before the pre-registration that cites it.
- The simulator is easier than the game, and its hidden-terrain model drives the answer.

---

## 7. Tests

### 7.1 JavaScript unit tests (`node --test`, offline world builder in `resource-bridge.test.js`)

- **Block classes.** Stone sources and their drops are recognized. Granite, andesite, diorite, tuff, calcite, sandstone, terracotta and mossy cobblestone are never sources. `infested_*` is refused.
- **Tool choice.** The pickaxe is used on `mineable/pickaxe` blocks and the hand otherwise. Ores and obsidian are refused, as is anything over 4 s.
- **`digVerdict`.** Every refusal reason in section 4.3 has a test, including liquid on each of the 6 faces, a waterlogged neighbour, a chain of sand above, dripstone, ice, a spawner, own column above and below, the margin, and the own table.
- **`planStairStep` and heading choice.** Cases:
  - flat dirt, with the worked example's dig counts of 1, 2, 3, 3 and 2;
  - a slope;
  - a sand ceiling at step 4;
  - a cave below;
  - a river beside the staircase;
  - the radius edge;
  - the depth limit;
  - an unloaded column.
- **Reversibility.** With the real `Movements`, as the existing test at `:171-190` uses it, and `canDig = false`:
  - every staircase cell has a move one step back up;
  - every step has a drop-down move.
- **Drop band.** The rest height is computed correctly for targets from feet −2 to feet +3.
- **Visibility.** The ray fan sees an exposed face and ignores buried stone and stone behind leaves. `nearby_stone` and `MINE_STONE` use one function.
- **Recipes.** A stone pickaxe can be made from 3 cobblestone, or from 2 cobblestone and 1 cobbled deepslate through the fill. It is refused with 2 materials or 1 stick.
- **Goal gating.** Under `--goal wooden_pickaxe`, actions 8–12 are refused. Under `stone_pickaxe`, `already_completed` appears only after a stone pickaxe.
- **Static checks.** The updated `digBlock` count; no chat, no creative mode, `canDig = false`; `DIG_DOWN` never digs a stone source.

### 7.2 Python unit tests

- **State encoding.** Caps and encoding work; `stone_visible` is 1 whenever `nearby_stone` is 1; missing fields raise errors.
- **Potential.** Phi stays between 0 and 1.1, and terminal Phi is 0. Every progress transition in section 3.3 has a positive shaped reward, and a step that changes nothing gives −0.01 − 0.03·Phi.
- **Agent.**
  - a five-action agent saves and loads, and loading a policy from the other stage is refused;
  - `initial_q` is carried;
  - the retained `87673b8` checkpoint still loads (read only).
- **Reference mask.** The diagnostic mask `stone_reference_v1` is never used to choose actions. It allows:
  - search or dig when `nearby_stone=0` and fewer than 3 items;
  - mine when `nearby_stone=1` and fewer than 3 items;
  - return when 3 items and away from the table;
  - craft when 3 items at the table.

### 7.3 Simulator tests

- materials come only from `MINE_STONE`;
- a craft uses 3 materials and 2 sticks;
- crafting needs the table;
- refusals change nothing;
- seeded runs reproduce;
- the hidden-terrain draws follow the declared probabilities.

### 7.4 Session tests with a fake bridge (patterns from `tests/test_resource_session.py`)

- Stage 2 starts only after a confirmed, held wooden pickaxe.
- The stage-1 policy is never updated, and the stone learner updates only on stage-2 transitions.
- Truncation is per stage and is not termination.
- Every transition has a `stage` field.
- The handling of invalid episodes is unchanged.
- A changed source stops the next episode.
- The metadata records the wood checkpoint's SHA-256 and the new disclosure text.

### 7.5 Report, scheduler and seed tests

- **Report:**
  - every failure cause is classified: `stage1_*`, `no_stone_reached`, `stone_short`, `not_back_at_table`, `stone_pickaxe_not_crafted`, `pickaxe_broken`, `stage2_death`;
  - runs with different stage-1 checkpoints are never pooled.
- **Scheduler:**
  - the new plan keys are validated;
  - `scripted_baseline` is accepted;
  - a continuation passes the stage-1 policy, and `initial_q` only on the first part.
- **Seeds.** A new `configs/seed_registry.json` lists every range used or reserved. `tests/test_seed_registry.py` expands every `configs/*.json` plan and every retained `metadata.json` (`first_world_seed` plus `episodes_requested`) and fails on any overlap between experiments.

### 7.6 Live checks before any learning (all on development or validation seeds)

1. **Passive survey.**
   - **Setup.** Seeds 61000–61019, 1 slot, about 25 minutes (estimate). The frozen wood checkpoint plays its normal wood run with the new bridge; no stone action runs.
   - **What it records** at hand-off:
     - `stone_visible`, `nearby_stone`, `stone_targets`;
     - the time of day;
     - the cost of the ray fan in milliseconds.
   - **Survey-only diagnostics** (`diagnostic_only`, never read by the state encoder; a test checks this):
     - the depth to stone below the feet;
     - what `planStairStep` would refuse on each heading for 6 steps.
   - **What it tells us.** How deep stone is, and how often each check refuses, before the depth cap and budgets are frozen.
2. **Helper probes.**
   - **Setup.** Seeds 61020–61049, 1 slot, about 1–2 h (estimate).
   - **Questions:**
     - Is a stone dig with the wooden pickaxe accepted at about 23 ticks?
     - Does `durabilityUsed` go from 0 to 1, and stay unchanged on a dirt dig by hand?
     - Are drops collected?
     - For each staircase step: what are the planned against the actual y and the dig times, and does the bot walk back up?
     - Does a mixed-material craft work, if mixed materials occur?
   - **Footage.** One `--viewer` run for the presentation.
3. **Scripted smoke** on the same seeds. The fixed order:
   1. with 3 items: return to the table, then craft;
   2. otherwise, with stone in reach: mine;
   3. otherwise, with stone visible: search;
   4. otherwise: dig down, and search if digging is refused.

   Expect 2–3 short rounds of fixes (estimate). Every revision's runs are retained, failures included.
4. **Helper gate and stone contention check.** Validation seeds 62000–62019, declared in advance. The scripted reference runs at 1 slot, then at 2 slots, in two schedules.
   - **The gate.** At 1 slot:
     - at least 15 stage-2 successes among the episodes that reached stage 2;
     - 0 invalid episodes;
     - at most 1 death.
   - **Two slots.** They are adopted for stone only if the benchmark's rule (a) to (d) passes against 1 slot (`configs/parallel_contention_benchmark.json`). Otherwise the experiment runs at 1 slot.

---

## 8. The pre-registered live experiment

### 8.1 World seeds

**Seeds already used.**
- In retained runs: 1–3, 42, 100–149, 10000–10009, 20000–20019 and 30000–30019 (computed from every retained `metadata.json`).
- Declared tonight: 100–149 again and 40000–40019 (`configs/initial_q_experiment.json`).
- Published in the example plan: 50000–50107.

**Stone ranges.** None of these has been used:

| Range | Purpose | Rule |
| --- | --- | --- |
| 61000–61049 | Survey, probes, smoke, viewer footage | Reusable; never held-out evidence |
| 62000–62019 | Helper gate and stone contention check | May inform helper fixes; never a test |
| 63000–63049 (reserve to 63099) | Stone training; the same worlds for all six learners | As in the masking experiment |
| 64000–64019 | Stone held-out test, all 8 arms | Never used before, and never used to choose anything |
| 64100–64119 | Replacement test range | Used only if the code changes after the first test episode |

### 8.2 Arms and runs

- **Training.** Q-learning and SARSA, at agent seeds 0, 1 and 2, give 6 runs. Each runs N live episodes from 63000. N comes from the simulator rule (30, 40 or 50; estimate 40), with epsilon falling linearly from 0.6 to 0.1, initial Q by section 3.5, and no simulator pretraining.
- **Test.** 8 arms, each on 20 episodes of 64000–64019, at epsilon 0: the 6 frozen checkpoints, uniform random over the 5 stone actions, and the scripted reference. Every arm has the same frozen wood stage in front.

### 8.3 Outcomes and decision rules

- **C2, primary for learning.** Stage-2 success among episodes where stage 2 started.
  - **Why this rate.** The frozen stage 1 is identical in every arm, and its outcome cannot depend on the stone arm. Spawn points differ even on the same seed (README, "The same seed does not give the same start"), so pairing by seed does not pair the starts anyway.
  - **The rule.** For each checkpoint, its Wilson 95% interval must lie entirely above uniform random's (unpaired), as in `configs/resource_masking_experiment.json:49`. Fisher's exact test is a supplement only. An algorithm-level claim needs 2 of 3 checkpoints.
- **C1.** End-to-end success from a fresh world, by the rule in section 1. It counts the seeds both arms completed, and adds the report's conservative check for episodes that were not measured.
- **C3.** The scripted reference is reported descriptively.
- **Also reported:**
  - stage-1 success per arm;
  - decisions and task seconds per stage;
  - where the cobblestone came from;
  - `DIG_DOWN` steps and refusals by reason;
  - pickaxe uses;
  - deaths, with the cause from the server log;
  - biome and start y at the hand-off;
  - game time at the end;
  - share of choices outside the reference mask;
  - both loop fractions.

### 8.4 Power and false positives

- **Power.** With 20 per arm, a benefit can be shown only against weak baselines. For example, 14 learned successes against 5, or 19 against 10; against 14 or more, nothing can be shown (`configs/resource_masking_experiment.json:50`). Stage-1 failures reduce n a little (estimate: 0–2 per arm). The simulator's prediction of random success at the chosen step limit goes into the power note before any run.
- **False positives.** The six learned-against-random comparisons stay below 2% in total, at most 0.32% each.

### 8.5 Invalid episodes and integrity

- **Invalid episodes.** These follow the scheduler's declared rule, as in `configs/initial_q_experiment.json`. An invalid episode counts as attempted and is never replayed; the run continues from the next world seed.
- **Integrity.**
  - The experiment uses one committed revision, and the run code stays frozen while the schedule runs.
  - If the code changes after the first test episode, the results so far are retained and reported as they are, and testing restarts on 64100–64119 after retraining.

### 8.6 Files and commands

- `configs/stone_pickaxe_experiment.json`, the declaration, committed before any stone live run. It records:
  - the stage-1 path and SHA-256;
  - the condition `stone_unmasked` (state `stone_v1`, the 5 actions, mask `none`);
  - the stage-2 step limit;
  - the learners and initial Q;
  - training from 63000 with N from the study;
  - test on 64000–64019;
  - the replacement range;
  - C1 to C3, power, invalid episodes and integrity.
- `configs/stone_pickaxe_plan.json`:
  - `slots 2`, `interleave true`, `evaluate_after_all_training true` (required for parallel evaluations, `parallel_runs.py:252-254`);
  - training jobs in Q-learning and SARSA pairs per agent seed;
  - evaluation jobs `policy_from` each training job, plus `random_baseline` and `scripted_baseline`.
- **Commands:**
  1. `run.cmd resource-parallel configs\stone_pickaxe_plan.json --dry-run`;
  2. commit the code and both configs. The scheduler refuses uncommitted run code (`parallel_runs.py:815-818`);
  3. launch.
- **Retention.** Retain with `python -m infrastructure.retention runs\<plan> --out results\stone_runs\<commit>`, then run `resource-report` into `report-test/` and `report-training/`.

### 8.7 Time (estimate)

- **Training episode at 1 slot:** about 2–4 min. It is made of:
  - world start: median 16 s (computed at `87673b8`);
  - wood stage: about 25–40 s (computed for the frozen unmasked Q-learning test: median 24 s, mean 39 s of task time);
  - stone stage while exploring: 1–3 min (estimate).
- **Training block:** 240 episodes at N = 40, about 8–16 h at 1 slot, or 5–10 h at 2 slots, at the wood-task speed-up of 1.57.
- **Test block:** 160 episodes, about 2.5–5 h at 2 slots.
- **Total:** about 7.5–15 h at 2 slots. Plan the PC for an evening start that may run into the next afternoon. At 1 slot, use two nights.

---

## 9. Schedule to 19 November 2026 (ordered steps)

The PC runs one live run or schedule at a time, because of the machine-wide lock (`infrastructure/resource_server.py:31-32`, `:72-102`).

**Worktree rule.** All stone code goes in a separate git worktree, on branch `codex/stone-tools`, created from the commit that adds this document or later.
- **Why.** The overnight schedule ends launching with `code_changed` if run code in this checkout changes (`parallel_runs.py:399-414`, `:517-528`). Commits that touch only docs or configs do not count.
- **Setup.** A worktree has no `.local` runtime, `node_modules` or EULA record. Point offline tests at the main checkout's runtime, or run `setup.cmd` there. Do not record the EULA without asking the user.
- **Load.** Keep heavy CPU work, such as the simulator study, off the PC while a schedule runs.

| Date | Step | PC |
| --- | --- | --- |
| Sat 24 Oct | **S0.** The initial-Q overnight schedule runs (the user's request; estimate 10–12 h). The team reviews this design and creates the worktree. Nothing in this checkout's run code changes. | Busy |
| Sun 25 – Tue 27 Oct | **S1, offline.** Seed registry and its test; mechanics tests against `minecraft-data`; the pure JS functions and their tests; `--goal`; the observation fields and survey diagnostics; the Python stage specification, `stone_v1`, the potential and the `StoneSimulator` skeleton. | Free after S0 |
| After S0 is retained (estimate Sun 25 or Mon 26) | **S2.** Passive survey on 61000–61019. The main checkout switches to the stone branch only after the schedule has ended. | About 30 min |
| Tue 27 – Sat 31 Oct | **S3.** The five actions, coordinator, scripted controller, report, scheduler, CLI and disclosure text; review round 1. Teammates' bridge fixes, such as the deep-canopy fix, merge now or wait until after the experiment. | Free |
| Sun 1 Nov | **S4.** Helper probes and scripted smoke on 61020–61049; viewer footage. **Decision F1.** | About 2 h |
| Mon 2 Nov | **S5.** Fixes; simulator calibration and the budget study (on the laptop's CPU); review round 2. | Free |
| Tue 3 Nov | **S6.** Helper gate and stone contention check on 62000–62019. **Gate decision; 2 slots adopted or not.** | About 2–3 h |
| Wed 4 Nov | **S7.** Commit the frozen revision, the study results, the declaration and the plan; dry-run; final review. | Free |
| Thu 5 Nov | **S8.** Date gate (`docs/full_survival_plan.md:34`). The wood milestone already met it on 23 October, so report the stone status honestly. Launch the experiment in the evening. | Experiment |
| Fri 6 – Sun 8 Nov | **S9.** The experiment ends (it may run into Friday). Retention, reports, README, `TEAM_STATUS.md`, push. | Free after |
| Mon 9 – Fri 13 Nov | **S10.** Analysis, plus buffer for one full rerun with an amended declaration and test range 64100–64119. **The last launch is on the evening of Fri 13 Nov.** | As needed |
| Sat 14 – Wed 18 Nov | **S11.** Write-up and failure analysis for the assessment. | — |
| Thu 19 Nov | **S12.** Assessment of natural gathering and basic survival: wood results, stone results C1 to C3, bottlenecks, the next stage. | — |

**Effort (estimate).** About 12–15 person-days of implementation and review, so roughly one calendar week for 3–4 students. The parallel-run change needed three review rounds.

---

## 10. Risks and fallbacks

| Risk | How it shows | Response |
| --- | --- | --- |
| The bridge returns `already_completed` once a wooden pickaxe is held (`:1006`) | Stage 2 does nothing | `--goal` first; a test covers it |
| Stone is deeper than expected, or exposed stone is rarer | The survey's depth and refusal counts | Set the depth cap and budgets from the survey before freezing them |
| Sand or gravel ceilings, aquifers, rivers (desert, beach and swamp were 32 of 400 spawns) | `stair_refused` by biome | Record them as failures by biome; never weaken the safety checks to pass |
| Caves, lava, falling blocks, mobs in dark places | `cave_below`, `liquid_revealed`, `took_damage`, deaths | Refuse and record; no combat skill; report each death's cause |
| The hidden-block checks are a perception advantage | Instructor objection | Disclose them; ask the instructor early (section 11) |
| The pickaxe breaks | `pickaxe_broken` | Dig everything that is not a pickaxe block by hand; track durability |
| Frozen loops in stage 2 | Repeated no-progress choices | The state equals the preconditions; helper memory; positive shaped progress; the initial-Q rule; measure both loop fractions |
| Uniform random is strong because the helpers are reliable | 14/20 or more random | No benefit can be shown; declare the predicted power before the run |
| A heavier `observe()` changes timings under 2 slots | Time per decision, invalid episodes | Stone scans run in stage 2 only; measure the ray-fan cost; stone contention check |
| `RETURN_TO_TABLE` fails after long searches | `table_path_blocked` | 36 s deadline; carrying the table in v2 |
| The mixed recipe is wrong | `craft_not_confirmed` with materials in hand | Offline test plus one live check |
| Stage-1 losses (canopy strandings were 10 of 19 learned failures) | Fewer stage-2 starts | C2 counts conditional on stage 2 starting; a deep-canopy fix only if merged before the freeze |
| Night in long episodes | `time_of_day` past 12000 | Record and flag; revisit if the probes show it |
| A teammate commits run code during a schedule | Schedule ends with `code_changed` | Work in the worktree; announce freezes in `TEAM_STATUS.md` |
| The initial-Q declaration cites missing evidence | See section 11, item 12 | Retain those folders before tonight's launch |

**Fallback ladder (decision dates fixed now).**
- **F1, 1 Nov, after the probes.** If one helper causes most failures, fix the top refusal or pickup reasons on development seeds. If `RETURN_TO_TABLE` fails often, move carrying the table into v1. Both happen before the pre-registration, never after.
- **F2, at the 3 Nov gate or after the experiment.** If the scripted reference passes but learning shows no benefit, report C2 as "no benefit shown" and C1 with whichever label applies. The learning result of the report then remains the `87673b8` wood experiment.
- **F3.** If the gate fails (the scripted reference reaches fewer than 10 of the stage-2 starts on 62000–62019) and it is not fixed by 13 Nov, there is no stone claim. On 19 Nov, present the wood results, the retained stone-helper failure analysis, the bottleneck and the plan (`docs/full_survival_plan.md:36`). Do not relabel a smaller task as the milestone.
- **Slots.** If the stone contention check fails, run at 1 slot over two nights: about 13–25 h (estimate), with the training block one night and the test block the next.

---

## 11. Decisions for the user

**Decided 24 October 2026: the user accepted all 12 recommendations below.** Three of them still need an action:
- **Item 1.** Someone on the team must still ask the instructor whether the hybrid scope, and the disclosed hidden-block checks of item 2, are acceptable.
- **Item 4.** The rule is fixed; the value follows from the initial-Q experiment (not yet run) and the stone simulator check.
- **Item 9.** The lit-face rule applies only if the offline sky-light check in step S1 shows that Mineflayer keeps sky light for 1.21.4 chunks; otherwise stone in dark caves counts as visible and this is disclosed.

1. **Accept `DIG_DOWN`.** It is a new, disclosed, scripted staircase helper that digs dirt and rock that is not a recipe material. *Recommendation:* yes. Ask the instructor now whether the hybrid scope is acceptable (`docs/course_requirements.md:39`).
2. **Accept safety checks that read blocks the bot cannot see** (liquids, gravity blocks, floors, ice, spawners, infested stone), provided they only refuse. *Recommendation:* yes, with the disclosure in section 4.3.
3. **Choose the stage-1 checkpoint.** *Recommendation:* the `87673b8` unmasked Q-learning policy, fixed by SHA-256 `7d0fcce8…0349`. Do not pick one of tonight's 12 checkpoints by its test result.
4. **The stone initial Q.** *Recommendation:* the rule in section 3.5: 0, unless tonight's experiment favours −1 and the simulator shows that −1 does no harm.
5. **Primary outcome.** *Recommendation:* stage-2 success, conditional on reaching stage 2, for the learning claim; end-to-end success for the milestone claim.
6. **Milestone bar.** *Recommendation:* 2 of 3 checkpoints of one algorithm at 15/20 or more end to end.
7. **Stage-2 step limit.** *Recommendation:* 30, falling back to 40 only by the declared simulator check.
8. **Run the scripted reference on the held-out test worlds.** *Recommendation:* yes, descriptive only. It answers course question 4 (`docs/course_requirements.md:50`).
9. **Count only stone on a lit face as visible.** *Recommendation:* use this rule if an offline check shows that Mineflayer keeps sky light for 1.21.4 chunks; otherwise disclose that stone in dark caves counts as visible.
10. **Count cobbled deepslate and blackstone toward the 3 materials.** *Recommendation:* yes, as the recipe allows, with the mixed craft checked once live.
11. **PC time.** About two evenings or nights at 2 slots after the initial-Q run, with all development in a worktree. *Recommendation:* accept. Who owns food and "basic survival" for 19 November should be decided separately, because these runs only measure health, food and deaths.
12. **Missing evidence for tonight's run (resolved).** `configs/initial_q_experiment.json` cited `results/initial_q_study` and `results/initial_q_study_harsh` before they existed. Both were generated at `cb63ef2` with provenance and are committed with this document; they reproduce the declared numbers.

---

## 12. Where the three designs disagreed, and what was decided

| Topic | Options | Decision | Why |
| --- | --- | --- | --- |
| Action set | 4 new actions plus the wood ones; 5 stone actions with a new `RETURN_TO_TABLE`; 5 with action 6 reused | 5 stone actions, with a new `RETURN_TO_TABLE` (36 s) | Action 6 must stay unchanged for the frozen wood policy, and 12 s of navigation may not cover a return from 48 blocks |
| `DIG_DOWN` per decision | Up to 6 steps until stone is in reach; one step plus a `depth_bin` field | Up to 6 steps and 30 s | One step per decision repeats a state that does not change, which is the frozen-loop mechanism, and needs an extra state field |
| State | 5, 8 or 10 fields | 5 fields (`stone_v1`) | `table_placed` is constant; depth is not needed with multi-step digging; wood fields are almost constant; pickaxe wear is recorded instead |
| Potential | Material plus proximity; material plus table | All three terms | Every step forward then has a positive shaped reward |
| Initial Q | Chosen by the simulator; inherited from tonight's experiment | Inherited, with a declared rule and a simulator safety check | The simulator never produced the live loops |
| Step limit | 40 or 30 | 30, with a declared fallback to 40 | 5 times the 6-decision optimum, daylight, and overnight time; the gap is never the reason |
| Primary outcome | Stage-2 conditional; end to end | Conditional for C2, end to end for C1 | Stage 1 is identical in every arm, and seeds do not pair starts |
| Milestone bar | Any checkpoint at 15/20 | 2 of 3 checkpoints of one algorithm | Best of six inflates the chance of passing (0.55 against 0.04 at a true 0.6, computed) |
| Seeds | "30000 and up"; 61000–64119 | 61000–64119 with a registry | 30000–30019 is used by the contention benchmark; 40000–40019 is declared tonight |
| Order of live work | Survey, probes, scripted, then learning; gate plus contention check | All of them, in that order | Measure the helpers before learning results depend on them |
| Survey diagnostics | Log the depth to stone or not | Log it in the survey only, never as a state input | It sets the depth cap without leaking hidden information into results |
