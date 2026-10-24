# Stone-tools milestone: design

**Status.** Design accepted: on 24 October 2026 the user accepted all 12 recommendations in section 11. Offline step S1 (section 9) was built on branch `codex/stone-tools` on 24 October 2026 (commit `c441c5f`) and corrected after a review the same day; this document describes the corrected code. No stone-stage episode has been run. Numbers marked *computed* were computed on 24 October 2026 from retained files under `results/pc_resource_runs/`, or from the pinned libraries where the text says so. Numbers marked *estimate* are guesses; measurements replace them. Timings marked *measured offline* come from the offline tests on the development PC and vary with its load. Line numbers refer to commit `2618695` on `codex/pc-canopy-descent`; the initial-Q fixes in `cb63ef2` moved a few lines in `minecraft_rl/resources.py` and `infrastructure/resource_session.py`. Code added in S1 (`minecraft_bridge/stone-helpers.js`, `minecraft_rl/stone.py` and the stone parts of `resource-bridge.js`) is cited by function name. Two of the computed figures were re-checked independently from the retained files before this was committed: 114 of the 124 held-out successes ended with 0 logs, 3 planks and 2 sticks, and a stone-tool material was under or beside the bot in 34 of the 400 `87673b8` episodes. On `codex/pc-canopy-descent`, `minecraft_bridge/resource-bridge.js` has not changed since `529bf7d`; S1 changed it on `codex/stone-tools`.

This document merges three independent designs: execution helpers and safety, the learning problem, and evidence and schedule. Section 12 lists where they disagreed and what was decided.

**Assumptions.** These are team defaults that the user has not confirmed.
- **A1.** Learning is reported only without a feasibility mask. At `87673b8`, masked random play succeeded 19/20, so a masked comparison cannot show learning (`results/pc_resource_runs/README.md`).
- **A2.** The state is plain: current observations only, with no `last_action_failed` bit.
- **A3.** Later stages follow a recipe dependency graph from `minecraft-data`. Each stage has its own small tabular learner, and the stage transitions are programmed (`docs/full_survival_plan.md:7`, `:28`).
- **A4.** The stone learner's initial Q is chosen after tonight's initial-Q experiment (`configs/initial_q_experiment.json`, 0 against −1), by the rule in section 3.5.

---

## 1. Goal and definition of done

**Goal.** The bot starts in a fresh Survival world with an empty inventory and ends holding a stone pickaxe. It crafts the pickaxe at its own crafting table from 3 stone-tool materials and 2 sticks. It mines the materials itself with its wooden pickaxe. No server commands, supplied items, teleports or creative mode are allowed, in training or in testing.

**Facts checked in the pinned `minecraft-data` 3.117.0 (`pc/1.21.4`).** These come from the data files only, not from a running game. `minecraft_bridge/stone-mechanics.test.js` checks every row against the data, except the silverfish, which is game behaviour.

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
- **Mixed materials.** It is not known whether the server accepts a grid that mixes materials, for example 2 cobblestone and 1 cobbled deepslate. I expect it does, because vanilla matches a tag in each slot. But `minecraft-data` lists only single-material variants, so `bot.recipesFor` will not find a mixed recipe. The bridge fills one from the materials actually held instead (`fillStoneRecipe`, tested offline).
- **Dig times.** The library values were checked offline in S1 (`stone-mechanics.test.js`). Whether the server accepts a dig at these times is still a question for the helper probes (section 7.6). `prismarine-block`'s `digTime` and Mineflayer's `bot.digTime` agree in every case. The values, with the wooden pickaxe unless marked (computed):
  - stone: 1.15 s, or 7.5 s by hand with no drop; cobblestone: 1.5 s; blackstone: 1.15 s; cobbled deepslate: 2.65 s;
  - deepslate: 2.30 s, not the 2.25 s estimated from the vanilla formula. The library computes in double precision, where 1 / (2/3 / 30) is just above 45, so it rounds up to 46 ticks. Vanilla computes in single precision and needs 45 ticks. Mineflayer sends the finish after the library's time, so a deepslate dig ends 1 tick later than vanilla needs, never earlier;
  - dirt: 0.75 s by hand, and no faster with the wooden pickaxe; grass block: 0.9 s by hand;
  - off the ground the bot breaks at a fifth of the rate, rounded up to whole ticks: stone 5.65 s (4.91 times), cobblestone 7.5 s, deepslate 11.25 s, dirt 3.75 s, grass block 4.5 s. With the eyes in water a dig is 5 times slower too.
- **Durability.** It is not known whether a tool loses durability on dirt, grass and leaves. I believe it loses 1 for every block with non-zero hardness.
- **Reading durability.** Checked offline in S1: `prismarine-item` 1.18.0 reads the `damage` component of a 1.21.4 `set_slot` packet (`index.js:369-386`), so `durabilityUsed` is 0 for a new pickaxe and 1 after one use (`resource-bridge.test.js`). Not verified live: that the server sends the component after each use.
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

**Size.** 4 × 2 × 2 × 2 × 3 = 96 nominal states, 64 of them with health above 0. Excluding `nearby_stone=1` with `stone_visible=0` leaves 48 consistent live states, so the table holds at most 240 Q values. For comparison, the unmasked wood learner visited 180 states (computed from its `policy.json`). The encoder raises on an inconsistent observation, and loading a policy file that holds an inconsistent state is refused (the `consistent` check of `STONE_STAGE`).

**Why these fields.** Each bit is the precondition of the helper that uses it, computed by the same function, as `nearby_log` is for `CHOP` (`:695`, `:894`). An unmasked learner can therefore learn exactly which actions are feasible from the state; only failures that depend on the terrain stay random. `stone_items=3` is the "can I afford the next craft" field from open problem 4 in `TEAM_STATUS.md`. The wood-shortage loop cannot recur here, because nothing uses up stone except the final craft.

**Left out:**
- **Logs, planks and sticks.** No stone action uses them, and they are almost constant at hand-off: 114 of 124 successes ended with 0 logs, 3 planks and 2 sticks (computed).
- **`table_placed`.** It is always 1, because no stone action removes the table.
- **Depth.** `DIG_DOWN` runs until stone is in reach and keeps its own memory.
- **Pickaxe uses left.** It is recorded and is a failure cause, but it is not a state field (section 4.9).
- **History bits.** Excluded by assumption A2.

**Disclosure.** `stone_visible` depends on helper memory. Failed targets are blocked for a few attempts, as in `SearchMemory` (`:40-54`). It is the helper's precondition, not a pure perception bit. Both bits also pass the checks of section 4.3 that read blocks a player cannot see. A hidden liquid, loose gravity block, dripstone or spawner next to visible stone can turn either bit from 1 to 0. The drop's fall path and the climb check can turn `nearby_stone` from 1 to 0, and `stone_visible` only through it. Nothing hidden can turn a bit on.

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

- **Inventory counts:** `inventory_counts.stone_materials` and `stone_pickaxes`, in `inventoryCounts` (`:372-383`). They are on every observation under every goal. The wood state ignores both, and in a wood episode both are 0.
- **State bits:** `nearby_stone` and `stone_visible`, from one scan (`stoneScan`) in the context `MINE_STONE` and `SEARCH_STONE` use.
- **Targets:**
  - `stone_targets`: the up to 3 nearest ray-fan targets, in the order `SEARCH_STONE` would try them (memory-blocked targets are left out), as {name, x, y, z, distance, face};
  - `stone_targets_withheld`: true when targets exist but the memory gives none, after two failed searches in a row;
  - `stone_reach_candidates`: the up to 3 best blocks of the reach list that `MINE_STONE` re-checks, as {name, x, y, z, distance, drop, rest_y}.
- **Durability:**
  - `pickaxe_uses_left`: the wooden pickaxe's 59 minus `durabilityUsed`, the held pickaxe first, or null when none is left or it cannot be read;
  - `pickaxe_digs`: the bridge's own count of completed digs that started with the wooden pickaxe held.
- **Hostile mobs:** `hostile_nearby` lists the mobs `digVerdict` refuses on, as {name, x, y, z, distance}: within 12 blocks, in sight, and in the hostile set of section 4.3.
- **Time:** `time_of_day` (`bot.time.timeOfDay`), on every observation, before the wooden pickaxe too.
- **Depth and the stage start:**
  - `hand_off_y`: the y of the bot's cell at the first observation with the wooden pickaxe held;
  - `stage_start_y`: the natural floor under that cell, plus one (section 4.4), latched at the same observation;
  - `depth_below_stage_start`: `stage_start_y` minus the current feet y. It is −1 on a log stump at the hand-off, for example.
- **Records (S3):** `staircase` and `stone_mining`, each published as soon as it is created, as `recovery` is in `descendCanopy` (`:775`).
- **Survey only:** `diagnostic_only` (section 7.6).

**When they appear.** Under the default goal without the survey, these fields are absent and no stone scan runs, so a wood observation differs from `87673b8` only by the two inventory counts. Under the stone goal or the survey, every observation has them: `time_of_day` always, the others null until a wooden pickaxe is first in the inventory. The stage start and the depth stay null until the pickaxe is held. Once set, the fields stay, so a pickaxe that breaks does not remove them.

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

**Discovery (`stone_visible`, the targets of `SEARCH_STONE`; `visibleStoneTargets`).**
- **Ray fan.** Rays are cast from the eyes at 32 headings by 13 elevations, from −60° to +60°, each up to 32 blocks.
- **Occlusion.** Each ray walks the cells with prismarine-world's raycast iterator and uses the same per-shape occlusion as `canSeeFace` (`:223-242`). A ray stops at the first unloaded cell; `world.raycast` would pass through it.
- **Targets.** A first hit counts if it is a stone source, lies inside `TARGET_MARGIN` (`:14`, `withinRadius` `:180-182`), passes `digVerdict`, and is not blocked in `StoneMemory`. That memory has the rules of `SearchMemory`, keyed by column, plus a read-only `peek` for the observation. Targets are sorted nearest first. They do not pass the drop band or the climb check of the reach list.
- **Cost (measured offline).** 416 rays and about 12,000 block reads per scan (11,986 on the test's open ground with a cliff, 12,735 on the worked example), 8 to 25 ms. The tests print it on every run.
- **Gaps.** The rays are about 6 blocks apart at 32 blocks, so `stone_visible` understates what is there.
- **Rejected:** `findBlocks` with `count: 64`, the pattern in `visibleLogs` (`:515`). It would return the 64 nearest stone blocks, almost all of them buried.

**Reach (`nearby_stone`, `MINE_STONE`; `reachableStone`).** A block is in reach when all of these hold:
- it is a stone source within 6 blocks of the floored feet, with at least one open face: a face next to a cell with no collision box and no liquid;
- `canSeeFace` passes for one of those faces;
- the block centre is at most 4.5 from the eyes. This is the bridge's existing limit; `canDigBlock` allows 5.1 (`mineflayer/lib/plugins/digging.js:220-226`);
- the block is not in the bot's hitbox columns, either below or above. This extends `underHitbox` (`:174-178`) upward, because a block dug above the head can let gravel fall onto the bot;
- the drop comes to rest at a height the bot can collect (section 4.5);
- `digVerdict` passes;
- after the dig and the pickup, the bot can still climb back up without digging: the climb check of section 4.4, from where it stands and from every node that could pick the drop up. So `MINE_STONE` never mines a floor the bot climbs out on, such as the floor of the last staircase landing.

Candidates are sorted nearest to the eyes first, then lowest, then by x and z. The observation calls this function now, through `stoneScan`, in the context `MINE_STONE` will use from S3: `stoneScene()` in the bridge, with the shared climb search, the own table, the stage start, the staircase record and the pickaxe type. This is the same rule that fixed the CHOP ping-pong loop at `e88cb7c`. The function throws without a climb search, so no caller can skip that check.

**Limits to disclose.**
- Sight covers 360°, as it does for logs.
- **Light.** Stone counts as seen and in reach on any open face, lit or dark (decision 9's fallback, section 11). So stone in a dark cave counts as visible, which a player could not see. A lit-face rule cannot work with Mineflayer: it never recomputes light, so every cell the bot digs reads light 0 (section 4.4).
- Rays pass through water. Stone under water touches it, so the liquid rule removes it.
- Leaves block rays.
- Infested stone is visible to the bot by name, while a player sees ordinary stone. This is a perception advantage.
- **Seen but not minable.** A fan target can fail the drop band or the climb check, for example stone over a pit. The bot then has `stone_visible=1` and `nearby_stone=0` next to stone that `MINE_STONE` refuses. `SEARCH_STONE` re-checks the reach list on arrival and reports `stone_path_blocked` (section 4.6).

### 4.3 One safety check for every dig: `digVerdict`

`digVerdict(world, cell, ctx)` in `minecraft_bridge/stone-helpers.js` is a pure function, tested offline. It returns `{ ok, reason, block, neighbour }`, plus the tool and the dig time when it passes, and is modelled on `planCanopyDescent` (`:82-105`). The reach list, the ray fan and the staircase planner use it, so `MINE_STONE`, `SEARCH_STONE` and `DIG_DOWN` will. It takes a purpose: `mine` or `stair`. It checks in this order and reports the first refusal (the order is exported as `DIG_REFUSALS`; cheap position checks come first):

1. **`unloaded`.** The cell is in an unloaded chunk. An unloaded cell on one of its 6 faces, or under a gravity block beside it, refuses the same way at that check, with the face.
2. **`nothing_to_dig`.** The cell is air.
3. **`undiggable`.** Bedrock, or another block with no hardness.
4. **`own_table`.** The cell is the bot's table or the block under it (`ownTable`, `:456`).
5. **`stair_support`.** The cell is the floor under the staircase's start or under one of its landings (the staircase record, `ctx.stair`). These are the floors the bot climbs back up on.
6. **`own_column`.** The cell is in the bot's hitbox columns, above or below.
7. **`outside_margin`.** The cell's centre is more than `MAX_RADIUS - TARGET_MARGIN` (61 blocks) from the spawn, horizontally.
8. **`liquid_adjacent`.** The cell itself, or a block on one of its 6 faces, is water, lava, a bubble column, seagrass, kelp or any waterlogged block. The pathfinder's own check, `dontCreateFlow`, covers 5 faces (`mineflayer-pathfinder/lib/movements.js:256-264`).
9. **`gravity_above`.** The scan goes up through a chain of sand, red sand, gravel, suspicious sand or gravel, concrete powder or anvils, for at most 6 blocks. The pathfinder checks only one block (`movements.js:266-271`).
10. **`gravity_beside`.** One of the 4 side neighbours, or the block below, is a gravity block whose own cell below is free: air, a block with no collision box, or a liquid. A dig updates its 6 neighbours, and such a block then falls. A floating blob falls as a whole once one block of it moves, so the check applies wherever the bot stands. Gravity blocks resting on something stay allowed.
11. **`dripstone_below`.** Pointed dripstone is directly under the cell, pointing down or up. A hanging one would fall.
12. **`melts_to_water`.** The cell is any kind of ice.
13. **`infested`.** The block name starts with `infested_`.
14. **`spawner_nearby`.** A spawner or trial spawner is within 8 blocks of the cell. The scan covers 45 blocks around the bot: the fan's 32, 5 for the eyes and the hit cell, and 8. It reads chunk palettes, as Mineflayer's `findBlocks` does.
15. **`hostile_nearby`.** A hostile mob is within 12 blocks of the bot, with a clear line from the eyes to its middle or its head. Hostile means that `minecraft-data` 3.117.0 gives it the type `hostile` or the category "Hostile mobs": 43 names (computed), against 33 of type `hostile` alone. The category adds the slime, magma cube, phantom, ghast, shulker, hoglin, the zombie and skeleton horses, the ender dragon and evoker fangs. This is the more conservative choice, and the `hostile_nearby` field lists the same mobs.
16. **For the staircase only:** `stone_source`, `log` (logs, wood, stems and hyphae), `leaves`, `ore`, `container` (the bridge's list of blocks a click would use, which includes crafting tables and furnaces) and `not_dig_class` (any block outside the two dig lists of section 4.4, such as coloured terracotta).
17. **`wrong_tool`.** `chooseTool` takes the pickaxe for `mineable/pickaxe` blocks (and those marked `incorrect_for_*`) and the hand for everything else. The check refuses a pickaxe block whose harvest tools exclude the pickaxe (ores that need stone, obsidian), and any dig over 4 s with the chosen tool, for example an ender chest (16.9 s with the pickaxe) or a door by hand (4.55 s). A block dug by hand needs no drop, so snow, which drops nothing without a shovel, is allowed. The pickaxe is the wooden one unless `ctx.pickaxeType` is null, which the bridge sets when no wooden pickaxe is in the inventory. Pickaxe blocks are then timed and harvested by hand, so stone (7.5 s) refuses.

**Disclosure.** Checks 8 to 11 and 14 read loaded blocks that a player could not see; check 13 knows infested stone by name. Two checks outside `digVerdict` read hidden blocks too: the drop's fall path (up to 16 cells below a reach target, section 4.5) and the climb check (section 4.4), which reads the way back up. They only ever refuse: targets come only from visible blocks, and the staircase heading uses no knowledge of where stone is.

Because `nearby_stone` and `stone_visible` are these helpers' preconditions (section 3.2), the refusals also enter the learner's state. A hidden liquid, gravity block, dripstone or spawner next to visible stone can turn either bit from 1 to 0. The fall path and the climb check can turn `nearby_stone` from 1 to 0, and `stone_visible` only through it. Nothing hidden can turn a bit on, and the state never shows where hidden stone is.

The canopy descent already reads hidden blocks in the same way, scanning 12 blocks down (`:571-580`). The user accepted this on 24 October 2026; whether the instructor accepts it is still to be asked (section 11, item 1).

### 4.4 `DIG_DOWN`: a staircase

S1 built the pure planner: `planStairStep`, `chooseStairHeading` and `planStaircase` in `stone-helpers.js`, which the survey runs on a simulated world. The live helper, with its moves, stop rules, memory and record, comes in S3.

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
- **Consequence.** The bot can walk back up the staircase without digging. The climb check below enforces this for every step and every `MINE_STONE`, with the strict movements of section 4.5. `RETURN_TO_TABLE`'s ordinary movements allow every move the strict ones do.

**Light (decision 9, settled in S1).** The lit-face rule is off (`STONE_LIT_ONLY = false` in the bridge). Mineflayer 4.39 stores light only from `map_chunk` and `update_light` packets and never recomputes it after a `block_change` (`stone-helpers.test.js`). The vanilla 1.21.4 server sends `update_light` only for chunks on the edge of a player's view (`ChunkHolder.broadcastChanges`, read from the server jar during the S1 review), so never for the chunk the bot digs in. Every cell the bot digs therefore reads light 0, even when it is open to the sky, and a lit-face rule would never count the stone a staircase exposes. Stone in dark places counts as visible, and this is disclosed (section 4.2). The earlier expectation, that sky light falls by about 1 per block from the opening, assumed light computed by the client, which Mineflayer does not do. The `litOnly` option of `reachableStone` and `visibleStoneTargets` stays as a tested capability that the bridge does not use.

**Before the first step:**
- If the bot stands on its own table or on a log (4 of 124 hand-offs), it first steps onto natural ground next to it, in one move of at most 4 s (`stepped_off_table`).
- If it stands on leaves (1 of 124), it first runs the existing canopy descent, as `SEARCH` does (`:848-858`).
- It centres on its cell, because `underHitbox` refuses cells under an off-centre hitbox.

**Checks on each step**, made by a pure `planStairStep(world, feet, heading, ctx)` against the current world before every dig, in this order. It throws without the stage-start height or the climb search, so no check can be skipped silently.
1. Inside a staircase, the reverse heading is refused before anything is read (`reverse_heading`): its column holds the floor the bot climbs back on.
2. The column is inside the margin (`outside_margin`).
3. The landing is at most 12 below the stage start (`depth_limit`; the stage start is defined below).
4. H, M, L and the floor are loaded (`unloaded`).
5. H, M and L, top first: a stone source ends the digging (the expose step below); a liquid refuses (`liquid_adjacent`); a passable cell needs no dig; any other cell must pass `digVerdict` with the purpose `stair`.
6. The step's total dig time is at most 8 s (`dig_time`). With the allowed dig classes this check cannot fire: `digVerdict` refuses any cell over 4 s, and the slowest allowed cell takes 1.15 s (granite, diorite, andesite and tuff with the wooden pickaxe), so a step takes at most 3.45 s (computed).
7. For a step that moves:
   - the floor at (x+d, y−2) is a `walkableFloor` and not a `FALL_HAZARDS` block (`:314-320`, `:31-32`); otherwise `cave_below`;
   - a gravity floor must rest on something: sand or gravel over a free cell refuses (`floor_unsupported`), even when L was already open;
   - L and M pass `passableCell` (`landing_blocked`) and have no liquid (`liquid_adjacent`) and no unloaded cell (`unloaded`) beside them;
   - nothing loose hangs over the open column: the first cell above H that is not passable, within 6 cells, must not be a gravity block (`gravity_above`);
   - the pathfinder offers a direct drop-down move onto the landing, and the climb check passes from the landing (`not_reversible`).

**The expose step.** When H, M or L is a stone source, the step digs only the cells above it and does not move (`move: false`); the staircase then ends with `stone_exposed`. It is refused as `stone_source` when nothing needs digging, because that stone would already be in reach if it could be mined. It is also refused when the exposed stone would fail `digVerdict` for mining (`exposed_refusal` names why). The bot must keep its way up from where it stands. The worked example's fifth step is an expose step.

**The climb check** (`keepsWayUp`, with the search `climbSearch`). It replaced the one-step reversibility check, which accepted a dig that cut the climb further up. It runs the pinned pathfinder's own moves with the strict movements, no digging and no placing, on a view of the world with the planned cells dug. The search goes at most 16 blocks from its start on each axis and expands at most 400 nodes.
- Before a dig, it finds the bot's way up: a path to the stage-start height or, when that is beyond the search's reach, to the highest cell it can reach now.
- After the dig, every cell the bot ends on must still reach that height, or its own height if higher, or a cell of the known path. These cells are the landing for a step that moves, the bot's own cell for an expose step, and, for `MINE_STONE`, every standable node whose pickup box reaches the drop.
- A dig can cut a path only by removing the floor of one of its cells, so the bot's own cell is searched again only then.
- It fails closed: an error or a read of an unloaded cell refuses the step or the target (`unloaded` or `not_reversible`). So within about 17 blocks of an unloaded chunk, `nearby_stone` is 0 and staircase steps refuse; with the server's view distance of 8 chunks this should not happen inside the 64-block radius (estimate).
- The stage-start height stands in for the table: the check does not prove that the table's approach cells can be reached.
- **Cost (measured offline, per observation).** About 0 to 2 ms on the surface and in the worked example; 10 to 16 ms in a closed 13×13 cave, where the search runs to its bound; 55 to 100 ms in an open cave with 89 targets and a one-wide exit. `stone-helpers.test.js` prints it.

**The stage start** (`stageStartReference`), from which the depth cap and `depth_below_stage_start` count. It is the natural floor under the hand-off cell, plus one. The scan skips the bot's own table, logs, leaves and passable cells, at most 12 cells down (the canopy descent's range). With no such floor (an unloaded cell, or none in range) it stays the hand-off y, which can only make the cap shallower. It is latched once, at the first observation with the wooden pickaxe held, and never derived again, so a walk into a valley cannot deepen the cap. The cap is therefore 12 below the ground wherever the hand-off stood within 12 cells of it: on the ground, on its own table (3 of 124 hand-offs), a log (1) or a canopy (1).

**Choosing the heading** (`chooseStairHeading`).
- At a staircase's first step, among the headings whose step passes:
  1. drop the headings whose first column lies within one column of the bot's own table (`near_own_table`, listed in `near_table`). This applies to the first step only; `digVerdict`'s `own_table` protects the table and the block under it at every step;
  2. prefer the first column with the most solid cells at H and M, because digging into a slope reaches stone sooner;
  3. break ties in a fixed order rotated on each call, like `explorationIndex`.
- Inside a staircase (its record `ctx.stair` has a landing), it continues the staircase's heading and turns to a side only when that heading is refused. It never takes the reverse heading, and the near-table rule does not apply.
- `DIG_DOWN` calls it only at a staircase's start or with the staircase record set.

**Stop rules**, checked after every dig and before every move:
- `stone_exposed`: a valid `MINE_STONE` target now exists from the new landing, or an expose step uncovered one. No floor of the staircase is a target (`stair_support`), and no target may cut the way back up (the climb check);
- `time_budget`: 30 s have passed, or fewer than 8 s remain for a new step (estimate);
- `step_limit`: 6 steps;
- `dig_unconfirmed`, using the existing check that the cell became empty (`:661-662`);
- `move_failed`;
- `took_damage`: health dropped;
- `hostile_nearby`: a hostile mob within 6 blocks;
- `depth_limit`;
- `liquid_revealed`: the bot steps back one step.

**What it digs.**
- **By hand**, with a hotbar slot that holds no tool: grass block, dirt, coarse and rooted dirt, podzol, mycelium, clay, mud, sand, red sand, gravel, snow and snow block. `minecraft-data` gives snow layers no collision box, so the planner walks through them instead of digging.
- **With the pickaxe:** granite, diorite, andesite, tuff, calcite, sandstone, red sandstone and plain terracotta. Coloured terracotta is refused (`not_dig_class`).
- **Never:** stone sources, logs, leaves, ores, containers, ice, any other block outside these two lists, its own support, the floors of its staircase or its own table. A heading that needs one of these is refused for that step, except that a stone source ends the staircase with an expose step.

**Memory.** It remembers refused headings per position. When all four are refused, it moves up to 4 blocks along the surface, inside the margin (`moved_along_surface`), so a repeated `DIG_DOWN` tries something new, as `SEARCH` does. If the bot already stands in its staircase, the next call continues it, by the heading rule above.

**Record.** The `staircase` record holds `heading`, `steps[{cells:[{name, pos, tool, seconds}], floor, planned_y, actual_y, damage}]`, `refused{heading: reason}`, `items_picked_up`, `start_y`, `end_y` and `stopped`. Dirt and sand picked up on the way go into the inventory like any natural item, and the record lists them. The checks also need the live staircase as `{ heading, start, landings }`: the heading of the last move, the feet where the staircase began, and each landing after a confirmed move. The observation's scan and `MINE_STONE` read the same record, so both protect its floors.

**Worked example (the terrain is made up; the real dirt depth is unknown).** Take flat ground with grass, then 3 dirt, then stone 5 blocks below the feet. The planner and the reach rule give (computed, and tested on all four headings):
- The five steps dig 1, 2, 3, 3 and 2 cells; the last step exposes the top of a stone block level with the floor.
- In total that is 11 hand digs: 3 grass blocks at 0.9 s and 8 dirt at 0.75 s, 8.7 s of digging. With the moves it takes about 15–20 s (estimate), in one decision.
- Then 3 × `MINE_STONE` gives 3 cobblestone for 3 pickaxe uses. On the +x heading the targets are the exposed block, picked up after one step down into its cell, and then the two side walls of that cell, at feet level.
- The floor of the last landing is never a target, and after each mine the bot can still climb to the surface.

### 4.5 `MINE_STONE`

- **No walking before the dig.** Like `CHOP` (`:888-928`), it re-checks up to 3 candidates from the reach list.
- **The dig.** It equips the wooden pickaxe (`no_pickaxe` if there is none), then digs with `digBlock`. The dig is bounded by dig time + 3 s and by death (`:745-766`).
- **Drop band.** A target is allowed only if its drop comes to rest between feet −1 and feet +2, measured from the bot's actual y. The vanilla pickup box covers that range (`nodeCollectsItem`, `:110-113`). The rest height is the collision top of the first block below with a shape; a fall into a liquid or an unloaded cell, or of more than 16 blocks, refuses. A drop at feet −1 needs a one-block step down, and the climb check requires a way back out of it.
- **Pickup.** It waits up to 1 s for a dropped item that is a stone material. Dirt drops from `DIG_DOWN` lie around, so the type must be checked; `droppedItemNear` (`:542-545`) does not check it. It then walks with `GoalPickup` (`:443-447`), using `CHOP`'s 9 s deadline, its 2 attempts and its rule for items out of bounds (`:912-925`).
- **Stricter movement.** Pickups and staircase moves use a second Movements object (`strictMovements`), with `maxDropDown = 2` (a feet drop of at most 1; see the comment at `:424-427`) and with `pointed_dripstone` added to `blocksToAvoid`. Its step exclusion treats the pathfinder's stand-in for an unloaded block as excluded instead of throwing. The climb check uses the same rules. The wood stage's `safeMovements` (`:416-436`) stays unchanged, so stage 1 behaves as it did at `87673b8`.

### 4.6 `SEARCH_STONE`

The structure follows `search()` (`:844-887`):
1. canopy descent if the bot is on leaves;
2. `stone_already_reachable` if a stone is in reach;
3. foliage escape if the bot is enclosed;
4. otherwise, walk to `StoneMemory.next(targets)` with a `GoalMineStone(block)`, for at most 18 s. `GoalMineStone` is modelled on `GoalPickup`; its `isEnd(node)` requires the eyes within 4.5 of the block centre, a node outside the block's column, and a block at most 2 above the node;
5. after arriving, re-check against the reach list, then report `stone_reached` or `stone_path_blocked` (the target is blocked in memory). A fan target can fail the reach list's drop band or climb check (section 4.2), so this re-check is needed;
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

- **Which item to hold.** The bridge holds the pickaxe only for `mineable/pickaxe` blocks and those marked `incorrect_for_*` (`chooseTool`). For everything else it selects a hotbar slot that holds no tool, which digs at hand speed and costs no durability. It checks `heldItem` after each switch.
- **Uses per episode (estimate).** The worked example costs 3 pickaxe uses; holding the pickaxe for every dig would cost about 14. An episode should use well under 59.
- **A broken pickaxe.** If the pickaxe breaks, the stage cannot finish: stone dug by hand drops nothing. `MINE_STONE` then returns `no_pickaxe`, and the failure cause `pickaxe_broken` is recorded. With no wooden pickaxe in the inventory, the stone checks run with `pickaxeType: null`, so every stone source refuses as `wrong_tool` and `nearby_stone` and `stone_visible` are 0.
- **Wood stage.** Its `CHOP` stays unchanged.

### 4.10 Night, mobs and health

- **Night.** Each episode starts in a fresh world, presumably in the morning. A successful wood-plus-stone episode should take about 2 to 2.5 minutes (estimate). The bridge records `time_of_day` in every observation under the stone goal or the survey, before the wooden pickaxe too. Episodes that pass tick 12000 are to be flagged (S3). The value wraps every 24000 ticks (20 minutes), so the flag should look at every reading, not only the last. No time cap is declared; revisit this if the probes show episodes reaching dusk.
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
   - fewer than 3 stone materials held (`handoff_failures` in `minecraft_rl/stone.py` names the failure `stone_materials_already_held`).

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

### 7.1 JavaScript unit tests (`node --test`, offline 1.21.4 worlds)

S1 put these tests in three files: `stone-mechanics.test.js` (the facts of section 1, against the pinned `minecraft-data` and libraries), `stone-helpers.test.js` (the pure helpers, on the offline world builder of `resource-bridge.test.js`) and `resource-bridge.test.js` (the bridge, partly through a stand-in Mineflayer bot on an offline world). Every item below has a test now, except those marked S3.

- **Pinned facts.** `minecraft-data` 3.117.0 for pc 1.21.4; the three stone pickaxe recipes; the five blocks that give a material, with their hardness; the lookalikes and infested blocks; durability 59 and 131; the furnace; the dig times of section 1 through `block.digTime` and Mineflayer's `bot.digTime`.
- **Block classes.** Stone sources and their drops are recognized. Granite, andesite, diorite, tuff, calcite, sandstone, terracotta and mossy cobblestone are never sources. `infested_*` is refused.
- **Tool choice.** The pickaxe is used on `mineable/pickaxe` blocks and the hand otherwise. Ores and obsidian are refused, as is anything over 4 s. With no pickaxe (`pickaxeType: null`), pickaxe blocks are timed by hand and refused as `wrong_tool`.
- **`digVerdict`.** Every refusal reason in section 4.3 has a test, and the exported order is pinned. The cases include:
  - liquid on each of the 6 faces, and a waterlogged neighbour or cell;
  - a chain of sand above; a loose gravity block beside or below (a gravel chain over a tunnel, the edge of a floating 3×3 blob, sand floating over a cave), with supported gravity blocks still allowed;
  - dripstone, ice and infested stone;
  - a spawner within 8 blocks, including one 8 blocks from a fan hit at the 32-block edge;
  - own column above and below, the margin, the own table and a staircase floor;
  - a hostile in sight, and not behind a wall;
  - unloaded neighbours on a chunk border.
- **Staircase planner** (`planStairStep`, `chooseStairHeading`, `planStaircase`). Cases:
  - flat dirt, with the worked example's dig counts of 1, 2, 3, 3 and 2 and 8.7 s of hand digs;
  - a slope; a sand ceiling at step 4; a cave below; a river beside the staircase; the radius edge; the depth limit;
  - an unloaded column, and unloaded cells beside L and M;
  - a log, an ore, leaves or stone in the way; an expose step; granite dug with the pickaxe;
  - a landing floor of loose sand or gravel, and the bot parked under floating gravel;
  - liquid beside an M or L that is already open, and exposed stone that `MINE_STONE` would refuse;
  - the near-table rule on the first step only, with a heading through the table refused as `own_table`;
  - inside a staircase, the heading continues, turns only to a side, and never reverses.
- **Climb check.** With the real pinned `Movements` (the strict ones, `canDig = false`, no placing):
  - every step of the worked example has a drop-down move and a move back up;
  - after the worked example, three `MINE_STONE`s in a row on every heading keep y 64 reachable, and the floor of the last landing is never a target;
  - a pickup step-down with no headroom is refused;
  - the reverse heading would have cut the climb;
  - an open cave's one-wide exit keeps its floor;
  - the test prints the check's cost.
- **Drop band.** The rest height for targets from feet −2 to feet +3, and reach refusing stone over a pit, on a slab or pillar ledge, or over water.
- **Visibility.** The ray fan sees an exposed face and ignores buried stone, stone behind leaves, lookalikes and stone behind an unloaded chunk. A hostile in sight empties it. `nearby_stone` and `MINE_STONE` use one function. The stone memory blocks failed columns, and `peek` does not advance it.
- **Light.** Mineflayer's own blocks plugin keeps the light of `map_chunk` and `update_light`, and a `block_change` leaves the dug cell at sky light 0. The unused `litOnly` option still works.
- **Recipes.** A stone pickaxe can be made from 3 cobblestone, or from 2 cobblestone and 1 cobbled deepslate through the fill. It is refused with 2 materials or 1 stick.
- **Observation fields and survey.** The fields of section 3.6 and the survey record of section 7.6, on offline worlds: the fan targets and reach candidates, the withheld flag, a dark world counting like a lit one, stone over a pit, a hostile in sight, no pickaxe, the stage start on the ground, the own table, a stump, a canopy and a slab, and the pickaxe-use count and depth over several requests.
- **Goal gating.** Under `--goal wooden_pickaxe`, actions 8–12 are refused. Under `stone_pickaxe`, `already_completed` appears only after a stone pickaxe, and a stone action ends the episode as invalid until S3.
- **Wood stage unchanged.** Under the default goal the observations keep the wood keys, carry no stone field apart from the two inventory counts, and run no stone scan. The bridge answers a scripted stand-in bot exactly as the `409560e` bridge did, apart from the two stone counts; that test needs git and commit `409560e`, and fails without them.
- **Static checks.** No chat, no creative mode, `canDig = false`; the stone helpers never dig, craft, equip, place or move the bot. S3: the updated `digBlock` count, and `DIG_DOWN` never digs a stone source. Until then the survey test checks that no planned staircase dig is a stone source.

### 7.2 Python unit tests

- **State encoding.** Caps and encoding work; `stone_visible` is 1 whenever `nearby_stone` is 1; missing fields raise errors.
- **Potential.** Phi stays between 0 and 1.1, and terminal Phi is 0. Every progress transition in section 3.3 has a positive shaped reward, and a step that changes nothing gives −0.01 − 0.03·Phi.
- **Agent.**
  - a five-action agent saves and loads, and loading a policy from the other stage is refused;
  - loading a policy file that holds a state the encoder never produces (`nearby_stone=1` with `stone_visible=0`) is refused;
  - a copied or unpickled agent saves the same file, and a stone one still evaluates;
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
- **Seeds.** `configs/seed_registry.json` lists every range used or reserved. `tests/test_seed_registry.py` expands every plan and declaration under `configs/` (subfolders included) and every retained `metadata.json` (`first_world_seed` plus `episodes_requested`). It fails:
  - on a used seed that no range of its experiment covers;
  - on seeds that several experiments use when the one range holding them does not name them all;
  - on a held-out range used by more than one experiment, and on any training run on a held-out range;
  - on any evaluation on a training range;
  - on a committed plan outside `configs/`.

  A second test pass inside one experiment is not detected there; the experiment's declaration governs it (section 8.5).

### 7.6 Live checks before any learning (all on development or validation seeds)

1. **Passive survey.**
   - **Setup.** Seeds 61000–61019, 1 slot, about 25 minutes (estimate). The frozen wood checkpoint plays its normal wood run with the new bridge and `--stone-survey`; no stone action runs. It runs by hand with the command in `docs/resource_training.md`, because a parallel plan cannot set the survey.
   - **What it records** once a wooden pickaxe is in the inventory: the fields of section 3.6, among them `stone_visible`, `nearby_stone`, `stone_targets`, `stone_reach_candidates` and the stage start. The time of day is on every observation.
   - **Survey-only diagnostics** (`diagnostic_only`, never read by the state encoder; a test checks this), once the pickaxe is held:
     - `lit_only` (false) and `stage_start`: the hand-off y, the floor, and the cells skipped above it;
     - `depth_to_stone`: the depth to stone below the feet, the layers above it, and the depth from the stage start;
     - `table_offset`, `on_own_table` and `table_dropped_headings`: where the bot stands relative to its table, and which headings the first-step table rule drops;
     - `staircases`: what `planStairStep` would refuse on each of the four headings for up to 6 steps. Every heading is planned, so each reports the terrain. A heading stops at a refusal, at exposed stone, or once `MINE_STONE` would have a target, with the floors of the staircase so far protected;
     - the cost of the ray fan, of the reach scan and of the whole record (about 35 to 60 ms on the worked example, measured offline).
   - **Not simulated.** The survey plans from the hand-off cell. It does not simulate `DIG_DOWN`'s step off a table or log, the canopy descent or the centring; `on_own_table` and `stage_start.skipped` mark those hand-offs.
   - **What it tells us.** How deep stone is, and how often each check refuses, before the depth cap and budgets are frozen.
2. **Helper probes.**
   - **Setup.** Seeds 61020–61049, 1 slot, about 1–2 h (estimate).
   - **Questions:**
     - Is a stone dig with the wooden pickaxe accepted at about 23 ticks, and a deepslate dig at 45 (the library waits 46)?
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

**Seeds already used.** `configs/seed_registry.json` lists all of these and the stone ranges below, and `tests/test_seed_registry.py` checks it (section 7.5).
- In retained runs: 1–3, 42, 100–149, 10000–10009, 20000–20019 and 30000–30019 (computed from every retained `metadata.json`). The registry reserves 42–46, the worlds a run started without `--world-seed` plays.
- Declared tonight: 100–149 again, 40000–40019 (`configs/initial_q_experiment.json`) and 40100–40119. The last is the initial-Q filler run, which only keeps the second slot busy at the end of the schedule and is never analysed (`configs/initial_q_plan.json`).
- 777: one ad hoc frozen evaluation of the `87673b8` unmasked Q-learning policy on 24 October 2026. It was not retained and is not evidence.
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
| Sat 24 Oct (planned Sun 25 – Tue 27) | **S1, offline. Done** (see below). Seed registry and its test; mechanics tests against `minecraft-data`; the pure JS functions and their tests; `--goal`; the observation fields and survey diagnostics; the Python stage specification, `stone_v1`, the potential and the `StoneSimulator` skeleton. | Free after S0 |
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

**S1 as built (24 October 2026).** Everything in the S1 row exists on `codex/stone-tools`, offline. Built at `c441c5f`, it was reviewed the same day, and the confirmed findings were fixed. In the bridge: loose gravity blocks beside a dig, the climb check, the staircase floors, the reverse heading, no pickaxe, unloaded cells, the spawner scan radius, the light rule, the table rule, the stage start, `time_of_day` and the target fields. In Python: copied agents and inconsistent policy states. In the seed test: the reuse rules, range kinds, seed keys and plans in subfolders.
- Of the code changes in section 4.12, S1 did items 1 to 3, the pure functions of item 4, item 5, and item 8 for `stone-helpers.js`.
- Under `--goal stone_pickaxe` a stone action ends the episode as invalid until S3, and `resource_session` refuses the stone goal, so the stone stage cannot run yet.
- `--stone-survey` exists on `resource-train` and `resource-demo`.

**Before S2.**
- S0 has ended and is retained; only then does the main checkout move to the stone branch.
- The survey runs by hand at 1 slot (section 7.6, item 1).
- Its retained folder itself must be registered under `stone_development` in `configs/seed_registry.json`, or the seed test fails.
- S2 then measures what sets the depth cap and budgets: the depth to stone from the stage start, each check's refusals on all four headings, the fan cost and the time of day.

**What S3 inherits.**
- `MINE_STONE`, `DIG_DOWN` and `SEARCH_STONE` take their context from the bridge's `stoneScene()`: the shared climb search, the own table, the latched stage start and the pickaxe type. `DIG_DOWN` replaces its `stair: null` with the live staircase record, and the observation's scan and `MINE_STONE` read the same record.
- `DIG_DOWN` calls `chooseStairHeading` only at a staircase's start or with the record set, sets the record's heading after a turn, and counts its depth cap from `stage_start_y`, not from the hand-off y.
- `SEARCH_STONE`'s goal must expect to arrive at stone that `MINE_STONE` refuses (section 4.2).
- The live parts of sections 4.4 and 4.5: the moves before the first step, the stop rules, the memory and the records.
- The static test: the new `digBlock` count, and a check that `DIG_DOWN` never digs a stone source.
- The session: the coordinator, loading the stone policy with its stage, the stone condition note, the disclosure text of section 4.12 item 7 with the state-bit sentence of section 4.3, and digesting `minecraft_rl/stone.py` once the session imports it.
- The tick-12000 flag (section 4.10).
- The climb check's cost in a large open cave, about 55–100 ms per observation (measured offline). The S4 probes show whether that matters live.

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

**Decided 24 October 2026: the user accepted all 12 recommendations below.** Two of them still need an action, and S1 settled a third:
- **Item 1.** Someone on the team must still ask the instructor whether the hybrid scope, and the disclosed hidden-block checks of item 2, are acceptable. The question should include what S1 added to section 4.3: loose gravity blocks beside a dig, the drop's fall path and the climb check. It should also say that these refusals enter the state bits `nearby_stone` and `stone_visible`, which they can only turn off.
- **Item 4.** The rule is fixed; the value follows from the initial-Q experiment (not yet run) and the stone simulator check.
- **Item 9 (settled in S1): the fallback applies.** Mineflayer does keep the sky light of 1.21.4 chunks, but only from `map_chunk` and `update_light`; it never recomputes it. The vanilla server sends `update_light` only for chunks on the edge of a player's view, so every cell the bot digs reads light 0, and a lit-face rule would never count the stone a staircase exposes. The rule is off. Stone in dark places counts as visible, and this is disclosed (sections 4.2 and 4.4).

1. **Accept `DIG_DOWN`.** It is a new, disclosed, scripted staircase helper that digs dirt and rock that is not a recipe material. *Recommendation:* yes. Ask the instructor now whether the hybrid scope is acceptable (`docs/course_requirements.md:39`).
2. **Accept safety checks that read blocks the bot cannot see** (liquids, gravity blocks, floors, ice, spawners, infested stone), provided they only refuse. *Recommendation:* yes, with the disclosure in section 4.3.
3. **Choose the stage-1 checkpoint.** *Recommendation:* the `87673b8` unmasked Q-learning policy, fixed by SHA-256 `7d0fcce8…0349`. Do not pick one of tonight's 12 checkpoints by its test result.
4. **The stone initial Q.** *Recommendation:* the rule in section 3.5: 0, unless tonight's experiment favours −1 and the simulator shows that −1 does no harm.
5. **Primary outcome.** *Recommendation:* stage-2 success, conditional on reaching stage 2, for the learning claim; end-to-end success for the milestone claim.
6. **Milestone bar.** *Recommendation:* 2 of 3 checkpoints of one algorithm at 15/20 or more end to end.
7. **Stage-2 step limit.** *Recommendation:* 30, falling back to 40 only by the declared simulator check.
8. **Run the scripted reference on the held-out test worlds.** *Recommendation:* yes, descriptive only. It answers course question 4 (`docs/course_requirements.md:50`).
9. **Count only stone on a lit face as visible.** *Recommendation:* use this rule if an offline check shows that Mineflayer keeps sky light for 1.21.4 chunks; otherwise disclose that stone in dark caves counts as visible. *Outcome:* the fallback (see the note above).
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
| Survey diagnostics | Log the depth to stone or not | Log it in the survey only, never as a state input | It sets the depth cap without putting the hidden depth into the state (the safety refusals do enter the state bits, section 4.3) |
