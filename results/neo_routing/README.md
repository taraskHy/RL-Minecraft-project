# Neo routing: the neo planner without the course list

Branch `neo-routing`, a pull request into `main` for the main session to review and merge. Opt-in code:
`RouteConfig.blocked` (`parkour/route_env.py`), `parkour/route_neo.py`, `parkour.live --router R --assist JSON`,
`bridge/router.js` (`cfg.blocked`), `RouteAssist` in `bridge/assist.js`, the planner service's router requests
(`neo_assist.serve`); tests `tests/test_route_neo.py` and `tests/test_route.py`. The defaults and everything in
HANDOFF section 5 are unchanged. Times are Israel time, read from `date` (Git Bash labels it JST).

## Headline

The chosen setting is A + `plan_on`. On the single-neo sets (320 courses), counts are completions with the neo done
as a neo. The files for every number are in the sections below.

- **Validation** (family 40, 320 single-neo courses, simulator, CPU): **255, 257 and 251** for search seeds 0, 1 and
  2 (mean 254.3). As shipped, the router completes 8 to 13. The course-list planner gets 286 at seed 0 and 276 at
  seed 1.
- **Test** (family 41, run once): **252** of 320. The course-list planner gets 276.
- **The router's own validation sets** (1,800 courses each):
  - Family 2: **1,762**, the shipped number. Nothing is planned there (no walls), and the run equals the 30 Oct
    filter-off what-if on every course.
  - Family 5: **1,728** against the shipped 1,720 (9 courses won, 1 lost; 7 of the wins are planned jumps).
- **Bots** on the 64 test courses of `single41-first64`: **52 of 64 offline and 52 of 64 in real Minecraft**. Each
  matches its simulator re-run on every course (outcome and ticks), with 0 server corrections, and the two bots match
  each other. The course-list planner gets 55 in both.
- **Choices**, by rules written before each run: rule 1 kept A over B, which led by 3.0 (under the rule's 4); rule 3
  added `plan_on`, which led A by 15.3.
- **Where the router loses courses against the course list** (seed 0): 15 against 4 before the neo, and 36 against 24
  at the neo. After the neo it loses 10 against 6.

## Why (measured on 30 Oct before this branch; simulator, GPU; files in the main checkout's `runs/`)

- **As shipped, the router fails on neo courses.** The learned router (`results/routing/router_policy.pt`)
  completes 8 to 13 of the 320 single-neo validation courses (family 40) with any of four networks jumping and
  does the neo in at most 2 (`runs/neo-route-check/`, `runs/neo-route-check-summary.json`): at the neo's
  takeoff it is offered nothing to jump to.
- **One rule causes it.** The planner's straight-path rule (`PlannerConfig.clear_path`; `planner.path_clear`,
  `route_env.path_clear`) drops the neo's landing on all 320 courses; the height limit, the gap limit and the
  room-to-stand check pass (`runs/neo-route-check-filters.json`).
- **Without that filter, routing works on neos:** the router picks the neo's landing at the takeoff on about 99%
  of the courses that reach it; with `neo-dagger4` updates 250 jumping it completes 147 of 320 (145 with the neo),
  against 146 with the course list; the failures left are the jumps (162 failed jumps, 11 wrong platforms, 0 with
  nothing in view) (`runs/neo-route-check-nopath/`, `runs/neo-route-check-rulecost.json`).
- **The filter is worth nothing to the router on its own validation sets** (families 2 and 5, 1,800 courses
  each, final policy jumping): as shipped 1,762 and 1,720; filter off with the head-hitter flags kept 1,762 and
  1,720 (`runs/neo-route-check-rulecost.json`; per course, 2 and 4 courses change outcome, in both directions).

## The mode

`RouteConfig(blocked="plan")` (simulator: `python -m parkour.route_neo`; bots: `parkour.live --router R --assist
JSON`):

1. A platform whose straight flight path from the platform planned from passes through blocks stays selectable,
   with its head-hitter flag (`blocked="keep"` is this step alone; `"drop"`, the default, is the planner's rule).
2. When the router's first choice is such a platform and a block in the way is no part of a platform in view (a
   wall, pillar or beam: the blocks `neo_dagger.neo_takeoffs` counts on the course list; `route_env.wall_blocked`),
   the jump to it is a neo: the neo planner (`neo_assist.plan_once`) searches it from the bot's exact state with
   the chosen platform as the target, and the actions found replace the jumper's until they have landed. A path
   blocked only by other platforms (a jump past or over one) stays the jumper's, as with `"keep"`. With no landing
   the jumper keeps control; with `brake` the bot first stops on the takeoff, with `retries` the search runs again
   while it stands there (as in the course-list planner). (Step 2 first planned every blocked choice; see the
   amendment under the rules.)
3. With `plan_on` (an opt-in `AssistConfig` field, router mode only; kept by rule 3), a decision made on a plan's
   landing is planned too, whatever the router chose there, and so on while plans land.

The course list only names the two platforms for the search: the platform stood on is the request's `cp`, the
chosen one its `target` (the router's perception equals the course platforms, `tests/test_route.py`), so the
search lands on the chosen platform, its fall line is the lower top of the two, and a choice that is no course
platform (an obstacle's top, where standing ends the episode) is not planned. `prior` and `approach` need the
course list and are refused; the blocked-path rule decides which jumps are planned (`jumps` stays "neo").

## Rules (each written before its run)

### Rule 1 (written 19:15:44, before any run of the mode on a validation or test course): the setting, chosen on family 40 only

- **Fixed parts:** the shipped router (`results/routing/router_policy.pt`, its own fill `policy`); the final
  policy jumping (`results/final_policy.pt`, the router's own jumper, so that its ordinary sets can stay
  unchanged); the planner's perception and limits (`PlannerConfig()` defaults); `RouteConfig(blocked="plan")`.
- **Protocol:** simulator, CPU for the episodes and the searches (as the course-list planner's runs), the
  simulator's own start (settle 0), first episode per course, argmax, stall 200, 4,000 ticks, fall rule `local`.
  The metric is the course-list planner's strict count: completed **and** the neo done as a neo (`neo_done`); the
  plain completions are reported next to it.
- **Candidate A:** the course-list planner's chosen levers (V3b, HANDOFF section 3) as far as they apply without
  the course list: `{"samples": 1024, "iterations": 25, "horizon": 32, "retry_horizon": 48, "retries": 1, "brake":
  14, "restarts": 3}` (jumps "neo", approach 0, prior 0, batch 1, every other field its default).
- **Candidate B:** A plus the course-list plan's next lever, which never ran there (V4, a bigger neo search):
  `"neo_samples": 4096, "neo_iterations": 40`.
- **Runs and choice:** A and B with search seed 0. If they differ by 10 or more courses, the higher is chosen.
  Otherwise both also run with seeds 1 and 2, and B is chosen only if its mean over the three seeds is at least 4
  courses above A's (the 29 Oct lever rule); else A. If the choice was made at seed 0, the chosen setting still
  runs seeds 1 and 2, for the comparison below.
- **Reference (reported; nothing is chosen on it):** the course-list planner, 286 of 320 on validation
  (`results/neo/assist_validation.jsonl`, V3b at seed 0; 276 at seed 1) and 276 of 320 on test
  (`results/neo/assist_test.jsonl`). The chosen setting's three seeds are compared with it.

### Rule 2 (written 19:15:44, with rule 1): after the choice

- **Test, family 41:** the chosen setting once, search seed 0. Nothing is tuned after it.
- **The router's ordinary validation sets:** the chosen setting on families 2 and 5 (300 courses per level,
  1,800 each; `route_eval`'s sets and protocol, episodes on the GPU as the shipped numbers were measured, searches
  on the CPU), compared course by course with the shipped router's 1,762 and 1,720
  (`results/routing/router_validation_route-full-f_it29_*.json`).
- **The bots:** the 64 courses of `results/games/single41-first64-courses.json` with the chosen setting plus the
  course-list planner's bot-start fixes, `"plan_start": 1, "corner_limit": 0.25` (both kept on validation there,
  HANDOFF section 3), CPU for everything (`CUDA_VISIBLE_DEVICES=-1`, as `single41-*-corner`). The offline bot
  first; real Minecraft only if the offline bot equals its `parkour.live` simulator re-run on every course
  (outcome and ticks). Real Minecraft is then checked the same way, with its server corrections.
- **A measurement, no choice:** the bots' setting on family 40 from the bots' start (settle 10), seed 0, so the
  bots' setting has a validation number from the bots' own start.

### Amendment (written 19:45:19): the trigger counts only walls, as `neo_takeoffs` does; rule 1 runs again

- **What happened.** With rule 1's first runs going (candidate A at seed 0 on family 40, `single40-A-s0`, from
  19:31), the check of the router's ordinary sets from rule 2 ran early beside it with A's setting, as a
  measurement (`route2-A-s0`, `route5-A-s0`, 19:34 to 19:37; GPU episodes, CPU searches). Family 2: 1,759 of
  1,800 against the shipped 1,762 and the filter-off what-if's 1,762 (`scripts/neo_routing_ordinary.py`). The 4
  courses lost (311, 761, 818, 1346) are all completed by both, and each was lost on a planned jump. No family-2
  course has a single obstacle block, so every one of its 10 planned jumps was blocked only by other course
  platforms (jumps past or over one). On 818 and 1346 the plan stood on the platform in the way during its
  flight, the router took that platform as visited, and from the landing it found nothing to jump to. Course 761's
  planned choice went back along the course. Family 5: 1,726 against 1,720, with 6 courses won on planned jumps;
  10 of its 12 planned jumps were blocked by obstacle blocks (pillars or beams), 2 only by platforms.
- **Why it is a fix, not a choice.** The mode as briefed treats a blocked choice as a neo, "the same geometry as
  `neo_dagger.neo_takeoffs`", but `neo_takeoffs` counts only non-platform blocks (the course's walls, pillars and
  beams), and the first trigger counted every block. The trigger now makes that test without the course list: a
  block on the straight path (the cells `path_clear` tests) that is not part of a platform in view at its own
  height. On the 32 single-neo types every neo's landing is still walled from its takeoff
  (`tests/test_route_neo.py`). Nothing else changes: the filter stays off for every blocked platform (step 1), and
  the candidates, seeds and choice rule of rule 1 are as written.
- **What runs again.** Rule 1 from the start under the new trigger (names `single40-wall-*`). The runs under the
  first trigger are kept and reported (`single40-A-s0`, `route2-A-s0`, `route5-A-s0`), but nothing is chosen on
  them. Rule 2 is unchanged and uses the new trigger (`route2-wall-*`, `route5-wall-*`).

### Rule 3 (written 19:56:40, before any run with `plan_on`): one more lever, `plan_on`, on rule 1's choice

- **Why.** A's first run (first trigger, seed 0, 241 of 320; nothing is chosen on it) failed 15 courses before
  the neo, 36 at it (no landing found), 17 on the jump right after it, from the plan's landing state, and 7
  later. The course-list planner's V3b at seed 0 failed 4, 24, 4 and 2 there (it plans every jump, so the jump
  after the neo starts from a planned landing too). The search ends a plan the tick it lands, in whatever state;
  the jumper then has to go on from a state its own jumps do not leave it in.
- **Lever C, `plan_on` (opt-in `AssistConfig` field, router mode only):** a decision made on a plan's landing is
  planned too, whatever the router chose there, and so on while plans land; a search that finds nothing leaves
  that jump to the jumper after the brake and retry, as for a neo. The planner then keeps the bot from the first
  neo on. It changes nothing on courses with no wall (family 2), since nothing is planned there.
- **Runs and choice:** rule 1's chosen setting plus `"plan_on": 1`, search seeds 0, 1 and 2 (rule 1's choice has
  those three seeds too). C is kept only if its mean over the three seeds is at least 4 courses above the chosen
  setting's mean (the 29 Oct lever rule). The setting for rule 2 is rule 1's choice, with `plan_on` if kept.

### Rule 4 (written 03:21:45 on 31 Oct, before any run of `plan_ahead`; branch `neo-routing-2`): the router's jumps "all", near walls

Asked for by the main session, for Ethan, after PR #3 was merged.

- **Why.** C (rule 3's choice) loses 15 courses before the neo's takeoff at every seed, and 10 of them on the jump
  onto the takeoff: the platform the course ends on is the one before the takeoff (per course, in C's details
  files). It loses 36, 35 and 40 at the neo. The course-list planner's V3b, which plans every jump, loses 4 before
  the neo and 24 at it. In C every jump before the neo is the jumper's.
- **Lever D, `plan_ahead`** (opt-in `AssistConfig` field, router mode only): a decision whose route crosses a wall in
  a later slot is planned too.
  - The route is the router's three slots, and a slot's jump is blocked by a wall: `RouteRunner.decide`'s
    `walled_chain`, the same test as step 2 of the mode.
  - The plan is the decision's first jump, with the chosen platform as the target, so the one or two jumps that
    lead to a neo become the planner's.
  - With `plan_on`, planning goes on while plans land.
  - On a course with no wall it plans nothing, so family 2 is unchanged by construction.
  - The bots refuse it (not ported).
- **Runs and choice:** C + `"plan_ahead": 1` on family 40, search seeds 0, 1 and 2, in rule 1's protocol (simulator,
  CPU episodes and searches, settle 0, stall 200, 4,000 ticks, fall rule local, the strict count). D is kept only if
  its mean over the three seeds is at least C's mean + 4: at least 258.3, a three-seed total of at least 775 (C:
  255 + 257 + 251 = 763).
- **If kept:**
  - Family 41 once, seed 0, reported as an additional, later test next to C's 252, which stays the headline test
    (`docs/MISSION_ROUTING.md`: a later tested version is reported as additional). Nothing is chosen on it.
  - Then families 2 and 5, a measurement, as far as the night allows. Episodes run on the CPU, since the GPU is the
    main session's tonight; a subsample would be said plainly.
  - Then the bots, offline first, once the lever is ported.
- **If not kept:** it is reported, and no test runs.
- **Compute:** CPU only (`CUDA_VISIBLE_DEVICES=-1`), BelowNormal, mask 0xC03C0F, at most 6 search jobs at once.

## Results

### Rule 1: candidate A (outcome written 23:09:59, before any run of rule 3)

Family 40 (320 single-neo validation courses), simulator, CPU episodes and searches. The count is completed with
the neo done as a neo (`neo_done`); plain completions are in brackets. Rows are in
`results/neo_routing/validation.jsonl` (`run_id` `neo_routing/NAME`), every course in
`results/neo_routing/details/NAME.json.gz`, the parts in `runs/neo_routing_parts/NAME/`.

| Search seed | A (`single40-wall-A-sS`) | B (`single40-wall-B-sS`) |
| --- | --- | --- |
| 0 | 241 (245) | 243 (247) |
| 1 | 238 (241) | 243 (247) |
| 2 | 238 (242) | 240 (243) |
| Mean | 239.0 | 242.0 |

At seed 0 the margin was 2, under 10, so both ran seeds 1 and 2. B's mean is 3.0 above A's, under the rule's 4, so
the choice is **A**. B searched about 4 times as long (19,750 to 20,995 s of searching per run, against 4,728 to
4,878 s for A).

### Rule 3: lever C kept (outcome written 00:11:03 on 31 Oct, before any run of rule 2)

Same set, protocol and files.

| Search seed | A | C = A + `plan_on` (`single40-wall-C-sS`) |
| --- | --- | --- |
| 0 | 241 (245) | 255 (259) |
| 1 | 238 (241) | 257 (260) |
| 2 | 238 (242) | 251 (255) |
| Mean | 239.0 | 254.3 |

C's mean is 15.3 above A's, more than the rule's 4, so `plan_on` is kept. Rule 2's setting is `{"samples": 1024,
"iterations": 25, "horizon": 32, "retry_horizon": 48, "retries": 1, "brake": 14, "restarts": 3, "plan_on": 1}`.
Up to the neo's landing C plays as A, course for course at each seed:
- the same 15 courses lost before the neo (in all six runs);
- the same neo jumps passed (267, 268, 263);
- the same courses lost at the neo (36, 35, 40).

Of the courses that passed the neo, C loses 10, 10 and 10 later, against 24, 29 and 23 for A (per course in the
details files; the course-list planner's V3b lost 6 there, the 4 and 2 in rule 3's "Why"). The rows' `failed_at`
"basic" (25 for C, 38 to 44 for A) also counts the 15 courses lost before the neo.

C makes 4.7 to 4.8 times as many takeoff searches (1,417 to 1,450 per run, against 303) in about 1.6 times A's search
time (7,649 to 7,982 s per run), and every plan played landed as planned (1,396, 1,391, 1,356).

### Where the family-40 courses are lost

Per course, from the details files. `scripts/neo_routing_summary.py` writes `results/neo_routing/breakdown.json`,
which has the router's first choice at the neo's takeoff and how each course ended.

| Search seeds 0, 1, 2 | A | B | C |
| --- | --- | --- | --- |
| Lost before the neo's takeoff | 15, 15, 15 | 15, 15, 15 | 15, 15, 15 |
| Lost at the neo (no landing found) | 36, 35, 40 | 36, 31, 36 | 36, 35, 40 |
| Lost after passing the neo | 24, 29, 23 | 22, 27, 26 | 10, 10, 10 |
| Completed, neo done as a neo | 241, 238, 238 | 243, 243, 240 | 255, 257, 251 |
| Completed, neo not done as a neo | 4, 3, 4 | 4, 4, 3 | 4, 3, 4 |

- **The router finds the neo.** At the neo's takeoff it first chooses the neo's landing on 302 of the 304 courses
  where it decides there, in every run; on the other 2 it chooses a platform further on. The remaining 16 courses
  never decide on the takeoff.
- **Every course lost at the neo is a search that found no landing**, at the takeoff and on its retry. No plan
  failed: every plan played landed as searched.
- **Against the course-list planner** (V3b, seed 0: 4 lost before the neo, 24 at it, 6 after it, 286 completed;
  rule 3's "Why"), C has almost closed the gap after the neo (10 against 6). The rest of the gap is in two places:
  - before the neo, 15 against 4: those jumps are the jumper's, while V3b plans every jump;
  - at the neo, 36 against 24: the bot reaches the takeoff from the jumper's own landing, not from a planned one.

### Test, family 41 (run once, after the choices)

Rule 3's setting (C), search seed 0, same protocol: **252 of 320** completed with the neo done as a neo (254
completed). The row is `neo_routing/single41-wall-C-s0` in `results/neo_routing/test.jsonl`, the courses are in
`details/single41-wall-C-s0.json.gz`.

- **Course-list planner on the same set:** 276 (277 completed), `assist_parts/single41-test-planner` in
  `results/neo/assist_test.jsonl`.
- **The same setting on validation:** 255, 257 and 251.
- **Where the courses were lost:** 19 before the neo, 39 at it (no landing found), 8 after passing it.
- **Searches:** 1,404 at takeoffs (1,332 found) and 70 retries (13 found). All 1,344 plans played landed as
  planned.

### The router's ordinary validation sets (families 2 and 5)

Rule 3's setting (C), search seed 0. Simulator: episodes on the GPU, as the shipped numbers were measured, and
searches on the CPU. Rows `route2-wall-C-s0` and `route5-wall-C-s0` are in `validation.jsonl`. The course-by-course
comparison is in `results/neo_routing/ordinary.json` (`scripts/neo_routing_ordinary.py`), against two references:

- **Shipped router:** `results/routing/router_validation_route-full-f_it29_standard.json` and `..._obstacles.json`,
  fall rule local.
- **Filter off, flags kept:** the 30 Oct what-if, variant `keephead` in the main checkout's
  `runs/neo-route-check-rulecost.json`.

| Family (1,800 courses) | Shipped router | Filter off, flags kept | This mode (C) | Courses searched | Searches |
| --- | --- | --- | --- | --- | --- |
| 2 (standard) | 1,762 | 1,762 | **1,762** | 0 | 0 |
| 5 (obstacles) | 1,720 | 1,720 | **1,728** | 10 | 66 |

- **Family 2** has no walls, so nothing is planned and the run equals the filter-off what-if on all 1,800 courses.
  It differs from the shipped router on 2: course 524 is lost and 1712 is won, both because of the filter.
- **Family 5:** on the 1,790 courses never searched, the run equals the what-if course for course. All 10 searched
  courses are completed (66 searches, every one found and landed as planned); the shipped router completes 3 of
  them. Against the shipped router, the run wins 9 courses and loses 1:
  - lost: 246 (the filter's);
  - won by the filter: 690 and 1661;
  - won by the plans: 1027, 1095, 1105, 1164, 1360, 1438 and 1757.

### The bots' setting from the bots' start (a measurement; nothing is chosen on it)

The bots' setting (C + `plan_start` 1 + `corner_limit` 0.25, seed 0) on family 40 from the bots' start (settle 10);
simulator, CPU: **248 of 320** (250 completed). The row is `single40-wall-C-settle10-bots` in `validation.jsonl`.

- **Where the courses were lost:** 22 before the neo, 40 at it, 8 after passing it.
- **Corner limit:** it dropped 14 landings.
- **Compared with:**
  - C from the simulator's own start (settle 0), seed 0: 255.
  - The course-list planner's bot setting from the bots' start: 292 (`assist_parts/single40-v3b-settle10-planstart-corner`
    in `results/neo/assist_validation.jsonl`).

The run was first started at 00:12 and stopped at 00:13 to keep the CPU under its limit, before any part had
finished. It started again at 01:05 on 3 jobs, so its log has both start lines.

### The bots: the 64 test courses of `results/games/single41-first64-courses.json`

**Setting.** C plus the course-list planner's bot-start fixes: `{"samples": 1024, "iterations": 25, "horizon": 32,
"retry_horizon": 48, "retries": 1, "brake": 14, "restarts": 3, "plan_on": 1, "plan_start": 1, "corner_limit": 0.25}`.
The router fills slots 2 and 3 with its own `policy`, and `--router-blocked` is `plan`, its default with `--assist`.
Everything ran on the CPU (`CUDA_VISIBLE_DEVICES=-1`):
the episodes' re-run, the planner service and the searches. In router mode, `plan_start` only searches a first jump
the router chose across a wall, and does so before the first action.

**Check.** Each run is compared with `parkour.live`'s own simulator re-run of the same courses from the bots' start
(settle 10). The offline bot (offline_bot.js: Mineflayer's JavaScript on prismarine-physics, no server) ran first.
Real Minecraft ran only after the offline bot had matched its re-run on every course.

| | Offline bot | Real Minecraft (1.21.4 server) |
| --- | --- | --- |
| Run | `results/neo_routing/games/single41-offline-routeneo` | `results/neo_routing/games/single41-live-routeneo` |
| Completed | **52 of 64** | **52 of 64** |
| Simulator re-run, same courses | 52 | 52 |
| Courses where outcome or ticks differ from the re-run | 0 | 0 |
| Replaying the recorded actions in the simulator | max 7.1e-15 blocks; 64 of 64 within 1e-6 on every tick | max 2.1e-14 blocks; 64 of 64 within 1e-6 on every tick |
| Planner requests: found, verified, played | 290, 290, 290 of 306; 8 brakes; 0 errors | 290, 290, 290 of 306; 8 brakes; 0 errors |
| Server corrections | (no server) | 0, and no "moved wrongly" in the server's log |
| Wall time | 1,006 s | 1,646 s |

The two bots play the same courses with the same ticks on all 64 (from 00:12 to 00:46 and from 00:46 to 01:30 on 31
Oct). The course-list planner on the same 64 courses completes 55 in both bots (`results/games/single41-offline-corner`,
`single41-live-corner`). The two sets overlap on 44 courses: 11 are completed only with the course list and 8 only
with the router.

### The first trigger's runs (superseded by the amendment; nothing is chosen on them)

These runs used the first trigger, which planned every blocked choice. Rows are in `validation.jsonl`, details in
`details/`.

- **`single40-A-s0`** (A, seed 0, family 40): 241 (245). It has the same outcome and ticks as `single40-wall-A-s0` on
  all 320 courses. It searched one jump more, on course 142 from platform 5 to 6 (at the takeoff and on the retry),
  which the wall trigger leaves to the jumper.
- **`route2-A-s0`**: 1,759 of 1,800. The 4 lost courses are listed in the amendment.
- **`route5-A-s0`**: 1,726 of 1,800.

Both `route*-A-s0` runs used A, with episodes on the GPU and searches on the CPU.

### Follow-ups (not run)

1. **Plan the jumps before the neo too.** 15 courses are lost there, against the course-list planner's 4, and those
   jumps are the jumper's. A lever that plans every router decision from the start while the plans land would be
   the router's version of the course-list `jumps: "all"`. It needs its own rule on family 40.
2. **The searches at the neo.** 36 courses are lost there against the course list's 24, every one with no landing
   found, and B's bigger neo search did not help (3-seed mean +3.0). A plan for the approach to the router's walled
   choice, one decision earlier, would mirror the course list's `approach`, which is refused here because it needs
   the course list.

## Reproduce

The runs used these commands (each run's own command line is the first line of its log, `runs/NAME.log`). C's JSON
is `{"samples": 1024, "iterations": 25, "horizon": 32, "retry_horizon": 48, "retries": 1, "brake": 14, "restarts": 3,
"plan_on": 1, "seed": S}`; the bots add `"plan_start": 1, "corner_limit": 0.25`.

```text
set CUDA_VISIBLE_DEVICES=-1        CPU runs: the single-neo sets and the bots
python -m parkour.route_neo --set single --family 40 --device cpu --name NAME --tag TAG --parts 16 --jobs 4 --assist JSON
python -m parkour.route_neo --set single --family 41 --device cpu ...          the test set (run once)
python -m parkour.route_neo --set single --family 40 --device cpu --settle 10 ...   from the bots' start
python -m parkour.route_neo --set standard --family 2 --threads 1 --name NAME --tag TAG --assist JSON   (and --set obstacles --family 5; episodes on the GPU)
python -m parkour.live --offline --checkpoint results/final_policy.pt --router results/routing/router_policy.pt --assist JSON --courses-file results/games/single41-first64-courses.json --out runs/NAME
python -m parkour.live --checkpoint ... (the same without --offline: real Minecraft, with the user's EULA acceptance on the PC)
python scripts/neo_routing_summary.py --out results/neo_routing/breakdown.json NAME ...
python scripts/neo_routing_ordinary.py route2-wall-C-s0 route5-wall-C-s0 --keep runs/neo-route-check-rulecost.json --out results/neo_routing/ordinary.json
python scripts/archive_neo_routing_games.py runs/NAME ...
```

The project's processes ran on the 10 cores of mask 0xC03C0F. The simulator runs were at BelowNormal priority; the
real-Minecraft run was at normal priority.

- **Rows:** `validation.jsonl`, `test.jsonl`.
- **Every course, search and router decision:** `details/NAME.json.gz`.
- **The parts of each split run:** `runs/neo_routing_parts/NAME/`, copied with the bot runs (`runs/single41-*-routeneo`)
  and the logs into the main checkout's `runs/`.
