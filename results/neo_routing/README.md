# Neo routing: the neo planner without the course list

Branch `neo-routing`, a pull request into `main` for the main session to review and merge. Opt-in code:
`RouteConfig.blocked` (`parkour/route_env.py`), `parkour/route_neo.py`, `parkour.live --router R --assist JSON`,
`bridge/router.js` (`cfg.blocked`), `RouteAssist` in `bridge/assist.js`, the planner service's router requests
(`neo_assist.serve`); tests `tests/test_route_neo.py` and `tests/test_route.py`. The defaults and everything in
HANDOFF section 5 are unchanged. Times are Israel time, read from `date` (Git Bash labels it JST).

## Headline

(Filled in after the runs below.)

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

## Reproduce

```text
python -m parkour.route_neo --set single --family 40 --device cpu --threads 3 --name NAME --tag TAG --assist JSON
python -m parkour.route_neo --set standard --family 2 --name NAME --tag TAG --assist JSON       (and obstacles 5)
python -m parkour.live --offline --checkpoint results/final_policy.pt --router results/routing/router_policy.pt --assist JSON --courses-file results/games/single41-first64-courses.json --out runs/NAME
```

Rows: `validation.jsonl`, `test.jsonl`; every course, search and router decision: `details/NAME.json.gz`.
