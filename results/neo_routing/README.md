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
2. When the router's first choice is such a platform, the jump to it is a neo (the line passes through a wall,
   the test `neo_dagger.neo_takeoffs` makes on the course list): the neo planner (`neo_assist.plan_once`)
   searches it from the bot's exact state with the chosen platform as the target, and the actions found replace
   the jumper's until they have landed. With no landing the jumper keeps control; with `brake` the bot first stops
   on the takeoff, with `retries` the search runs again while it stands there (as in the course-list planner).

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

## Results

(Filled in as the runs finish.)

## Reproduce

```text
python -m parkour.route_neo --set single --family 40 --device cpu --threads 3 --name NAME --tag TAG --assist JSON
python -m parkour.route_neo --set standard --family 2 --name NAME --tag TAG --assist JSON       (and obstacles 5)
python -m parkour.live --offline --checkpoint results/final_policy.pt --router results/routing/router_policy.pt --assist JSON --courses-file results/games/single41-first64-courses.json --out runs/NAME
```

Rows: `validation.jsonl`, `test.jsonl`; every course, search and router decision: `details/NAME.json.gz`.
