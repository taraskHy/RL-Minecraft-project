# Parallel-run contention benchmark, 24 October 2026

**REAL MINECRAFT NATURAL RESOURCE TASK; an engineering check, not a learning
result; NOT FULL SURVIVAL COMPLETION.** The question, worlds, order and
adoption rule were committed before any run, in
[`configs/parallel_contention_benchmark.json`](../../configs/parallel_contention_benchmark.json).
The question: **do several live episodes at once on this PC change what the
execution helpers achieve, through CPU contention on their wall-clock
limits?**

- **What ran.** The masked random baseline, frozen, with no learning:
  `--max-steps 60 --pause 0`, on worlds 30000–30019, in four chunks of five
  worlds with agent seeds 0 to 3.
- **Order.** 1 slot, then 4 slots, then 2 slots. Each schedule ran alone on
  the PC. 4 slots fails criterion (d), so the declared order called for the
  2-slot run, which started 19 s after the 4-slot schedule ended.
- **Code.** All three schedules ran on the run code of commit `88c8ec7` (the
  parallel-run merge). The 1-slot schedule started at `88c8ec7` and ended at
  `5fbf293`, a docs-only commit; the scheduler compares the committed run-code
  tree and recorded no change.
- **Evidence.** Three schedule folders, each holding its `plan.json`,
  `schedule.json` and nested runs, with SHA-256 digests in `retention.json`
  (486 files). `report/` was built by `resource-report` from the retained
  copies; it keeps each slot count in its own group.
- **PC.** AMD Ryzen 5 5600G (6 cores, 12 threads), 63.4 GB RAM, Windows 11.

## Results

| Slots | Wall time, 20 episodes | Successes (Wilson 95%) | Deaths | Truncated | Mean decisions (all / success) | Mean task s (all / success) | Task s per decision | `exploration_time_budget` events | Mean world startup s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 22 m 55 s | 19/20 (0.76–0.99) | 0 | 1 | 17.45 / 15.2 | 50.2 / 49.9 | 2.88 | 0 | 16.5 |
| 2 | 14 m 35 s | 17/20 (0.64–0.95) | 0 | 3 | 20.85 / 13.9 | 58.8 / 43.3 | 2.82 | 2 | 18.3 |
| 4 | 12 m 59 s | 18/20 (0.70–0.97) | 1 | 1 | 16.25 / 13.6 | 59.9 / 40.8 | 3.69 | 2 | 27.0 |

"Task s per decision" is the mean task seconds over completed episodes divided
by the mean decisions, as the declaration defines it. World startup is
`elapsed_wall_seconds − seconds` per completed episode.

- **No invalid episodes.** All 60 episodes were measured, with no busy port,
  server-startup timeout or bridge timeout.
- **Clean schedules.** No port was retired. Each schedule's
  `max_running_observed` equalled its slot count.
- **How long each load level lasted.** The 4-slot schedule ran four episodes
  at once for only 272 s of its 779 s. Chunk 1 ran alone for its last six
  minutes, and the schedule ran a single episode for 372 s in all. The
  2-slot schedule ran two at once for 709 s of 875 s. So the 4-slot figures
  mix four-way and much lighter load, and they understate sustained four-way
  contention; it still failed the rule.

## Adoption rule

A slot count N is adopted only if, against 1 slot on the same 20 worlds:

- (a) no episode is invalid because of the parallel setup;
- (b) successes are at least the 1-slot count minus 2;
- (c) `exploration_time_budget` events are at most the 1-slot count plus 2;
- (d) mean task seconds per decision are at most 1.25 times the 1-slot value.

| Criterion | 4 slots | 2 slots |
| --- | --- | --- |
| (a) invalid from the parallel setup | 0: pass | 0: pass |
| (b) successes ≥ 19 − 2 = 17 | 18: pass | 17: pass (at the limit) |
| (c) budget events ≤ 0 + 2 = 2 | 2: pass (at the limit) | 2: pass (at the limit) |
| (d) task s per decision ≤ 1.25 × 2.88 = 3.60 | 3.69 (1.28×): **fail** | 2.82 (0.98×): pass |
| Verdict | not adopted | **adopted** |

**On this PC, live episodes may run two at a time.** As the declaration says,
an adopted N is used for whole experiments only. Every arm of one experiment
runs with the same N, in one schedule, with `interleave` and
`evaluate_after_all_training`, and the report never compares runs with
different slot counts.

## What else the runs show

- **World 30000 changed most.** It succeeded in 31 decisions (140 s) at 1
  slot.
  - At 2 slots it used all 60 decisions and failed (361 s;
    `table_not_placed`, last event `explored_locally`).
  - At 4 slots it also used all 60 and failed (327 s; `pickaxe_not_crafted`,
    last event `table_path_blocked`).
  - Each decision took somewhat longer: 4.5 s at 1 slot, then 6.0 s and
    5.5 s. Most of the extra time, though, is the extra decisions after a
    different trajectory. The bot collected three logs at 1 slot and one at
    2 and 4.
  - It spawned at a different point each time, so the change cannot be
    attributed to contention.
- **Per-world failures.**
  - World 30008 failed at both 1 and 2 slots: `leaf_escape_refused` at 1
    slot, and no wood collected at 2 slots.
  - At 2 slots, world 30012 also failed: stranded on a canopy.
  - The only death in the benchmark was at 4 slots, on world 30006: the
    server log records "RLResourceBot was slain by Spider" during a `SEARCH`.
    The rule has no death criterion, but this episode counts as one of the
    two 4-slot failures in (b). It also gave one of the two 4-slot
    `exploration_time_budget` events in (c): the final `SEARCH` ran its full
    36 s budget after the bot was already dead. Whether contention slowed the
    bot's reaction cannot be told from one event.
- **Startup cost.** World startup rose from 16.5 s at 1 slot to 18.3 s at 2
  slots and 27.0 s at 4.
  - At 4 slots the extra time is spread over JVM launch, server start-up and
    the bot's join, not mainly world generation: the server-reported
    spawn-area time rose only from 2.8 s to 3.9 s.
  - It is largest for the first episodes of the four chunks, which started
    together. Later episodes that ran alone took 15.5 to 16.8 s.
  - This fits several servers competing while they start, but CPU use was
    not measured.
- **Speed-up.** 2 slots finished the 20 episodes 1.57 times faster than 1
  slot, and 4 slots 1.77 times faster. With four chunks, the slowest chunk
  sets the 4-slot wall time: chunk 1, holding world 30000, ran 13 minutes.

## Limits

- **An engineering check on 20 worlds, not a statistical test.** Random
  choices, spawn points and server timing make outcomes noisy, and the
  thresholds are declared tolerances.
- **Two criteria were met at the limit.** At 2 slots, successes and
  exploration-budget events were exactly at their limits.
- **What it does not show.** It does not show that contention is harmless for
  other helpers, tasks, policies or machines. A run that mostly crafts may
  load the CPU differently from one that mostly searches; this benchmark did
  not measure that.
- **Uneven load at 4 slots.** Uneven chunk lengths meant the 4-slot schedule
  ran fully loaded for only about a third of its time. A future benchmark
  could use more, shorter chunks to keep every slot busy.
- **Other machines.** Rerun it, with new world seeds declared first, before
  trusting parallel runs on another PC, such as the RTX 5080 machine. The
  GPU plays no part in this.
