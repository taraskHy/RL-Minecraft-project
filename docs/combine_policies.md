# Combining the specialist policies into one network

Status (29 Oct 2026, cloud session): **design only**. Nothing here is
implemented or trained yet. The user chose this direction and asked that
the local session on the training PC implement it and start the training.
The cloud research was stopped early at the user's request, so line
numbers refer to this branch at `8c8bdfa` and the PC branch
`codex/advanced-ice-replay-export-20261029` at `aa476ea` (written `codex:`
below); check them against the merged code.

## The policies to combine

All numbers are simulator results, validation families 2/5 and test
families 1/4, unless stated otherwise.

| Policy | Trained on | Selected | Numbers | Source |
| --- | --- | --- | --- | --- |
| final policy (`results/final_policy.pt`, run `obstacles-g995`) | levels 0-5, obstacles | iteration 699 | validation 97.6%; test 98.4% standard, 95.25% obstacles | `codex:results/operations/g995-initial-comparison.json` |
| `g995-adv` | levels 0-8: neos, momentum (runways), wide jumps | iteration 99 | validation 81.8% (9 levels; levels 0-5: 96.2%); test 82.3% / 78.7% (9 levels), levels 0-5: 97.4% / 94.5% | same file |
| `g995-ice` | levels 0-5, ice on 30% of platforms | iteration 449 | validation 97.3%; test 97.5% / 94.5% (these sets have no ice) | same file |
| `g995-ice-s1` | the same, seed 1 | iteration 599 (at 749 of 1000) | validation 96.2% | `codex:HANDOFF.md` session log, 02:23 |
| `g995-extra` | levels 0-5 + six extra kinds (head hitter, hard head hitter, ice run-up, shaped blocks, slime bounce, ladder climb) | still training: best 699 at iteration 749 of 1000 | validation 89.8% (12 levels) | same |
| neo policy (`codex:results/neo_policy.pt`, run `neo-dagger3`) | levels 0-8 with wide neos, imitation + 3 DAgger rounds, discount 0.99 | iteration 449 | test levels 6/7/8: 80.5 / 46.5 / 71.5%; levels 0-5: 96.5% / 92.5% | `results/neo/one_policy_comparison.json` |

For comparison, the final policy on test levels 6/7/8 makes 71.0 / 3.5 /
1.5% (same file). Every specialist was fine-tuned from the final policy (the
neo policy from the older discount-0.99 line), with the obstacle setting 0.3.

## Facts the design rests on

- **One network, one observation.** Every run above uses hidden layers
  512-256-128 and the standard 255-number observation (`extended_obs` off,
  nothing hidden; `codex:results/runs/*/config.json`). So one student can
  start from any of them, and their action distributions over the same
  five heads (3, 3, 2, 2, 7) can be compared directly.
- **The observation has no block types** (`parkour/env.py`, `OBS_LAYOUT`
  and `Observer`): ice, slime and ladders look like stone; only shaped
  blocks change the scanned heights. A policy feels ice only through the
  physics. The final policy, never trained on ice, still completes 95.6% of
  the all-ice test courses (98.4% on stone; `results/ice_preview.json`,
  family 1).
- **Validation and test have no ice.** `parkour/train.py` builds them with
  `held_out_courses(...)` without `ice_prob` (the flag's comment: "training
  courses only"), so `g995-ice` was selected without ever being measured
  on ice.
- **Course mixing.** `courses.generate` (`parkour/courses.py`, line 569)
  takes half of a course's jumps from its level and half from *all* easier
  levels. One table with `--advanced --extra` would therefore put neos and
  runways into the extra-level courses (levels 9-14), which the extra
  specialist never saw.
- **Imitation already exists** as hard-label behaviour cloning:
  `--bc-pairs`, `--demos`, `--bc-coef`, `--bc-decay`, a loss
  `-log pi(a | s)` added inside the PPO minibatch loop (`parkour/train.py`,
  around lines 424-431).
- **The rollout** stores the student-normalised observation (`obs_n`); the
  raw one is `obs`. Each environment's course is `env.course`, its level
  `env.pool.level[env.course]`, read before `env.step` (which resets
  finished environments).
- **The PC queue** (`codex:scripts/training_queue.py`, lines 162-172;
  `codex:configs/training_queue.json`) starts `python -u -m parkour.train
  --run NAME --init-from INIT --obstacle-prob 0.3 --gamma 0.995
  --iterations 1000 --test-once --compact-ice --eval-batch-size 256` plus the
  job's flags (and `--num-envs 8192 --rollout 64 --minibatches 32` when two
  run together), and after the listed jobs it starts further seeds of the
  best recipe on its own (`next_seed`).
- Seed families in use: 0 training, 1 test, 2 validation, 3 stress, 4/5
  obstacle test/validation, 11 analysis, 12/13 random neo, 14/15 mixed.
  **16 and 17 are free** on both branches.
- A run of 1000 iterations takes about 1.5-2.5 hours on the PC (120-140k
  steps/s, 524,288 steps per iteration, two runs sharing the GPU).

## The decision

**One network**: the user chose distilling the specialists into one student
(on-policy, next to PPO) over a per-jump switcher (the bots would have to
load several networks and classify every jump; a switch in mid-run can
fail) and over plain training on every skill (kept as the control run).
Block types in the observation are not added now: the teachers do not see
them either, and the observation is on the do-not-touch list (both bots
mirror it). Revisit only if the per-group validation shows that one network
cannot match all teachers.

## Design (every change opt-in; the defaults change nothing)

### 1. Skill groups: `--skills basic,advanced,extra`

Each training course comes from one specialist's own distribution, so every
jump on it is something its teacher trained on:

| Group | Table and levels | Global level ids | Ice | Teacher |
| --- | --- | --- | --- | --- |
| basic | `load_levels()`, levels 0-5 | 0-5 | `--ice-prob` (0.3), as `g995-ice` | best on basic + ice validation |
| advanced | `load_levels(advanced=True)`, levels 6-8 | 6-8 | none | best on advanced validation |
| extra | `load_levels(extra=True)`, levels 6-11 | 9-14 | none | best on extra validation |

- In `CourseFactory.make`, pick each course's group by its global level,
  call `C.generate` with that group's table and local level (unchanged
  function), and store the global id in the course's `level` (the pool, the
  environment's per-level statistics and the curriculum all read it; they
  then see 15 levels).
- Obstacles 0.3 on every group, as all specialists.
- `--curriculum uniform`: the adaptive frontier waits for 60% success on
  the edge level and `g995-adv` sits near 46% on level 7, so later levels
  would never be reached.

### 2. Teachers: `--teachers basic=CKPT,advanced=CKPT,extra=CKPT`, `--distill-coef`, `--distill-decay`

- Load each checkpoint like `--init-from` does into a frozen `ActorCritic`
  with its own `RunningNorm` (eval mode, no gradients, never updated).
- In the rollout, before `env.step`: each environment's group from its
  global level; for each teacher, log-softmax of its five heads on
  `teacher.norm(obs)` (the raw observation) for the environments of its
  group. Store the 17 numbers per step (`T x N x 17`, about 36 MB at 8192 x
  64) and a mask.
- In each PPO minibatch: student logits on `b_obs[idx]` (already
  student-normalised), then
  `loss += w * mean_over_masked_rows(sum_over_heads KL(teacher || student))`,
  with `w = distill_coef`, falling linearly to 0 over the run with
  `--distill-decay` (the `--bc-decay` pattern). Log `distill_kl` in
  `train_log.csv`.
- Write the teacher paths into `config.json`; a resumed or paused run
  reloads them.

### 3. Validation and test in skills mode

- The held-out sets are the specialists' own sets, so every per-group
  comparison is on identical courses: basic = `held_out_courses(load_levels(), ...)`;
  advanced = the advanced table's sets, levels 6-8 only; extra = the extra
  table's sets, levels 6-11 only; each with its standard and obstacle
  families (validation 2/5, test 1/4) and the global level ids.
- New **ice sets**: the basic table, levels 0-5, `ice_prob=0.3`, validation
  family **17**, test family **16**, as a separate group (ids 15-20). Add
  their fingerprints to `tests/test_heldout_fingerprints.py`.
- `best.pt` = the highest **macro average over the groups basic, ice,
  advanced and extra** on validation. This rule is fixed before any run.
  Write per-group validation columns to `eval_val_log.csv`.

### 4. Cross-evaluation: `python -m parkour.skills matrix --candidate NAME=CKPT ...`

- Validation only by default (a `--split test` option is for the one-time
  report after training). It reuses the group sets of section 3 and
  `evaluate_course_batches` / `evaluate`, and writes
  `results/skills/validation_matrix.json`: every candidate's success per
  group and per level, and the choices below.
- Rules, fixed now: each group's teacher is the candidate with the highest
  validation success on that group (basic: the mean of basic and ice); the
  student starts from the candidate with the highest macro average.
- Candidates: final policy, `g995-ice`, `g995-ice-s1`, `g995-adv`, the neo
  policy, `g995-extra` (after it finishes; its validation-selected
  `best.pt`). Run it on the PC, where the checkpoints are.

### 5. The two runs (the queue's two slots, after `g995-extra` and `g995-ice-s1`)

| Run | Job flags (plus the queue's common flags) |
| --- | --- |
| `combined-distill` | `--skills basic,advanced,extra --ice-prob 0.3 --curriculum uniform --teachers basic=...,advanced=...,extra=... --distill-coef 1.0 --distill-decay --pool-size 1024` |
| `combined-union` (control) | the same without `--teachers`, `--distill-coef`, `--distill-decay` |

Both start from the checkpoint the matrix chose. The queue sets
`--init-from` itself, so check how it picks `INIT` and give these jobs
their own. Keep the queue from starting further seeds in these slots.

### 6. After training

1. `best.pt` by the validation macro average (section 3), in the run.
2. The test sets once (`--test-once` does it at the end): per group, on the
   same courses as each specialist's test numbers; one test of the
   specialists on the new ice test set (family 16) for the comparison.
3. Downstream checks with the winner: the neo course sets with and without
   the planning assist (`parkour.neo_assist`, validation 2 first, test 1
   once), the mixed sets (validation 15, test 14), and the offline bot on
   `configs/neo_live_courses.json`.
4. Export the winner as `results/combined_policy.pt` / `.json` with a
   provenance file; never replace `results/final_policy.*`. Then
   `run.cmd report`, TEAM_STATUS.md and HANDOFF.md. Real Minecraft is the
   user's (their EULA on that PC).

## Tests

- Defaults unchanged: the held-out fingerprints pass as they are; with no
  new flag the factory makes the same courses and the training loop draws
  the same random numbers.
- The KL term is 0 when the student equals the teacher, and a few
  distillation-only steps lower it.
- Group-local generation: advanced courses contain no extra-kind specs,
  extra courses no neo or runway specs, global ids map back to the right
  table and level.
- Teacher assignment by global level, including after a pool refresh.
- A tiny CPU run with teachers (the pattern of
  `codex:tests/test_training_lifecycle.py`: `num_envs` 4, `rollout` 2,
  `hidden` (8,), one held-out course per level), including pause and
  resume.
- Then the whole suite (`run.cmd test`).

## Risks to watch

- The KL term against the PPO loss: watch `distill_kl`, the per-group
  validation, and the `stopped` / `updates` columns. The PPO KL early stop
  (`kl_stop` 0.05) and the KL-adaptive learning rate limit how fast the
  student can move towards the teachers.
- The teachers are blind to ice; ice stays on basic courses, where the ice
  specialist learned it.
- Courses that mix groups (a neo next to a slime bounce) are not trained;
  the mixed sets 14/15 measure them.
- The neo policy comes from the discount-0.99 line: only its actor serves as
  a teacher, so that does not matter for the KL term.

## Do not

Change default course generation, seed families 0-5, the 255-number
observation, `RewardConfig` or PPO defaults, or `results/final_policy.*`
(HANDOFF.md section 5). Choose only on validation families, and report test
families once.
