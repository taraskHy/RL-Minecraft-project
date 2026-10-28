# Design: learning Minecraft parkour that transfers to unseen courses

## 1. Task

A course is a chain of stone platforms floating in the void (Minecraft Java
1.21.4). The player starts on platform 0 and must jump along the platforms
until it stands on the last one. Landing on a later platform skips the ones
in between. Courses are generated procedurally, so every evaluation course is
one the agent has never seen. Success means
standing on the goal platform; falling below the course or standing still for
10 seconds ends the episode.

## 2. Why a simulator, and why it is faithful

A Mineflayer bot moves by running **prismarine-physics** on the client: each
50 ms tick it computes its own new position from its controls and the blocks
around it, and the server accepts that position. So the movement a policy
experiences in real Minecraft *is* prismarine-physics.

`parkour/physics.py` ports that engine (version 1.11.1, the one Mineflayer
4.39.0 uses) to batched PyTorch, keeping every expression and its operand
order in float64, with one deliberate change that the bot makes too
(`bridge/physics_patch.js`): block collisions use vanilla Minecraft's 1e-7
tolerance (`COLLISION_EPSILON`). Without it, rounding can let a player resting
against a block face slip into the block. The port covers what the courses contain:
air and full cubes with default slipperiness (walking, sprinting, jumping,
sprint-jump boost, air control, drag, gravity, axis-by-axis collision,
step-up, the jump cooldown). Ice is also ported but off by default: with an
optional per-block slipperiness grid (`World(slip=...)`, ice 0.98) the ground
friction comes from the block under the feet, matching the real library to
7.1e-15 blocks over 60,000 ticks (`results/physics_parity_ice.json`); no
course uses it yet.

Evidence (`results/physics_parity.json`, run through Node on the real library
with the same tolerance patch): on 24,000 random ticks on the CPU and 75,000
on the GPU, in random block fields with pillars, steps, ceilings, gaps and
edges, the largest position error was 1.4e-14 blocks and every on-ground,
collision and jump-cooldown flag matched. `tests/test_physics_parity.py` runs
a smaller version (25,200 ticks).

One Mineflayer detail: a 1.21 server sends its own sprinting speed modifier,
under a different id from the one prismarine-physics adds, so if
prismarine-physics read it the bot would sprint 1.3 x 1.3 times as fast as
walking instead of 1.3. With the installed minecraft-data (3.117.0) it does
not: an out-of-date attribute table stores the server's movement speed under
the key `generic.step_height`. So the bot sprints at 1.3 times walking speed,
like the vanilla client and the simulator (live replays match to 2.1e-14
blocks). The bot also deletes any movement-speed attribute after each tick as
a safeguard; in the live runs there was none to delete.

## 3. MDP

- **Decision rate:** one action per game tick (20 per second), as precise as
  a human player and the same rate the bot acts at.
- **Actions** (factored, 3 x 3 x 2 x 2 x 7 = 252 combinations): move
  back/none/forward, strafe left/none/right, jump, sprint, and a yaw step of
  -20, -6, -1.5, 0, 1.5, 6 or 20 degrees.
- **Observation** (255 numbers, horizontal quantities in the player's frame):
  - velocity, on-ground, jump cooldown, horizontal collision;
  - the previous action;
  - the platform last stood on and the next three: the four corners of each
    top face, its height above the feet, whether it is the goal;
  - a height scan of 98 points from 1 block behind to 5.5 blocks ahead and
    1.5 blocks to each side: floor height and a ceiling flag;
  - the goal's position and the fraction of platforms reached.
  The bot computes the same numbers from what the game shows it
  (`bridge/policy.js`); `tests/test_bridge_parity.py` checks the two
  implementations agree to float32 rounding.
- **Reward** per tick (`parkour/env.py`, `RewardConfig`):
  - *progress* (weight 1): decrease of the remaining path length, i.e. the
    distance to the next platform's top centre plus the chain of centre
    distances from there to the goal. This is the "distance to the goal"
    idea, measured along the course so that winding courses work.
  - *landing* (+1 per new platform): the "number of jumps" idea.
  - *goal* (+10) on standing on the last platform.
  - *fall* (-5) when the feet drop one block below every nearby platform top,
    which also ends the episode: the "height" idea. There is no way back up,
    so ending early saves samples.
  - *time* (-0.01 per tick).
- **Termination:** goal or fall. **Truncation** (bootstrapped): 200 ticks
  without reaching a new platform.

## 4. Which jumps exist: the feasibility study

`parkour/feasibility.py` builds every jump type (gap 0 to 5 air blocks,
height change -3 to +1, sideways offset 0 to 2, landing platform 1x1, 2x2 or
3x3) with a 1x1 takeoff block, and simulates 4,032 simple scripted strategies
for each (start position, aim point, jump tick, sprint or walk, when to stop
pressing forward). The fraction that lands is the jump's *forgiveness*. 194 of
243 jump types are possible from a standing start; the others (for example a
5-block flat gap) are never generated. Forgiveness orders the jumps into six
difficulty levels (`configs/jump_levels.json`), for example:

| Jump (1x1 landing) | Forgiveness | Level |
| --- | --- | --- |
| flat, 1-block gap | 0.47 | 0 |
| flat, 2-block gap | 0.15 | 1 |
| up 1, 2-block gap | 0.12 | 2 |
| flat, 3-block gap | 0.08 | 3 |
| up 1, 3-block gap | 0.02 | 4 |
| flat, 4-block gap | 0.01 | 5 |
| flat, 5-block gap | 0 | impossible |

## 5. Courses and curriculum

`parkour/courses.py` chains 5 to 12 jumps (training) or 10 (test), and ends a
course early when the next jump does not fit (training keeps courses with at
least 2 jumps). Half the jumps of a level-L course come from level L and half
from easier levels. Courses turn left or right with probability 0.25 per
jump, keep one air block from every other platform and keep each new jump's
flight path clear of the platforms already placed. A later platform can still
sit in an earlier flight path (132 of the 11,850 test jumps, almost always the
next platform). The start yaw is the direction of the first jump plus up to
60 degrees of noise.

The adaptive curriculum (`LevelSampler`) trains on levels up to a frontier
(weight 1 each) plus the next level (weight 3), and moves the frontier up
when the success rate on that next level reaches 60% over at least 2,000
episodes. The training pool holds 8,192 courses; one eighth is replaced with
new courses every 20 iterations.

**Held-out test set:** 200 courses per level (1,200), 10 jumps each except 49
that end early with 2 to 9, generated from a separate seed family, and
checked to share no course with training.
Every training run is evaluated on the same set with the deterministic
(argmax) policy.

## 6. Learning algorithm

PPO (course material: policy gradients, actor-critic, GAE) with separate
actor and critic MLPs (512-256-128, ELU), observation standardisation, 16,384
parallel environments, 32-tick rollouts (524,288 samples per update),
4 epochs of 8 minibatches, learning rate starting at 3e-4 (adaptive, below), clip 0.2, entropy 0.005,
discount 0.99, GAE 0.95. A tabular method cannot generalise to unseen
courses: the state is continuous and the course geometry changes every
episode, so function approximation is required.

**Stability.** The first runs (kept in `runs/_v1_unstable`, not used for
results) made a catastrophic update three times: the approximate KL between
the old and new policy jumped from about 0.004 to 0.3 or 1.3 (and once
overflowed), success fell from about 75-80% to 15-17% within two iterations,
and getting back within 5 points of the earlier level took 20 to 100
iterations. This is the known PPO failure where actions the old
policy rarely chose get huge probability ratios: with a negative advantage the
clipped objective does not bound them. The trainer now uses four standard
safeguards: the log-ratio is clamped to +-20; dual clipping bounds the
objective at 3A for negative advantages A (Ye et al., 2020); an update stops
as soon as a minibatch's KL exceeds 0.05; and the learning rate adapts
between 1e-5 and 1e-3 to keep the per-minibatch KL near 0.01 (the schedule
of rsl_rl / legged_gym). All reported runs use this trainer. In the 3,000-
iteration main run the update stop fired 45 times; once (iteration 1898) the
two or three steps taken before it fired, at the learning-rate ceiling of
1e-3, still caused a short dip (training success 94% to 70%, recovered within
about 5 iterations, against 20 to 100 before the safeguards). A lower ceiling
(3e-4) is the next step for final-model training; the ablation runs keep 1e-3
so that their arms stay comparable.

The GPU is used for both the simulator and the network. The step makes no
host synchronisation, which gives about 1 million environment ticks per
second on one RTX 5080 (about 450 thousand per second including PPO
updates).

## 7. From simulator to Minecraft

`bridge/parkour_bot.js` runs the exported network (`policy.json`) inside the
bot's physics tick: after each tick it reads the bot's state and the blocks
around it, chooses an action, and sets the yaw and controls for the next
tick, exactly the order of the simulator's step. Course building and the
start teleport use operator commands; they are the environment's reset, not
part of the policy.

`bridge/offline_bot.js` runs the same JavaScript episode loop on
prismarine-physics without a server. `parkour.live` compares either mode with
the simulator (tick-by-tick replay of the recorded actions, and the outcome
of the same policy on the same course in the simulator).
