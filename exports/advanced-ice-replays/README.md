# Advanced and ice replay export

These are **new offline showcase recordings of the validation-selected policies**, made on 29 October 2026. They are not original PPO training trajectories, held-out benchmark results, or recordings from a running Minecraft server. Both successful and failed episodes are included without selecting on the outcomes.

## What to load

- `advanced.json`: 27 episodes, levels 0-8, `g995-adv` checkpoint iteration 99.
- `ice.json`: 18 episodes, levels 0-5 with 30% ice probability, `g995-ice` checkpoint iteration 449.
- Each JSON is self-contained and includes metadata, geometry and every recorded frame. Concatenate their `episodes` arrays to display both policies.
- The downloadable attachment `replays.json` combines both datasets in the same schema.
- `manifest.json` records source commit, checkpoint hashes, file hashes and integrity checks.
- `verification.json` records CPU replay checks against the Python physics port. Three episodes (ordinary, advanced level 7, icy level 4), 538 ticks, matched exactly in position.

No GPU, Python, model checkpoint or Minecraft server is needed to animate the recorded frames. A cloud artifact can embed these JSON data directly. The recorder is included as `export_replays.py` for provenance; rerunning it requires the training PC's selected checkpoints and repository dependencies.

## JSON schema: parkour-replay-v1

Top-level `episodes` holds all replays. Each episode has `id`, `run`, `checkpoint_iteration`, `title`, `course`, `result`, `success`, `ticks`, `duration_seconds`, `platform_reached`, `actions`, and `frames`.

Coordinates use Minecraft world axes: **Y is up**, and one unit is one block. Positions are the **bot's feet**. Draw a 0.6 x 1.8 x 0.6 player with its center 0.9 blocks above that position.

Each `course.platforms` entry is `[x_min, x_max, z_min, z_max, block_y]`, with inclusive block coordinates. A rectangular platform has width `x_max-x_min+1`, depth `z_max-z_min+1`, height 1, and center `[(x_min+x_max+1)/2, block_y+0.5, (z_min+z_max+1)/2]`. Its top is at `block_y+1`. Draw `course.extra` entries as obstacle cubes at `[x+0.5,y+0.5,z+0.5]`. Entries in `course.ice` are **platform indices**, not coordinates; color those platforms as ice. `course.level`, `seed`, `name`, `spawn`, and `spawn_yaw` identify the course. The last platform is the goal.

Each `frames` row has columns:

`[x, y, z, vx, vy, vz, yaw, on_ground, jump_ticks, platform_reached]`

`frames[0]` is the initial state. `frames[t+1]` is the state after `actions[t]`. Each tick is **0.05 seconds** (20 ticks/second), and `time_seconds = frame_index/20`. Velocities are blocks/tick. Yaw is radians; forward is `[-sin(yaw),0,-cos(yaw)]`. Interpolate positions for smooth rendering and interpolate yaw across the shortest angular distance. Preserve the recorded timeline and result.

Actions are `[move, strafe, jump, sprint, turn_index]`: move 0/1/2 = backward/none/forward; strafe 0/1/2 = left/none/right; jump and sprint are 0/1; the turn index selects `[-20,-6,-1.5,0,1.5,6,20]` degrees.

## Suggested artifact

Reuse the previous Minecraft artifact's visual style. Provide selectors for policy, level and course; play/pause, restart, speed and timeline controls; orbit and follow cameras; a trajectory trail; ice/obstacle colors; and a HUD with tick, position, difficulty, reached platform, policy checkpoint and outcome. Label it **Recorded offline simulation** and **Showcase courses**. Keep failures visible and let the viewer inspect them. Display only statistics computed from this bundle as showcase statistics, not as the final test scores.

The recorded paths are the source of truth. Do not generate substitute movement or rerun physics in the visualization. The training PC and its two active training jobs are not needed for playback.
