# Challenge 7.5: G1 Hazard Maze — development notes

This document describes what was developed in **`g1_hazard_lab`** compared with the Week 7 workspace it started
from, **`simulation-for-humanoid/Week07_Curriculum_Maze/g1_maze_lab`** (called *the original* below). It covers
the methods, algorithms and code of the two parts the challenge lets a team submit:

| Part | File(s) | Role |
|---|---|---|
| Walking policy | `policy.pt` ← `logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt` | "legs": turns `(v_x, v_y, w_z)` commands into joint targets |
| Pilot | [`pilot.py`](pilot.py) | "brain": reads the sensors, maps the maze, plans the route, sends velocity commands |

---

## 1. Summary of results

All runs use `scripts/challenge.py` with the official rules: spike beds are deadly; a fall, any torso/pelvis/elbow/palm
contact, or < 0.25 m of movement in 10 s sends the robot back to the start while the clock keeps running.

| Run (`PILOT_MODE=short`) | Original-style setup ¹ | Final setup ² |
|---|---|---|
| Level 7, seed 2026 | 111.7 s, 2 falls | **62.1 s, 0 falls** |
| Level 7, seed 1 | 68.2 s, 0 | **64.4 s, 0** |
| Level 7, seed 2 | 104.5 s, 0 | **107.4 s, 0** |
| Level 7, seed 3 | 92.4 s, 0 | **90.0 s, 0** |
| Level 7, seed 4, random ends | 114.0 s, 1 | **55.6 s, 0** |
| Level 9, seed 1 | 129.6 s, 0 | **121.7 s, 0** |
| Level 9, seed 2 | did not finish (6 falls, 3 slalom spike hits) | **101.4 s, 0** |
| Level 9, seed 3 | 173.2 s, 1 | **108.3 s, 0** |
| Level 9, seed 5, random ends | 189.1 s, 1 ³ | **127.8 s, 0** |

¹ Level 7 seed 2026: `auto` mode with the earlier `model_6799.pt` policy. Other rows: the new policy `model_8497.pt`
with the pilot *before* the slalom, speed and curved-turn work (Section 4.5–4.7).
² `model_8497.pt` + the final `pilot.py`.
³ This run already had the first slalom change (sideways strafe), but not the knee offset.

For reference, the challenge slides report that the course's `DefaultPilot` did not finish level 7 seed 7
(51 falls, 49 of them spike hits), and that the instructor's own solution did not finish level 9 or the sample final mazes.

The pilot now runs in about **0.6× real time** (a 100 s run takes about 60 s of wall clock). Before the speed-up it
ran at 1.4–2.4×, which risked the runner's **18-minute wall-clock limit** on a 600 s final run.

---

## 2. What changed, file by file

| File | Original (`g1_maze_lab`) | Now (`g1_hazard_lab`) |
|---|---|---|
| `pilot.py` | 724 lines. Hazard-aware pilot, first version (4 layers, see Section 4). | 961 lines. Reworked perception, memory, planning triggers, slalom, curved turns and CPU execution. |
| `src/g1_maze_lab/tasks/curriculum/mdp/curriculums.py` | 3 curriculum terms (reward-weight ramp, adaptive push, command range). | + `terrain_levels_skill`, a terrain curriculum that works with turning commands. |
| `src/g1_maze_lab/tasks/curriculum/mdp/__init__.pyi` | Exports the 3 terms. | Also exports `terrain_levels_skill`. |
| `src/g1_maze_lab/tasks/hazard_starter/env_cfg.py` | `G1-Hazard-Starter` / `G1-Hazard-Pro`: maze-shaped trenches, ramps, boxes + rough ground. Uses Isaac Lab's `terrain_levels_vel`. | Same terrain; the terrain curriculum is replaced by `terrain_levels_skill`. |
| `scripts/challenge.py` | — | Loads `./pilot.py` by default when testing locally; shows the scanner rays in the viewer. Rules unchanged. |
| `tests/test_pilot_review.py` | — | New: 8 regression tests for the pilot. |
| `tests/test_terrain_curriculum.py` | — | New: 4 tests for the curriculum term. |
| `pyproject.toml` | — | Only the relative path to `IsaacLab` (different folder depth). |

The maze generator (`terrains/hazard_maze.py`), the challenge rules, the sensors and the policy's network shape are
**unchanged**, as the challenge requires.

---

## 3. Walking policy: fixing the terrain curriculum

### 3.1 Problem

`G1-Hazard-Pro` trains on four kinds of 8 m × 8 m tiles across 10 difficulty rows. For a tile in row *r*, Isaac Lab
sets the difficulty to about `(r + U(0,1)) / 10`. On the ramp tiles the height grows with difficulty from 0.08 m to 0.30 m:

```
ramp height = 0.08 + difficulty * (0.30 - 0.08)
```

So the maze's **steep ramps (0.24–0.30 m) only exist in rows 7–9**.

The TensorBoard log of the original training run (`g1_hazard_pro/2026-10-31_14-25-28`) showed:

| Iteration | 5600 | 5900 | 6200 | 6500 | 6799 |
|---|---|---|---|---|---|
| `Curriculum/terrain_levels` (average row) | 2.46 | 0.41 | 0.41 | 0.73 | **0.92** |
| Velocity-tracking reward | — | 0.76 | 0.81 | 0.78 | 0.76 |

The robots walked well (about 95% of episodes timed out normally) but stayed in rows 0–1, where ramps are only
0.08–0.13 m. In a real run the policy stalled at the foot of the 0.248 m ramp.

### 3.2 Root cause

Isaac Lab's `terrain_levels_vel` judges a robot by **straight-line distance** from where it spawned:

```
move_up   = distance > tile_size / 2                          (> 4 m)
move_down = distance < |command_xy| * episode_length * 0.5    (e.g. 0.85 m/s * 20 s * 0.5 = 8.5 m)
```

This assumes the robot walks straight, which is true with Isaac Lab's default `heading_command = True`. The hazard
task sets `heading_command = False`, because the pilot sends raw `(v_x, v_y, w_z)` commands. Each robot then gets a
random yaw rate of up to ±1.2 rad/s and **walks in a circle** of radius `v / w`. Its distance from the spawn never
exceeds `2v / w`, about 2.7 m at `v = 0.8`, `w = 0.6`. So it almost never moves up and almost always moves down,
however well it walks. The challenge slides (pages 8 and 19) describe the same failure.

### 3.3 Method: `terrain_levels_skill`

Code: [`curriculums.py:165`](src/g1_maze_lab/tasks/curriculum/mdp/curriculums.py:165), wired up in
[`hazard_starter/env_cfg.py:245`](src/g1_maze_lab/tasks/hazard_starter/env_cfg.py:245).

Instead of distance, the term judges **how well the robot walked** over the episode. It runs when environments reset,
before the episode reward sums are cleared:

```
track     = episode_sum(track_lin_vel_xy_exp) / episode_length_s / weight     # 0..1, mean tracking quality
fell      = base_contact OR physics_blowup terminated this episode
timed_out = episode reached its time limit
distance  = |root_xy - spawn_xy|

move_up   = timed_out AND NOT fell AND track > 0.75 AND distance > 2.0 m
move_down = (fell OR track < 0.50) AND NOT move_up
```

* `distance > 2.0 m` makes sure the robot actually left the flat spawn platform and met the hazard ring. On the ramp
  tile the ring starts at 1.0 m and its crest is at 2.0 m; on the trench tile the platform half-width is 1.1 m.
* `track < 0.5` demotes a robot that stood stuck at the foot of a ramp or trench without falling.
* Robots in between keep their row.

Unlike `heading_command = True` (the instructor's fix), this keeps training on the same direct
`(v_x, v_y, w_z)` commands the pilot sends, including sideways (`v_y` ±0.35 m/s) and backward (`v_x` down to −0.25 m/s)
commands.

### 3.4 Training and result

```bash
uv run isaaclab train --rl_library rsl_rl --task G1-Hazard-Pro --num_envs 1024 --checkpoint logs/rsl_rl/g1_hazard_pro/2026-10-31_17-40-07_curriculum_smoke/model_6998.pt agent.max_iterations=1500 agent.run_name=skill_curr1500
```

(`model_6998.pt` came from a 200-iteration check run started from `model_6799.pt`.)

| Iteration | 7185 | 7373 | 7560 | 7748 | 8123 | 8497 |
|---|---|---|---|---|---|---|
| Average terrain row | 2.2 | 3.9 | 5.2 | 5.9 | 6.2 | **6.4** |
| Velocity-tracking reward | 0.78 | 0.75 | 0.75 | 0.75 | 0.75 | 0.80 |
| Falls per episode | 4% | 5% | 6% | 7% | 6% | 5% |

The average levels off near 6.4 because Isaac Lab sends robots that clear the top row back to a random row. The
command curriculum also widened forward speed to 1.35 m/s, and pushes reached 0.8 m/s. The resulting `model_8497.pt`
crosses both 0.25–0.26 m ramps, the trench and the spike beds of level 7 seed 2026 with no falls.

---

## 4. Pilot (`pilot.py`)

### 4.1 Architecture

Both versions use the same four layers on top of the course's `MazeAutopilot` (grid indexing and lidar helpers):

```mermaid
flowchart LR
    S["obs: pos_xy, yaw, scan_xy/scan_z (33x33), lidar (72 rays), velocities"] --> P
    P["1 Perception<br/>walls + hazard type per cell"] --> M["2 Memory<br/>wall map, cell types, costs,<br/>safe-lane offsets, deaths"]
    M --> R["3 Planning<br/>hazard-weighted cost-to-goal"]
    R --> C["4 Control<br/>waypoint + speed -> v_x, v_y, w_z"]
```

What is new is inside each layer, plus a performance layer. In the tables below, **Original** means
`g1_maze_lab/pilot.py`.

### 4.2 Layer 1: perception

**Cell classification** (`_classify_cell_from_scan`). For each visible cell, scan points are converted to corridor
coordinates: `u` along the corridor, `v` across it. Rules, applied in order:

1. **Trench:** at least 3 points with `z < −0.15 m`.
2. **Elevated points** (`0.11 < z < 0.42`) split into four quadrants (`u` sign × `v` sign):
   * **Slalom:** each `u` half is elevated on one side only, the sides differ, and the centre band `|u| < 0.22` is
     clear. Safe lanes: `v = ∓0.48 m` per half.
   * **Ramp:** both sides elevated in the same `u` slice. Steep if height > 0.205 m, otherwise low.
   * **Single-lane spike bed:** elevated on one side only. Safe lane: `v = 0.48 m` on the other side.
3. **Rough** (at least 4 points at 2.5–9.5 cm), **ice** (flat with median `z` 0.6–1.6 cm), otherwise **plain**.

| Topic | Original | Now |
|---|---|---|
| Ramp height from a partial view | 90th-percentile height of what is visible. From the previous cell only the near slope shows, so a 0.24 m ramp read as about 0.19 m and was saved as a **low** ramp (cost 1.45 instead of 22). | If the flat top (`\|u\| < 0.2`) is not visible, `_ramp_height_from_slope` ([`pilot.py:324`](pilot.py:324)) fits a line `z = a + b·\|u\|` to the slope points (`0.30 < \|u\| < 0.95`) and extrapolates it to the top edge `\|u\| = 0.25`. |
| Downgrading | A later partial view could turn a steep ramp into a low one. | A steep ramp is never downgraded unless the top is in view. |
| Walls | The current cell's 4 sides were **overwritten every step** from the height scan, so walls behind the robot (out of scan range) were forgotten. Neighbour cells could only gain walls. | `_scan_walls_with_coverage` ([`pilot.py:346`](pilot.py:346)) also reports **coverage**: at least 8 scan rays inside the 0.3 m × 1.2 m wall box. `_update_cell_walls` **latches** a confirmed wall permanently and marks a side open only when it was fully covered. |
| Lidar | Angular cone check (`walls_from_lidar`) only near the cell centre. | Lidar hits are also converted to XY points (`pos + d·(cos θ, sin θ)`), and at least 2 points inside a wall box confirm a wall up to 2 cells away. The lidar flies at chest height above every floor hazard, so it only ever hits walls. |

Checked offline at five viewing distances: low ramps (0.14–0.18 m) still read as low, and steep ramps (0.24–0.30 m) read
as steep from the previous cell's centre onward.

### 4.3 Layer 2: memory and failure attribution

Both versions keep the wall map, cell types and costs across resets, and add **+50 cost** to the cell where a reset
happened (`50 × deaths` is part of the cell cost).

| Topic | Original | Now |
|---|---|---|
| Anti-stuck penalty | +15 was added straight into `cell_cost` and **erased on the next frame**, because re-classifying the cell recomputes `cost = base + 50·deaths`. | Kept in a separate `cell_penalty` map: `cost = base + 50·deaths + penalty`. |

### 4.4 Layer 3: planning

**Hazard-weighted cost-to-goal** (`weighted_flood_fill`, [`pilot.py:476`](pilot.py:476)). This is Bellman-Ford over the
grid. Entering a cell costs that cell's cost, confirmed walls block edges, and unknown walls are treated as open
(optimistic exploration):

```
dist[goal] = 0
repeat until no change:  dist[c] = min over open neighbours n of ( dist[n] + cost[n] )
```

`choose_direction` picks the open side with the lowest `dist[n] + cost[n] + turn_penalty`, where the penalty is 0.08
for a turn and 0.25 for turning back, so ties go straight on.

Two cost tables (`PILOT_MODE`):

| Cell | `auto` | `short` |
|---|---|---|
| plain / unknown | 1.0 / 1.05 | 1.0 / 1.05 |
| ice / rough / low ramp | 1.2 / 1.35 / 1.45 | 1.15 / 1.2 / 1.3 |
| spike lane | 1.75 | 1.5 |
| slalom / trench / steep ramp | 16 / 18 / 22 | 3.5 / 2.5 / 3.0 |

`auto` avoids the short route's hard hazards. `short` takes them; it is the right mode now that the policy can climb
the steep ramps.

| Topic | Original | Now |
|---|---|---|
| Replanning on hazards | `hazard_surprise` was true on **every step** while walking toward a slalom, trench or steep ramp the planner had already chosen, so the full Bellman-Ford ran 50 times a second. | The hazard type of the target is stored at decision time (`decided_tgt_type`). The planner reruns only when it changes ([`pilot.py:760`](pilot.py:760)). |
| Repeated computation | Recomputed on every call, including twice every 2 steps for the viewer's debug panel. | Cached until the wall map or the costs change (`_ff_cache`). |

### 4.5 Layer 4: local control

The controller turns a **waypoint** and a **speed limit** into commands. Normally it turns toward the waypoint
(`w_z = 1.6 · yaw_err`, capped at 1.1 rad/s, 0.65 on ice), scales `v_x` by `cos²(yaw_err)`, adds a small sideways
correction `v_y`, and limits acceleration (2.0 m/s², raised from 1.2; 0.6 on ice).

**Spike lane** (unchanged idea): hold `v = ±0.48 m` from the centreline, 0.65 m ahead, at 0.48 m/s. The lane is
pre-aligned from the previous cell.

**Slalom** (rewritten). Geometry: bed 1 at `u ∈ [−0.9, −0.3]` on one side, bed 2 at `u ∈ [+0.3, +0.9]` on the other,
a 0.6 m gap between them. Here `u_rel` is the position along the travel direction.

| Stage | Original | Now |
|---|---|---|
| 1. Approach in the entry lane | Until `u_rel = −0.22`, aiming at `u = −0.10`. | Until `u_rel = SLALOM_HOLD_U − 0.06`, aiming at `SLALOM_HOLD_U = −0.10`. Slows to 0.28 m/s within 0.55 m. |
| 2. Cross the gap | Aim at the exit lane with **forward motion still on** (≥ 0.25 m/s, turning toward the lane). Past `u_rel = 0.25` it jumped to stage 3 even if not yet in the lane. | **Sideways strafe** (`self.strafe`): face straight down the corridor (`w_z` holds the corridor heading), `v_y = 1.2 · lateral_error` capped at ±0.35 m/s (the trained range), `v_x` only holds the pelvis at `SLALOM_HOLD_U` (−0.15 to +0.12 m/s). Never advances until the robot is in the exit lane. |
| 3. Exit | Walk out in the exit lane. | Same, entered only once `\|v − v_exit\| ≤ 0.15 m`. |
| Cone sidestep | Generic sidestep active, and it pushed the wrong way (toward the second bed's side). | Off inside slalom cells; the stage logic handles the beds. |

**Why `SLALOM_HOLD_U = −0.10`.** A debug trace logged every robot link inside the bed rectangles. While strafing, the
policy crouches, and its **knees sit 0.21–0.32 m ahead of the pelvis at about 0.31 m height**. That is below the
0.37 m spike-hit height, so a pelvis held at the gap centre puts the knees into bed 2 (which starts at `u = +0.30`). No
link reached back toward bed 1. Holding the pelvis 0.10 m behind the centre keeps the knees (+0.22) and heels inside
the gap.

**Other control fixes**

| Topic | Original | Now |
|---|---|---|
| Arrival at a target cell | Only within 0.22 m (near hazards) or 0.52 m of its centre. After a slip past the centre, the waypoint stayed behind the robot and it turned around. | Also arrives once past the centre along the travel direction (`past_centre`). |
| Unstuck reflex (3.5 s without 0.28 m of progress) | Backs up at −0.25 m/s, strafing and turning ±, **alternating sides blindly**, so one attempt in two strafed into a spike bed. | `_side_away_from_cones` ([`pilot.py:689`](pilot.py:689)) picks the side away from cone-height points within 1.2 m; it alternates only when no cones are near. |

### 4.6 Curved turns (new)

The original picked the next cell 0.52 m before the corner cell's centre, then aimed at the next cell's centre, about
75° away. Any heading error above 0.7 rad drops the speed to 0.25 m/s, so every corner meant braking, turning and
speeding up again. The challenge slides (page 20) suggest curves for about 20% speed.

Method:

1. **Decide earlier** ([`pilot.py:750`](pilot.py:750)). In plain cells, pick the next cell once the robot is within
   `TURN_DECIDE_DIST = 1.2 m` of the centre and within 0.45 m of the centreline, but **only when the three sides ahead
   are already known**. Unknown sides are never chosen as open, so deciding before seeing them would bias the choice.
   This doesn't apply near spikes, trenches or steep ramps, or on ice.
2. **Pure pursuit on the cell path** (`_pursuit_point`, [`pilot.py:653`](pilot.py:653)). The path is the polyline
   *previous centre → corner centre → next centre*. The robot finds its closest point on it (arc length `s`) and aims
   at the point at `s + PURSUIT_LOOKAHEAD` (1.0 m; first tuned at 0.7 m). Near the corner the lookahead point moves onto the second segment,
   so the robot walks a smooth arc that cuts the corner by a few tenths of a metre, keeping well clear of the inner
   wall corner.
3. **Speed from curvature.** The curvature to the lookahead point is `κ = 2·|sin(yaw_err)| / d`; the speed is
   `v = min(v_max, √(a_lat / κ))` with `a_lat = LAT_ACC_MAX = 0.6 m/s²` (half on ice), floored at 0.25 m/s.
4. **No sideways correction in curves.** The lookahead point sits inside the corner, and `v_y` toward it would crab the
   robot toward the inner wall.
5. **Exceptions.** U-turns, the start, hazard waypoints (spike lanes, slalom, pre-alignment) and **ice corners** keep
   the old behaviour. Ice was excluded after a benchmark run fell on an ice corner (level 7 seed 2, cell (4, 5)).

### 4.7 Performance (new)

`act()` was timed with synthetic observations on the same machine:

| Version | Median per step | 90th percentile |
|---|---|---|
| Before the speed-up, tensors on CUDA (as in the original) | 27.4 ms | 30.4 ms |
| Before the speed-up, same code on CPU | 7.6 ms | 8.7 ms |
| **Final** | **1.4 ms** | 5.4 ms |

At 50 steps per second, 27 ms per step is about 14 minutes of pilot time for a 600 s final run, close to the
runner's 18-minute limit. Three changes:

1. **CPU execution** ([`pilot.py:85`](pilot.py:85)). The pilot does hundreds of tiny tensor operations and `.item()`
   reads per step, and on CUDA each read forces a GPU sync. The observations are copied to the CPU once per step, all
   pilot state lives on the CPU, and the command is copied back to the simulator's device.
2. **Sensing decimation** ([`pilot.py:715`](pilot.py:715)). The full wall and hazard scan runs every 3rd step (the
   robot moves about 3 cm in between), and also immediately on entering a new cell or when a decision is needed.
   Otherwise the stored walls are used. Spike repulsion and slalom logic still read the live scan every step.
3. **Route cache** (Section 4.4).

### 4.8 Dead ends and turning back on the spot (new)

**Problem.** A robot heading into a dead end walked all the way to its last cell before turning back, for two reasons:

1. Once a target cell was chosen, the pilot only reconsidered it if a wall appeared on the side it was leaving
   through, a hard hazard appeared in the target, or it arrived. A target found to be closed on three sides triggered
   none of these.
2. Walls were only mapped for the current cell and its 4 neighbours, so a dead-end corridor further ahead looked open
   (the planner treats unknown sides as open) even though the 6 m lidar could see its end.

Turning back was also risky. A reversal was walked as a normal turn: forward speed held at 0.25 m/s while turning at up
to 1.1 rad/s, which swings the robot about 0.7 m sideways, toward a wall.

No extra dead-end pruning is needed. Once a dead end's walls are in the map, the Bellman-Ford cost-to-goal already
prices entering it as "go in and come back out", so the planner never chooses it. What was missing was the knowledge
(fix 2) and the reaction (fix 1).

**Fix 1: re-decide when the route through the target gets worse.**
- At each decision the pilot stores the chosen target's cost-to-goal (`decided_cost`).
- Every step, while still in the decision cell, it compares that with the current value from the cached flood fill.
  If the cost rose by more than `ROUTE_REPLAN_MARGIN = 1.0` (a dead end adds at least 2 cells), it re-decides
  (`route_surprise`).
- If the robot is already inside a cell whose three sides ahead are known walls (`dead_end_here`), it decides
  immediately instead of walking on to the cell's decision point.

**Fix 2: lidar line-of-sight wall mapping** (`_map_walls_from_lidar`). Cell boundaries lie on the lines
`x = 2k + 1` and `y = 2k + 1`. For each of the 72 lidar rays that hits something within 5.5 m:
- If the hit point is within 0.12 m of a boundary line (wall face at ±0.075 m), and in the middle part of that cell
  edge (more than 0.15 m from the corner posts), it votes **wall** for that edge.
- Every edge interior the ray **crosses** before the hit (and within 4 m) votes **open**.

Votes build up over scans:
- 3 wall votes latch a wall permanently; walls always win.
- 2 open votes mark a still-unknown edge open.

Open votes are only cast on flat floor: on a ramp the chest rises and long rays can pass over the 1.0 m walls. Rays
that miss have no usable angle in `obs` (it comes from an infinite hit point), so only rays that hit something are
used.

This maps whole corridors several cells ahead. A wrongly marked opening is no worse than unknown for the planner, and a
wrongly marked wall needs 3 hits sitting exactly on a boundary.

**Fix 3: turn back on the spot** (`_start_pivot`, `_pivot_command`). This maneuver is used for:
- every turn-back: a decision that reverses the heading, or a heading error above `PIVOT_ENTER_ERR = 2.0 rad`
- any **90° turn chosen by a replan in the same cell once the robot is near or past the cell centre**
  (`u_along > −0.5 m`), where there is no room left to walk a curve. In testing, such a late turn on an ice cell swung
  the robot into a wall, and in another run into a spike bed.
- **every turn on an ice cell**: a 90° turn walked on ice (friction 0.2) made the robot slip and fall.

It goes through four phases:

| Phase | Command | Ends when |
|---|---|---|
| 1 Brake | Forward, sideways and yaw commands down to 0 at up to 2.0 m/s² (0.6 on ice); no turning while still walking. | Command and measured body speed (`lin_vel_b`) are both near zero, or after 1.5 s. |
| 2 Clear | If any lidar range is below `PIVOT_CLEARANCE = 0.5 m` (elbows reach about 0.3 m), step away at 0.15 m/s along the summed push-away vector of the close rays, with no rotation. Skipped in spike and slalom cells (the lidar can't see cones; the safe lane already keeps 0.45 m from the wall). | Clear of walls, or after 2 s. |
| 3 Rotate | `v_x = v_y = 0`, `w_z` up to 1.0 rad/s (0.6 on ice). The shortest way round, but for about 180° turns it rotates toward the side with more lidar clearance. The direction is fixed once chosen. If a wall comes closer than 0.4 m, it goes back to Clear. | Heading error below 0.3 rad. |
| 4 Go | Normal waypoint tracking resumes. | |

While the robot turns on the spot, the 3.5 s anti-stuck timer is paused (standing still is intended here). A full
turn-back takes about 3–5 s, well under the challenge's 10 s stuck check. Ordinary 90° corners (about 1.3 rad) never
use this maneuver; they stay curves (Section 4.6).

**Fix 4: stuck-check guard** (`_stuck_guard`). Testing showed a new failure mode. `challenge.py` checks the
position every 10 s and resets the robot if it moved less than 0.25 m since the previous check, even if it walked
away and came back in between. In a level 9 run the robot walked 1.7 m forward, turned back on the spot (3.5 s) and was
0.11 m from its previous check position at t = 20 s, so it was reset to the start.

The check times are fixed (every 10 s of `obs["t"]`), so the pilot handles this itself:
- It records its position at each check.
- In the last 1.5 s before the next check, it predicts its position at check time from the current command.
- If that would be within 0.35 m of the recorded position, it either stops (when it is already far enough away), or
  moves at 0.35 m/s **along the corridor axis** away from that spot. Moving along the axis keeps any safe lane beside a
  spike bed.

The guard is skipped during the slalom strafe (moving along the corridor there would reach a bed) and during the
unstuck reflex.

**Fix 5: no decisions inside the goal cell.** GOAL needs the robot within 0.6 m of the goal cell's centre, but the early
turn decision (Section 4.6) also fired in the goal cell, so it picked a "next" cell, back out. The old wide U-turn
happened to swing through the 0.6 m circle; with the turn on the spot the robot ping-ponged between the goal cell and
its neighbour until the stuck check reset it (level 7 seed 3). Now the robot simply walks to the goal centre.

**Fix 6: line up with a spike lane before reaching the bed.** The spike-lane pre-alignment aimed at the safe lane at
`u = −0.75`, only 5 cm before the bed starts.

Coming from straight along the corridor that is fine. But after a late turn on an ice corner (level 7 seed 2), the
robot slid and drifted 0.9 m off the centreline, directly above the bed. Its diagonal approach then clipped the bed's
near corner.

Now the robot first gets onto the lane at `u = −1.4` (0.7 m before the bed), and only then walks in. This applies
when it is still more than 1.0 m before the cell centre and **in the bed's half of the corridor** (more than 0.15 m past
the centreline on the bed's side).

Coming along the centreline (0.48 m off the lane) is the normal case and keeps the usual diagonal approach. A first
version that also staged from the centreline made every spike-lane approach a slow sideways step, and was less steady.

It moves onto the lane by **stepping sideways while facing down the corridor** (the slalom's strafe mode, with
forward speed held near zero). A first version turned toward the staging point instead, overshot on ice and hit a
wall.

Being slightly off the lane on the wall side is safe, so it doesn't trigger staging. On rough floor such a small
sideways step stalled and set off the unstuck reflex.

**Fix 7: slalom exit waypoint stays ahead.** The slalom's last stage aimed at a fixed point at `u = +0.95`, still
inside the cell. The robot reached it before crossing into the next cell, the waypoint ended up on top of the robot,
and the heading error turned into noise (it spun into a wall). Now the exit point is kept 0.8 m ahead of the robot
(`u` between 0.95 and 1.8), in the exit lane.

**Results.** All runs use `model_8497.pt` and the tuned constants (lookahead 1.0 m, turn decision 1.2 m before the
centre, `max_acc` 2.0).

Checks with the debug trace:
- **Lidar height:** the lidar sits at 0.51–0.74 m on flat floor, well below the 1.0 m wall tops.
- **Map accuracy:** the pilot's wall map was compared with the real maze every 10 s in 11 traced runs, with **no false
  walls and no false openings**.
- **Clearance:** every turn-back went brake → (clear) → rotate, and stayed at least 0.72 m from any wall while rotating.

Dead ends, in `auto` mode (which explores more), level 9 seed 2:

| | Old pilot | New pilot |
|---|---|---|
| Entries into true dead-end cells | 2 | 0 |
| Result | 252.8 s, 1 fall | 152.9 s* |

\* In that run, an out-and-back trip led to a false STUCK reset. That was the reason for Fix 4.

8-maze benchmark (`PILOT_MODE=short`), two runs each, because simulator runs vary:

| Maze | Old pilot, run 1 | Old pilot, run 2 | New pilot, run 1 | New pilot, run 2 |
|---|---|---|---|---|
| L7 seed 1 | 61.7 s, 0 falls | 61.4 s, 0 | 62.9 s, 0 | 64.2 s, 0 |
| L7 seed 2 | 95.3 s, 0 | 157.1 s, 1 | 92.9 s, 0 | 90.1 s, 0 |
| L7 seed 3 | 80.7 s, 0 | 83.7 s, 0 | 80.7 s, 0 | 82.6 s, 0 |
| L7 seed 4, random ends | 55.1 s, 0 | 54.5 s, 0 | 50.1 s, 0 | 50.4 s, 0 |
| L9 seed 1 | 113.9 s, 0 | 114.3 s, 0 | 122.4 s, 0 | 121.7 s, 0 |
| L9 seed 2 | 298.6 s, 1 | 95.1 s, 0 | 245.9 s, 1 ** | 107.5 s, 0 |
| L9 seed 3 | 105.0 s, 0 | 107.1 s, 0 | 92.8 s, 0 | 92.3 s, 0 |
| L9 seed 5, random ends | 122.1 s, 0 | 124.4 s, 0 | 87.5 s, 0 | 90.9 s, 0 |
| **Total** | 932 s, 1 fall | 798 s, 1 fall | 835 s, 1 fall | **700 s, 0 falls** |

\*\* A STUCK reset on the 0.296 m steep ramp in cell (6, 5), the tallest ramp the generator makes. The policy trained
mostly up to curriculum row about 6.4, and ramps that tall only appear around row 9.9. This is a walking-policy limit;
the other run crossed the same ramp.

On average the new pilot is **11% faster** (768 s vs 865 s per set) with no more falls. The biggest gains are on mazes
where the old pilot walked into dead ends or turned in wide arcs (level 9 seeds 3 and 5).

---

## 5. Testing

### 5.1 Unit tests (no simulator)

```bash
uv run pytest tests
```

75 tests pass. The new ones are:

| Test | Checks |
|---|---|
| `test_stuck_penalty_still_changes_the_plan` | The +15 stuck penalty survives re-classification and changes the route. |
| `test_steep_ramp_seen_from_previous_cell_is_not_low` | Slope extrapolation reads a 0.24 m ramp as steep. |
| `test_steep_ramp_is_not_downgraded_by_a_later_partial_view` | No STEEP → LOW downgrade. |
| `test_slalom_does_not_advance_into_second_bed_before_crossing` | The slalom waypoint never moves into bed 2 before the lane change. |
| `test_unstuck_reflex_never_strafes_toward_spike_bed` | The unstuck reflex strafes away from the bed. |
| `test_planner_does_not_rerun_every_step_toward_a_chosen_hazard` | At most one replan for a hazard that is already known. |
| `test_robot_past_target_centre_gets_a_new_target` | Re-targets after overshooting a centre. |
| `test_corner_is_decided_early_and_walked_as_a_curve` | Early decision and a lookahead point around the corner. |
| `test_dead_end_target_turns_back_before_entering` | A target found to be a dead end is dropped while the robot is still in the previous cell, with a turn-back on the spot. |
| `test_robot_inside_dead_end_decides_immediately` | Inside a three-wall cell the robot re-targets at once. |
| `test_lidar_maps_dead_end_corridor` | Ray-cast lidar maps a corridor's far walls (4–5 m away) and its open boundaries; the planner then picks the other way. |
| `test_reverse_brakes_then_rotates_in_place` | No turning while still walking; then rotation with zero forward and sideways speed. |
| `test_pivot_steps_away_from_close_wall_first` | A wall 0.3 m ahead makes the robot back away before rotating. |
| `test_pivot_does_not_trigger_unstuck` | 4.5 s of standing still while turning on the spot doesn't fire the anti-stuck reflex. |
| `test_normal_corner_does_not_pivot` | 90° corners stay curves. |
| `test_no_new_target_inside_goal_cell` | Inside the goal cell the target stays the goal (no turning back out). |
| `test_late_replan_turn_rotates_on_the_spot` | A 90° turn from a replan past the cell centre is done on the spot. |
| `test_spike_lane_lined_up_before_the_bed_when_far_off_lane` | Far off the safe lane on the bed side, the robot lines up 0.7 m before the bed by stepping sideways, not by aiming at the bed's corner. |
| `test_no_lane_staging_when_off_lane_on_the_wall_side` | Slightly off the lane on the wall side: no sideways staging. |
| `test_no_lane_staging_from_the_centreline` | On the centreline: the usual diagonal pre-alignment, no staging. |
| `test_slalom_exit_waypoint_stays_ahead` | Near the slalom cell's exit border the waypoint stays more than 0.5 m ahead. |
| `test_turn_on_ice_is_done_on_the_spot` | A 90° turn at an ice corner is done by stopping and rotating. |
| `test_stuck_guard_moves_off_last_check_position` | Standing at the last check's position just before the next 10 s check, the robot moves away without turning; far from it, the guard stays off. |
| `test_terrain_curriculum.py` (4 tests) | Circling but walking well → up; fall → down; stuck at a ramp → down; never left the platform → stays. |

### 5.2 Simulator benchmark

```bash
PILOT_MODE=short uv run python scripts/challenge.py --level 9 --seed 3 --pilot pilot.py --policy logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt
```

Eight mazes: level 7 seeds 1–3, level 7 seed 4 with `--random-ends`, level 9 seeds 1–3, and level 9 seed 5 with
`--random-ends` (seed 4 is skipped because the maze generator can't build a level 9 random-ends maze for it). Results
are in Section 1.

### 5.3 Debugging method

Falls were investigated with a debug-only wrapper around `Pilot`, kept outside the project and never submitted. It
logged position, target, cell types, safe-lane offsets and commands every step, plus robot link positions near spike
beds, combined with `CHALLENGE_SHOW_MAZE=1` (prints the real maze) and viewer snapshots (`MAZE_SNAP_AT`). This found:

* **Wide U-turns** at speed swinging the robot toward a wall (the likely cause of early wall crashes).
* **Partial ramp views** read as spike beds.
* A **reset in the start cell** penalising the long route's first cell.
* The **slalom knee geometry** (Section 4.5).

---

## 6. Known limitations and open items

| Item | Status |
|---|---|
| `PILOT_MODE` default | Now `short` in `pilot.py`, which is right for the submission (the runner sets no environment variables). |
| Wall clearance | The pilot has no body-width model. It relies on lane offsets, curve geometry and, when turning back, the 0.5 m lidar clearance check (Section 4.8). |
| Death attribution | Only one cell is penalised. A spike hit while the pelvis is still in the cell before the bed penalises the plain cell. A reset in the start cell penalises the target, which can be the other route's entrance. |
| Partial ramp seen as spikes | A ramp seen at the scan's edge on one side only can briefly be classified as a spike bed until more of it is visible. |
| False STUCK at the 10 s check | Mostly handled by the stuck-check guard (Section 4.8). It can't help when a check falls less than about a second after a reset: the robot starts from rest at the start cell and can't get 0.25 m away in time. |
| Tallest steep ramps (about 0.29–0.30 m) | The policy sometimes stalls on them (one STUCK reset in 16 benchmark runs). More training on the top curriculum rows would help. |
| Permanent wall latch | A false wall is never removed. Lidar walls need 3 votes on the same edge, but a false one would still be permanent. |
| Final level | The final adds a slalom on the long route too, and starts and goals at least 10 cells apart. Slalom handling is the same on either route, but no final-size maze could be run locally (its task folder is released on challenge day). |

---

## 7. How to reproduce

```bash
uv run pytest tests
```

```bash
uv run isaaclab train --rl_library rsl_rl --task G1-Hazard-Pro --num_envs 1024 --checkpoint logs/rsl_rl/g1_hazard_pro/2026-10-31_17-40-07_curriculum_smoke/model_6998.pt agent.max_iterations=1500
```

```bash
PILOT_MODE=short uv run python scripts/challenge.py --level 7 --seed 2026 --pilot pilot.py --policy logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt --viz newton
```

On Windows, add `PYTHONIOENCODING=utf-8` when redirecting output to a file; otherwise Isaac Lab's loading screen
crashes on a non-ASCII character.

Submission: upload `model_8497.pt` as `policy.pt`, plus `pilot.py` (with the `short` default).
