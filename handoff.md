# Handoff: Challenge 7.5 (G1 Hazard Maze)

Use this file to pick the work up in a new session. The full write-up of methods and results is in
[develop.md](develop.md); this file covers the current state, how to work, and what's next.

## 1. Where things stand (2026-11-03)

| Part | State |
|---|---|
| Walking policy | `logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt`. Trained with the fixed terrain curriculum (`terrain_levels_skill`). Climbs ramps up to about 0.28 m and crosses trenches. **`logs/` is git-ignored, so this file exists only on this machine. Back it up**, for example by copying it to `pretrained/`. |
| Pilot | [pilot.py](pilot.py), 1,348 lines. Includes the dead-end fix, lidar wall mapping, turn-back on the spot, stuck-check guard and the fixes found in testing (develop.md §4.8), plus (2026-11-02) trench crossing at 0.9 m/s and the stall-blame rule (section 5, item 7), and (2026-11-03) steep ramps climbed at 0.9 m/s after a 1 m run-up (section 3, "Ramp test") and ice turns decided at the ice edge (section 3, "Ice turn test"). The 2026-11-03 spike-entry and rough-floor turn experiments were reverted (section 3, "Spike-lane corners"; section 4). |
| Tests | `uv run pytest tests`: **87 pass** (no simulator needed). |
| Git | Branch `side-test`. Everything through the 2026-11-03 ramp and ice-turn work is committed (`git log` for the latest commit; `tools/` and `logs/` are git-ignored, so the checkpoint is not in git). Ask before committing. |
| Submission | Upload `model_8497.pt` as `policy.pt` plus `pilot.py`. `PILOT_MODE` already defaults to `short` (the runner sets no env vars). |

### Latest benchmark (two runs each; simulator runs vary)

Policy `model_8497.pt`, `PILOT_MODE=short`. "Old" is the pilot snapshot from before this session's dead-end work.

| Maze | Old run 1 | Old run 2 | New run 1 | New run 2 |
|---|---|---|---|---|
| L7 seed 1 | 61.7 s, 0 falls | 61.4 s, 0 | 62.9 s, 0 | 64.2 s, 0 |
| L7 seed 2 | 95.3 s, 0 | 157.1 s, 1 | 92.9 s, 0 | 90.1 s, 0 |
| L7 seed 3 | 80.7 s, 0 | 83.7 s, 0 | 80.7 s, 0 | 82.6 s, 0 |
| L7 seed 4, random ends | 55.1 s, 0 | 54.5 s, 0 | 50.1 s, 0 | 50.4 s, 0 |
| L9 seed 1 | 113.9 s, 0 | 114.3 s, 0 | 122.4 s, 0 | 121.7 s, 0 |
| L9 seed 2 | 298.6 s, 1 | 95.1 s, 0 | 245.9 s, 1 (STUCK on a 0.296 m ramp) | 107.5 s, 0 |
| L9 seed 3 | 105.0 s, 0 | 107.1 s, 0 | 92.8 s, 0 | 92.3 s, 0 |
| L9 seed 5, random ends | 122.1 s, 0 | 124.4 s, 0 | 87.5 s, 0 | 90.9 s, 0 |
| **Total** | 932 s, 1 fall | 798 s, 1 fall | 835 s, 1 fall | **700 s, 0 falls** |

**Current pilot (2026-11-02, trench 0.9 m/s + stall-blame rule)**, traced runs: run 1 **685 s, 0 falls**, run 2
**685 s, 0 falls** (L7 seeds 1–4: 63.1 / 91.8 / 82.0 / 49.7 and 62.9 / 90.6 / 81.6 / 50.5 s; L9 seeds 1, 2, 3, 5:
123.6 / 93.9 / 94.2 / 86.9 and 122.1 / 96.2 / 93.2 / 87.8 s). This is the new bar. Four more L9 seed 5 runs: 86.5,
85.4, 96.8 (rough-floor snag, kept the route), 86.2 s.

These numbers are the bar for any change: same 8 mazes, at least 2 runs each, no new falls, total time not worse.
(Level 9 seed 4 with random ends can't be generated: "no 9x9 hazard maze fits", so seed 5 is used.)

## 2. Map of the code

[pilot.py](pilot.py): one `Pilot` class built on the course's `MazeAutopilot`. All tunables are module constants
(lines 37–80).

| Layer | Method | What it does |
|---|---|---|
| Perception | `sense_environment` | Every 3rd step, and immediately on a new cell or when a decision is due: walls from the height scan, with lidar wall boxes for the current cell and its 4 neighbours, then hazard classification. |
| | `_classify_cell_from_scan` | Works out trench, slalom, ramp (steep or low, from a slope fit), spike lane, rough, ice or plain, plus the safe-lane offsets. |
| | `_map_walls_from_lidar` | Line-of-sight wall votes along all 72 rays, up to 5.5 m. 3 votes = wall; 2 open votes on flat floor = open. |
| Memory | `cell_type`, `cell_cost`, `ap.walls`, `cell_deaths`, `cell_penalty`, `wall_votes`, `open_votes` | All kept across resets. |
| Planning | `weighted_flood_fill`, `choose_direction` | Bellman-Ford cost-to-goal (cached until the map changes); cost tables `COSTS_SHORT` and `COSTS_AUTO`. |
| | `act()` decide block | Re-decides when the pilot needs a decision, on arrival (including the early decision before corners), on a new wall, a new hazard, a route-cost rise (`route_surprise`), or when the robot is in a dead end. **Never decides inside the goal cell.** |
| Control | `_compute_waypoint_and_speed` | Waypoint and speed limit: corner curves (`_pursuit_point`), spike-lane safe lane and pre-alignment (with sideways staging), slalom stages, cone avoidance. |
| | `_pivot_command` | Turn-back: brake → step away from walls → rotate on the spot. Used for turn-backs, late 90° turns, and every turn on ice. |
| | `_stuck_guard` | Moves off the previous 10 s check position just before the next check. |
| | `act()` control loop | Priority: unstuck reflex > stuck guard > pivot > slalom strafe > normal tracking. |
| Debug | `get_debug_snapshot` | Read by the viewer HUD. |

Constants changed by the user (keep them): `PURSUIT_LOOKAHEAD = 1`, `TURN_DECIDE_DIST = 1.2`, `ap.max_acc = 2`,
`PILOT_MODE` default `short`.

Other important files:
- `scripts/challenge.py`: the official runner and rules. **Don't change the rules.**
- `src/g1_maze_lab/tasks/curriculum/mdp/curriculums.py`: the `terrain_levels_skill` curriculum term.
- `src/g1_maze_lab/tasks/hazard_starter/env_cfg.py`: the training terrain and task `G1-Hazard-Pro`.
- `tests/test_pilot_review.py`: regression tests for every pilot fix, with synthetic scan, lidar (`_lidar`
  ray-caster) and wall (`_walls_z`) helpers.

## 3. How to work

Run from `g1_hazard_lab/`. On Windows always set `PYTHONIOENCODING=utf-8`, otherwise Isaac Lab crashes when output
is redirected to a file.

```bash
uv run pytest tests
```

```bash
PILOT_MODE=short PYTHONIOENCODING=utf-8 uv run python scripts/challenge.py --level 9 --seed 3 --pilot pilot.py --policy logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt
```

Add `--viz newton` to watch, `--random-ends` for random start and goal, and `CHALLENGE_SHOW_MAZE=1` to print the real
maze (local debugging only).

**Benchmark** ([tools/bench.sh](tools/bench.sh)). Copy `pilot.py` to a snapshot first if you keep editing while it
runs, because every run re-loads the file:

```bash
cp pilot.py /tmp/snap_pilot.py && PILOT=/tmp/snap_pilot.py OUT=/tmp/bench_a SPECS="7:1 7:2 7:3 7:4:--random-ends 9:1 9:2 9:3 9:5:--random-ends" bash tools/bench.sh
```

Two or three of these can run in parallel; one maze takes about 1–3 minutes of wall clock.

**Investigating a fall** ([tools/trace_pilot.py](tools/trace_pilot.py), debug only, **never upload**: it reads the
real map). It logs every step, dead-end entries and wall-map accuracy every 10 s:

```bash
PILOT_SRC=pilot.py TRACE_OUT=/tmp/trace.csv PILOT_MODE=short CHALLENGE_SHOW_MAZE=1 PYTHONIOENCODING=utf-8 uv run python scripts/challenge.py --level 7 --seed 2 --pilot tools/trace_pilot.py --policy logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt
```

Then read the CSV rows before the fall time printed in the log (`fall/crash #n in cell (i, j)`), together with the
hazard list in the log. Runs aren't repeatable, so the same seed may fail somewhere else, or not at all.

**Where the time goes** ([tools/time_breakdown.py](tools/time_breakdown.py), debug only). Reads trace CSVs and splits
each run into resets, dead-end exploration, and the final path by what limited the speed (corner, each hazard's speed
cap, turning on the spot, strafing), plus which real route (short / long) was driven:

```bash
uv run python tools/time_breakdown.py /tmp/trace_*.csv
```

Measured on 2026-11-02 (16 traced runs, 1,448 s; the same paths at 1.0 m/s = 770 s). Lost time per run: spike beds
11.5 s, the 0.68 rough / low-ramp cap 9.8 s, ice 7.7 s, dead-end exploration 5.1 s, trench / steep-ramp cap 4.0 s,
plain corners 3.6 s. The policy reaches only about 83% of any commanded speed. **In 4 of the 8 mazes (L7 seeds 1–3,
L9 seed 1) the robot drives the whole long route**: at the fork it always takes the cell it faces, `(1, 0)`.

**Trench test** ([scripts/trench_test.py](scripts/trench_test.py), task `G1-Trench-Test`). 500 robots, each in a
walled corridor with the maze's dimensions and crash rule, walk across one trench (rows = widths 0.16–0.24 m), steered
like the pilot's normal tracking. Prints crossed / fell / stuck and stumbles per width, start heading and speed in
about 90 s, so a policy or pilot change can be judged on hundreds of crossings instead of one per maze run:

```bash
PYTHONIOENCODING=utf-8 uv run python scripts/trench_test.py --checkpoint logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt
```

Result for `model_8497.pt` (2026-11-02, 200 crossings per cell, both start headings 0° / 15° together; the heading made
no clear difference):

| Width | 0.5 m/s | 0.7 m/s (pilot today) | 0.9 m/s |
|---|---|---|---|
| 0.16 m | 100% | 100% | 100% |
| 0.18 m | 94.5% | 99.5% | 100% |
| 0.20 m | 83% | 95.5% | 99% |
| 0.22 m | 73% | 85% | 89.5% |
| 0.24 m | 74% | 56.5% | 70.5% |

Slower is worse (more robots stuck straddling the trench). From 0.22 m about 40% of robots stumble at any speed: that
is the policy's limit (it trained on up to 0.24 m but settled around 0.19 m), and the maze uses up to 0.22 m.

So `pilot.py` now crosses trenches at `TRENCH_SPEED = 0.9` (was 0.70, shared with steep ramps, which keep 0.70) until
the pelvis is `TRENCH_CLEAR_U = 0.5 m` past the line; before the trench cell, ice / rough keep their own limits. Maze
benchmark (16 traced runs): 16 of 16 trench crossings clean, no stumble (baseline: 1 stumble in 13), about 0.25 s
faster per crossing. The run totals (1,488 s vs 1,448 s) got worse only through two unrelated incidents: an ice-turn
fall (L7 seed 4) and a rough-floor stall before a spike lane (L9 seed 5, +100 s; see section 5).

**Ramp test** (same script, `--task G1-Ramp-Test`, 2026-11-03). Corridors with the maze's ramp instead of a trench:
0.75 m slopes and a flat top for |u| ≤ 0.25. Rows are heights 0.24 / 0.26 / 0.28 / 0.29 / 0.30 m. With `TRENCH_START`
below 1.0 the robots start from a standstill on the slope.

```bash
PYTHONIOENCODING=utf-8 uv run python scripts/trench_test.py --task G1-Ramp-Test --checkpoint logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt
```

Result for `model_8497.pt`, share crossed (200 robots per cell, both start headings). Nobody fell except where
noted; every failure was a robot stuck on the slope:

| Height | 0.5 m/s | 0.7 m/s (old pilot) | 0.9 m/s | 1.1 m/s | From standstill halfway up, 0.7 / 0.9 / 1.0 / 1.1 m/s |
|---|---|---|---|---|---|
| 0.26 m | 99.5% | 100% | 100% | 100% | 100 / 100 / 100 / 100% |
| 0.28 m | 61% | 99% | 100% | 100% | 91 / 100 / 100 / 100% |
| 0.29 m | 29% | 82.5% | 100% | 100% | 79.5 / 100 / 100 / 99.5% (1 fall) |
| 0.30 m | 12.5% | 54% | 99.5% | 100% | 61.5 / 100 / 99.5 (1 fall) / 98.5% (3 falls) |

So the steep-ramp stalls (L9 seed 2's 0.296 m ramp, about 1 run in 3) came from climbing at 0.70 m/s. `pilot.py` now
climbs steep ramps at `STEEP_RAMP_SPEED = 0.9` from `RAMP_RUNUP = 1.0 m` before the slope, which starts at the cell
border. This also applies on rough floor before the ramp, but not on ice. Elsewhere before a ramp it keeps 0.70.
Restarting from a standstill on the slope at 0.9 also crosses, so the unstuck reflex's retry works as the fallback.
The policy was trained on forward speeds up to 1.35 m/s.

Maze benchmark with `STEEP_RAMP_SPEED = 0.9` (traced runs):

| Maze | Run 1 | Run 2 | Extra |
|---|---|---|---|
| L7 seed 1 | 63.6 | 63.0 | |
| L7 seed 2 | 91.5 | 90.7 | |
| L7 seed 3 | 82.3 | 83.6 | |
| L7 seed 4, random ends | 47.4 | 49.2 | |
| L9 seed 1 | 122.6 | 122.5 | |
| L9 seed 2 | 92.5 | 93.2 | 94.1, 95.2 |
| L9 seed 3 | 143.3 (ice-turn fall) | 95.9 | |
| L9 seed 5, random ends | 88.6 | 95.3 | |
| **Total** | 731.8 s, 1 fall | **693.4 s, 0 falls** | |

- **No ramp stalls:** L9 seed 2 got over the 0.296 m ramp in all 4 runs (before: stalled in 4 of 16 runs, about 120 s
  each).
- **The fall is unrelated:** L9 seed 3's fall was a turn on the spot on ice in cell (6, 0). No steep ramp comes before
  it, so the code that ran is the 685 s version's.
- **L9 seed 5 run 2's +8 s** was spent lining up for the slalom (the known rough-floor snag) and on a longer wrong
  branch.
- **Clean runs** are within about 2 s of the bar maze by maze. The expected saving is the stalls that no longer
  happen: about 30 s per set on average.

**Ice turn test** ([scripts/ice_turn_test.py](scripts/ice_turn_test.py), 2026-11-03). Corridors with one maze cell
of ice (friction 0.2, the challenge's own ice mesh through `HazardTerrainImporter`). The robots copy the pilot's turn
on the spot: walk in at 0.55 m/s, brake at 0.6 m/s² from `ICE_BRAKE_U` before the ice cell's centre, rotate, and
stand.
- `G1-Ice-Test` is a straight corridor (90° / 180° turns).
- `G1-Ice-Corner-Test` is an L-turn at the ice cell. Its rows are what follows the corner: plain floor, rough boxes, a
  0.28 m ramp, or a spike bed.

```bash
PYTHONIOENCODING=utf-8 uv run python scripts/ice_turn_test.py --task G1-Ice-Corner-Test --checkpoint logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt
```

Results for `model_8497.pt` (falls; 1,000 turns per rate, all four rows together):

| Rotation rate | Straight, braking from the cell border | Corner, braking from the cell border | Corner, braking 0.5 m before the centre (as the pilot did) |
|---|---|---|---|
| 0.30 rad/s | 11–17% (spins) | – | – |
| 0.45 rad/s | 0.5% | 0% | 4.0–5.6% |
| 0.60 rad/s (pilot) | 0.2% | 0% | 6.8–10.4% |
| 0.80 rad/s | 0% | 0.2% | 8.4–12.0% |

- **Where the robot rotates is what matters, not how fast.** In the maze, ice turns that rotated near the middle of
  the ice fell in 9 of 169 cases, and the 24 that rotated near the ice edge never fell. The pilot only decided an ice
  corner within 0.52 m of the cell centre (`turn_radius`): ice cells were excluded from the early turn decision.
- **What follows the corner makes no difference** in the test (plain floor falls as often as the ramp).
- **Slower rotation is not safer:** at 0.30 rad/s the robot spins.
- **Fix:** `pilot.py` now also decides early in ice cells, right after entering. A turn on the spot then brakes from
  the ice edge and rotates 0.1–0.4 m inside it (braking slides farther in the maze than in the test).
- **Exception:** not in an ice cell next to a spike lane or slalom. Stopping at the ice edge leaves the robot on the
  inside of the turn, up to 1 m from an outside lane. Lining up from there hit the slalom in L9 seed 3 cell (6, 0) in
  both runs, so these cells keep the old turn point.

Maze benchmark with this change (and `STEEP_RAMP_SPEED = 0.9`). L7 seeds 1 and 4 and L9 seed 1 come from the run
before the spike exception; they have no ice cell next to spikes, so the code they ran is the same.

| Maze | Run 1 | Run 2 | Extra |
|---|---|---|---|
| L7 seed 1 | 62.5 | 63.3 | |
| L7 seed 2 | 96.9 (rough-floor fall) | 93.3 | |
| L7 seed 3 | 81.0 | 81.2 | |
| L7 seed 4, random ends | 47.5 | 49.1 | 46.4, 46.3 |
| L9 seed 1 | 119.1 | 119.2 | |
| L9 seed 2 | 94.7 | 93.7 | |
| L9 seed 3 | 91.4 | 90.8 | |
| L9 seed 5, random ends | 88.5 | 88.0 | |
| **Total** | **681.6 s, 1 fall** | **678.6 s, 0 falls** | |

- **Both runs beat the 685 s bar.** Ice corners are also faster, since the robot no longer walks to the middle of the
  ice before turning: L9 seed 1 about 3.5 s, L9 seed 3 about 2.5 s, L7 seed 4 about 2 s.
- **No ice falls:** 30 ice turns over these runs and the run before the exception, 0 falls (the old rate was 5%).
- **The fall is unrelated:** L7 seed 2's fall came after the robot caught a foot on a loose box in rough corner
  (3, 5). That happens in every L7 seed 2 run, including the 685 s ones (3.5–5 s spent in that cell). It ended in a
  fall in 2 of 3 recent runs, and nothing new runs before that point.

**Spike-lane corners (2026-11-03).** Forcing the short route (a debug wrapper around `pilot.py` that reads the real
routes, kept out of `pilot.py`) showed deterministic spike hits on L7 seed 3 and L9 seed 1, plus a wall crash on L9
seed 2. All three were the same case: a corner onto a spike lane or slalom whose free lane is on the **outside** of
the turn. The corner curve leaves the robot on the bed's side, so the old pre-alignment stopped it and stepped ~1 m
sideways (about 0.15 m/s; on rough floor it stumbled into the wall). Then a sharp turn toward the entry point beside
the bed's corner let a swinging foot hit the bed (pelvis 0.25–0.29 m from the bed).

Two rejected attempts kept the sideways step but extended it (v1 / v2: until 0.28 m on the lane side and facing down
the corridor). The forced runs got safe, but the normal benchmark got worse: 966 s / 733 s with 3 + 1 falls, sideways
stepping 29 → 68 s per 16 runs, and two wall crashes on rough floor in L9 seed 2.

**Tried and dropped: the lane-shifted corner arc.** The idea: on a corner onto a spike cell, shift the corner arc's
path (`_pursuit_point`'s corner and target centres) sideways onto the entry lane. The robot then leaves the corner
already in the lane, with no stop and no sideways step.

It fixed the forced short routes: L7 seed 3 58.5 s and L9 seed 1 124.4 s, clean. Normal benchmark totals were 704.7 s
and 685.3 s, 0 falls (bar 685 / 685), with no time saved on the routes the benchmark drives today. It needed three
extra rules:
- **Outside-lane corners only.** With the lane on the inside the arc cut ~0.2 m past the lane toward the wall's end,
  and the robot hit it (L7 seed 4, corner (4, 4) → (4, 5); the user saw it).
- **Only while still turning.** A curve has no sideways correction, and the robot drifts to its left (section 4). After
  a right turn it ended up to 0.28 m past the lane, lidar 0.16 m (L7 seed 4, corner (2, 4) → (3, 4)). A 0.40 m/s speed
  cap didn't help.
- **Only within 0.45 m of the shifted path.** After a late turn on the spot 1.2 m off the path, the 1 m lookahead
  reached the lane only at the bed's corner (spike hit, L9 seed 2).

The user didn't like this way of entering spike cells, so `pilot.py` is back to the 685 s version's entry. That is the
diagonal pre-alignment to the lane's entry point, with sideways staging when on the bed's side. A copy of the
lane-arc version was kept only in that session's scratchpad. **The outside-lane corner problem on the short route is
open again** (spike hits on forced L7 seed 3 / L9 seed 1).

Which kind each benchmark corner is (from the traces):
- **Lane outside:** L7 seed 3 (1, 1) → (1, 2) and (2, 6) → (3, 6); L7 seed 4 (2, 4) → (3, 4); L9 seed 2 (4, 6) → (4, 5)
  and (8, 4) → (7, 4); L9 seed 1 (4, 4) → slalom (4, 5) on the forced short route.
- **Lane inside:** L7 seed 1 (3, 5) → (4, 5); L7 seed 4 (4, 0) → (4, 1) and (4, 4) → (4, 5); L9 seed 1 (5, 0) → (6, 0);
  L9 seed 2 (2, 3) → (1, 3); L9 seed 3 (8, 4) → (8, 5).

Two findings about the 685 s version's entry still apply:
- **The inside-lane corner L7 seed 4 (4, 4) → (4, 5) passes close to the wall's end:** lidar 0.19–0.33 m over 10 runs.
  The entry aims straight at the lane entry from where the turn is decided, so it passes the wall's end diagonally.
- **A reset just before a 10 s STUCK check counts twice.** The check compares with the position 10 s earlier, which may
  be the same start (L7 seed 4: an ice fall at 9.8 s, then a STUCK at 10.0 s).

## 4. Gotchas learned the hard way

- **Runs vary.** The same seed can differ by 50+ s. Never judge a change from one run.
- **Stopping a background benchmark** doesn't kill the simulator processes it started. Find and kill them, for example
  with `Get-CimInstance Win32_Process | ? { $_.CommandLine -like '*challenge.py*' }`, or the loop keeps going.
- **Lidar rays that miss** have a garbage angle in `obs` (it's computed from an infinite hit point). Use only rays with
  `0.15 < d < 5.5`.
- **The lidar sits at about 0.57 m** (0.51–0.74) on flat floor, below the 1.0 m walls. On ramps the chest rises, so open
  votes are only cast on flat floor.
- **Crash rules:** elbows, palms, torso or pelvis touching *anything* counts as a crash, and spike beds kill any body
  link lower than 0.37 m. While strafing, the knees sit 0.21–0.32 m *ahead* of the pelvis, which is why
  `SLALOM_HOLD_U = −0.10`.
- **`challenge.py`'s STUCK check** compares positions 10 s apart, even if the robot moved away and came back.
- **Upload check** rejects any file mentioning `GENERATED`, `HAZARDS`, `HARD_CELLS` or `g1_maze_lab.terrains`. Never
  copy debug code into `pilot.py`.
- **The training terrain has no walls**, and its fall check is the torso only. The policy knows nothing about walls.
- **The wall latch is permanent.** No false walls appeared in 11 traced runs, but any new wall-sensing code must keep
  that true. Check the `# MAP` lines from the trace wrapper.
- **The policy drifts to its own left** (about 8 cm per metre walking straight, more just after a turn). Normal
  tracking cancels it with `v_y`, but curve mode has no `v_y`, so a curve that ends near a wall (a lane 0.45 m from it)
  drifts into the wall after a right turn. Hand over from a curve to tracking as soon as the turn is done.
- **Don't speed up the turn on the spot on ice.** Rotating at 0.9 rad/s and braking at 1.0 m/s² (instead of 0.6 / 0.6,
  `PIVOT_TURN_RATE_ICE` / `PIVOT_BRAKE_ACC_ICE`) saved 0.85 s per ice turn but gave 4 falls on ice in 16 runs
  (0 in 20 ice turns at 0.6). Reverted. Those turns rotated on the middle of the ice (see section 3, "Ice turn
  test"); near the ice edge 0.8 rad/s didn't fall in the test, but that hasn't been tried in the maze.
- **Don't limit the turn rate on rough floor** (tested 2026-11-03). The test capped the yaw rate at 0.7 rad/s on
  rough cells (normally 1.1 walking, 1.0 rotating on the spot) for tracking, sideways stepping and rotating on the
  spot. Curves were slowed to keep their shape.
  - **Benchmark:** 699.3 s and 709.0 s with 1 fall, against 685 / 685 s with 0. Clean runs were about 12 s slower per
    set, mostly lining up for spike lanes after rough corners.
  - **The fall came from the limit:** in L7 seed 2 the slower turn at rough corner (3, 5) took a wider line. The robot
    entered the ice cell (4, 5) 0.3 m off the centreline and still turning, couldn't correct on ice, and fell turning
    on the spot after a late replan.
  - **Forced short route:** L7 seed 3 was clean twice (61.0 / 58.4 s; 0 of 2 without the limit), but L9 seed 1 still
    hit the slalom's spikes in both runs.
  - **No evidence it was needed:** in the 22 traced 685 s runs no fall came from turning on rough floor, although the
    robot turned faster than 0.7 rad/s on rough cells for over 100 s. Almost all falls while turning were turns on the
    spot on ice.

  Reverted.

## 5. Open problems and next steps (in priority order)

1. **Snags on rough floor** (a foot caught on a loose box): L7 seed 2 corner (3, 5) snags in every run, about 2 s,
   and sometimes falls. L9 seed 5 snags before its slalom. See item 7.
   *Fixed 2026-11-03:*
   - **Steep-ramp stalls** (0.29–0.30 m; L9 seed 2 stalled in 4 of 16 runs, about 120 s each) came from climbing at
     0.70 m/s. Steep ramps are now climbed at 0.9 m/s (section 3, "Ramp test").
   - **Ice turn-on-the-spot falls** (5% of ice turns) came from rotating on the middle of the ice. The turn is now
     decided on entering the ice cell, so the robot rotates near the edge (section 3, "Ice turn test"). Ice corners
     next to spike cells still turn at the old place; they had 2 falls in 32 turns before.
2. **Route choice at the fork.** In 4 of the 8 benchmark mazes (L7 seeds 1–3, L9 seed 1) the robot drives the long
   route, because `choose_direction` breaks the tie toward the cell it faces (unknown cells are priced as nearly plain).
   Clean short-route times with the route forced: L7 seed 1 ~54 s, seed 2 ~45 s, seed 3 ~58–62 s, L9 seed 1 ~130 s,
   against 63 / 91 / 82 / 122 s on the long route. Counting a ~10–18 s detour to look into the fork first, the gain is
   about 17 s per 8-maze set. First the short route needs a safe entry at corners whose spike lane is on the outside
   of the turn: today those give spike hits (section 3, "Spike-lane corners"). The pilot may only use what it senses
   (the generator's public rules: the short route is ≥ 6 cells shorter and its first hard hazard is within 3 cells),
   never the real map.
3. **Speed tuning** (list in develop.md / the earlier discussion). Raise one at a time and re-run the benchmark:
   - cruise speed `PILOT_SPEED` 1.0 → 1.2
   - speed limits near hazards: ramp and trench 0.70, rough and low ramp 0.68, ice 0.55, spike lane 0.48
   - `LAT_ACC_MAX` 0.6 → 0.8

   Also consider braking ahead of time, `v = sqrt(v_hazard² + 2·a·d)`: today the pilot slows as soon as the
   *target* cell is a hazard.
4. **Death attribution** (`reset()`): only one cell is penalised. A death while the pelvis is still in the cell before
   a hazard penalises the plain cell, and a reset in the start cell penalises the target, which may be the other route's
   entrance. Penalise both the robot's cell and the target.
5. **Wall clearance in normal driving:** there is no body-width model. The turn-back has a 0.5 m lidar clearance check,
   but normal tracking and the unstuck reflex don't.
6. **Slalom crossing speed:** the policy strafes at only about 0.18 m/s when commanded 0.30–0.35. Training with
   `lin_vel_y` up to ±0.5 would make crossings faster.
7. **One unexplained fall** in a level 7 seed 2 repeat: its log was overwritten. If L7 seed 2 falls again, trace it.
   **Snags on rough floor** (stall ≥ 1 s with the chest raised: a foot on a loose box): 9 in 68 traced runs, all in or
   next to rough cells. L9 seed 5 random ends snags at the same spot (about (9.65, 4.4), lining up for the slalom at
   0.5 m/s) in most runs. *Fixed 2026-11-02:* the unstuck reflex used to add +15 to the target cell on every stall;
   there it penalised the slalom ahead and the planner went back to the long route (+100 s). Now
   `_stall_blames_target` blames the target only if the robot is in it, a ramp / trench edge is within
   `STUCK_BLAME_DIST` (0.5 m), or it is the second stall in the same cell. Re-run: the snag recurred in 1 of 6 runs and
   cost 10 s (96.8 s instead of 188.6 s). The snag itself is still open (a gentler sidestep, or keeping 0.68 m/s
   longer before lining up, could avoid it). **Ice turns at 0.6 rad/s can still fall** (L7 seed 4, cell (2, 2), while
   rotating): 1 fall in about 30 ice turns over 32 runs, then 2 falls in 4 L7 seed 4 runs on 2026-11-03 (same cell,
   t ≈ 10 s, about 25 s lost each).
8. **The final level** (10 × 10, slalom on both routes, random ends) can't be run locally until its task folder is
   released. Practise with `--level 9 --random-ends` on several seeds.

## 6. Before submitting

- [ ] `uv run pytest tests` passes. `test_pilot_passes_server_upload_checks` checks the map-reading ban.
- [ ] 8-maze benchmark, at least 2 runs, no new falls compared with the table above.
- [ ] `PILOT_MODE` default is `short` in `pilot.py`.
- [ ] Upload `model_8497.pt` (as `policy.pt`) and `pilot.py` only. **Not** `tools/`, which reads the real map.
