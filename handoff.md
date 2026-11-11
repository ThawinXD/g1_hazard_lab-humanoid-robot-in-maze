# Handoff: Challenge 7.5 (G1 Hazard Maze)

Use this file to pick the work up in a new session. The full write-up of methods and results is in
[develop.md](develop.md); this file covers the current state, how to work, and what's next.

## 1. Where things stand (2026-11-08)

| Part | State |
|---|---|
| Walking policy | `logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt`. Trained with the fixed terrain curriculum (`terrain_levels_skill`). Climbs ramps up to about 0.28 m and crosses trenches. `logs/` is git-ignored, so a copy with its `params/` is committed as **`pretrained/hazard_pro/model_8497.pt`** (2026-11-07, byte-identical). |
| Pilot | [pilot.py](pilot.py), 1,539 lines. Includes the dead-end fix, lidar wall mapping, turn-back on the spot, stuck-check guard and the fixes found in testing (develop.md §4.8), plus (2026-11-02) trench crossing at 0.9 m/s and the stall-blame rule (section 5, item 7), and (2026-11-03) steep ramps climbed at 0.9 m/s after a 1 m run-up (section 3, "Ramp test") and ice turns decided at the ice edge (section 3, "Ice turn test"). The 2026-11-03 spike-entry and rough-floor turn experiments were reverted (section 3, "Spike-lane corners"; section 4). |
| Training setup | `G1-Hazard-Challenge` (real ice, pilot-like commands, sideways reward, harder hazards) plus the sideways and rough-floor tests, built 2026-11-03. **Tried 2026-11-04/05 and dropped: `model_8497` stays the policy.** Two runs (from `model_5600`, 3,000 iterations; from `model_8497`, 1,500) gave checkpoints much better on trenches and ramps, but each had a new weakness: long ice turns (late checkpoints), weak sideways steps (`model_7500`: 1152.5 s, 6 resets on the benchmark) or forward drift at slow / backward commands (`model_9500`: 1012.4 / 901.8 s, 5 / 3 resets, every level 9 slalom hit; otherwise 4–5% faster). See section 3, "Policy training setup". Challenge day: Saturday 2026-11-14. |
| Pilot changes (2026-11-04) | In the working tree, **not committed**: rough-floor speed 0.9 (`ROUGH_SPEED`), turn-backs moved off rough floor (`PIVOT_ROUGH_EXIT`), rough corners into ice at rough speed (`ICE_SLOW_DIST`), the old 0.68 before trenches, and the stuck guard using measured velocity. Benchmark **638.7 s and 641.0 s, 0 falls** (bar 685 / 685); residual risk at L9 seed 2's trench (section 3, "Rough corners into ice, trench approach, stuck guard"). Plus (2026-11-05) the re-decide when a late turn ends in the next cell (`lost`, L9 seed 50): seed 50 crash gone in 3 of 3 runs, benchmark 645.8 s / 0 falls and 728.4 s / 1 trench fall not caused by it (section 3, "Late turn that ends in the next cell"). |
| Pilot changes (2026-11-07, committed `95c9dc8`) | Ice turns always at the ice edge, slalom stage 2 only inside the gap, unstuck reflex without turning in spike cells. Slalom sweep over 8 mazes' entries (0 spike hits in 7 crossings); benchmark 645.8 / 644.6 / 655.6 s, 0 falls (section 3, "Slalom sweep, ice turns at the edge, ..."). |
| Route hints (2026-11-08, committed `95c9dc8`) | At a junction, prefer a branch whose next cell shows a hazard only routes carry (or loose boxes): exploring −8% on six extra mazes, no decision changed on 7 of the 8 benchmark mazes (section 3, "Route hints at junctions"). |
| Slalom sweep 2 + ramp classifier (2026-11-08, committed `040977b`) | 36 more level 9 mazes on the forced short route: **31 slalom crossings, 0 spike hits** (38 / 38 with the first sweep). Of the 6 falls elsewhere, 2 were steep ramps taken for a slalom and 1 a turn on the spot that rotated the long way inside a spike lane; fixed (ramp evidence per half, smooth-slope test, rotation sign re-chosen). Offline audit over the 44 sweep runs: no spike cell ever called a ramp, ramps misread as spike cells 69 s → 0.1 s. Benchmark 648.8 / 647.7 s (without the smooth-slope test) and 765.9 / 649.8 s (final); its falls and resets (L9 seed 2's trench, sideways staging on rough floor, L7 seed 2's 10 s stuck check) were traced to known problems, none to these changes (section 3, "Slalom sweep 2: ramps taken for spike cells, turn direction"). |
| Stuck guard walks on through (2026-11-08, committed `040977b`) | On the way back from a dead end toward the last 10 s check's spot (L7 seed 2's two resets), `_stuck_guard` now keeps walking through the spot at the commanded speed when that is predicted to end farther than stopping / backing, only facing the travel direction, on plain / rough floor and never into a wall or hazard cell. Benchmark 647.8 / 644.6 s, 0 falls or resets (first version 651.3 / 769.0 s; the 3 falls were L9 seed 2's trench and ice turns, guard idle). 16 L7 seed 2 runs, 0 resets, but the deep snag did not recur, so the new branch is checked by tests only (section 3, "Stuck guard: walk on through"). |
| Rough-corner line-up (2026-11-09/10, in the working tree, **on**) | Lining up for a spike cell right after a rough corner (sideways step while turning) falls in ~3% of such corners. A curve around a corner point 0.20 m inside the lane line (`LANE_CURVE_INSET`), at the rough-floor speed, instead: `scripts/corner_lineup_test.py` 4.0% → 0.2% falls, 3 s faster, S-bend run-in included. With it on (run snapshots): benchmark **660.5 s (1 fall: an ice turn, unrelated) / 640.4 s, 0 falls**; 25 curves in the maze, 0 falls (9 more forced runs on 2026-11-11, 9 slalom crossings, 0 hits), incl. the two corners that failed with the first version. **Switched on** 2026-11-10 (`ROUGH_CORNER_CURVE = True`; `False` brings back the sideways step) (section 3, "Rough-corner line-up"). |
| Tests | `uv run pytest tests`: **121 pass** (no simulator needed). |
| Git | Branch `side-test`. Everything through the slalom sweep 2 fixes and the stuck guard is committed (`git log` for the latest commit; `tools/` and `logs/` are git-ignored, so the checkpoint is not in git). Ask before committing. |
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
| | `_stuck_guard` | Moves off the previous 10 s check position just before the next check: stops, steps away, or walks on through it, whichever is predicted to end farther. |
| | `act()` control loop | Priority: unstuck reflex > stuck guard > pivot > slalom strafe > normal tracking. |
| Debug | `get_debug_snapshot` | Read by the viewer HUD. |

Constants changed by the user (keep them): `PURSUIT_LOOKAHEAD = 1`, `TURN_DECIDE_DIST = 1.2`, `ap.max_acc = 2`,
`PILOT_MODE` default `short`.

Other important files:
- `scripts/challenge.py`: the official runner and rules. **Don't change the rules.**
- `src/g1_maze_lab/tasks/curriculum/mdp/curriculums.py`: the `terrain_levels_skill` curriculum term.
- `src/g1_maze_lab/tasks/hazard_starter/env_cfg.py`: the training terrain and task `G1-Hazard-Pro`.
- `src/g1_maze_lab/tasks/hazard_challenge/`: task `G1-Hazard-Challenge` (`terrain.py` ice tile and harder hazards,
  `mdp/commands.py` pilot-like commands, `mdp/rewards.py` sideways reward).
- `src/g1_maze_lab/tasks/trench_test/` + `scripts/{trench,ice_turn,rough,strafe}_test.py`: policy tests in walled
  corridors (trench, ramp, ice, rough floor) and on open floor (sideways steps, drift).
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
lane-arc version was kept only in that session's scratchpad. The outside-lane corner problem on the short route
(spike hits on forced L7 seed 3 / L9 seed 1) was **fixed on 2026-11-07 by staging further back** (`STAGE_HOLD_U`,
section "Lining up for a spike cell from the bed's side"). The forced-route wrapper is `tools/force_route.py`
(git-ignored, reads the real map: `FORCE_SRC=<pilot snapshot> FORCE_ROUTE=short PILOT_SRC=tools/force_route.py`
with `--pilot tools/trace_pilot.py`); never copy it into `pilot.py`.

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

### Policy training setup (2026-11-03; trained 2026-11-04/05, dropped: `model_8497` stays)

**The policy never trained on ice.** `G1-Hazard-Pro` randomizes foot friction 0.2–1.0 to cover ice, but Newton's
MuJoCo solver uses the *larger* friction of two touching shapes (the reason the maze's ice needs `mjc:priority`), and
the training floor is 1.0. Simulator check (`G1-Ice-Test`, `model_8497`, 100 robots per group): on the floor, feet at
0.2 behave exactly like feet at 1.0 (foot slip in contact 0.073 vs 0.073 m/s standing, 0.099 vs 0.102 m/s turning
at 1 rad/s); on the maze's ice the feet slip at 0.42–0.64 m/s.

**New task `G1-Hazard-Challenge`** ([src/g1_maze_lab/tasks/hazard_challenge/](src/g1_maze_lab/tasks/hazard_challenge/)):
`G1-Hazard-Pro` (same observations, network, PPO settings, curricula and terminations) plus:
1. **Real ice:** the box tile's every-other-cell plates are maze ice (friction 0.2 with `mjc:priority`, through
   `HazardTerrainGenerator` / `HazardTerrainImporter`), no boxes on them. Foot friction is fixed at 1.0 (the old
   randomization had no effect).
2. **Pilot-like commands** (`PilotLikeVelocityCommand`): measured over the 16 latest benchmark runs, the pilot sends
   41% slow straight walking (training had 9%), 25% slow turning, 24% fast walking (training: 37%, mostly while
   turning), 5% turning on the spot (33% of the time on ice), 3% sideways (37% of slalom time). The new term picks a
   kind per robot: slow 30%, slow turning 20%, fast 15%, rotate on the spot 10%, sideways 10% (0.25–0.5 m/s), stand
   5%, fully random 10%; holds it 2–6 s (was 10 s) and ramps the command with the pilot's acceleration limits
   (2 m/s², 2.8 rad/s²) from zero at every episode start. The fast kind runs 0.75–1.1 m/s whatever the forward-speed
   curriculum says (further once that curriculum passes 1.1); only the random kind follows the curriculum.
3. **Sideways speed reward** `track_lin_vel_y_exp` (std 0.2, weight 0.5). `track_lin_vel_xy_exp` (std 0.5) is
   unchanged because `terrain_levels_skill` promotes by it.
4. **Harder hazards:** trenches 0.10–0.26 m (was 0.06–0.24), ramps 0.12–0.32 m (was 0.08–0.30). At the curriculum's
   average row (6.4) that is about 0.21 m / 0.26 m instead of 0.18 / 0.23 m.

Smoke run of the training config with `model_8497` acting (256 robots, 45 s, no training): 240 ice tiles, 9.3% of
robot time on ice (the maze: about 10%), foot slip 0.53 m/s on ice vs 0.095 off it, command kinds at their weights,
all four curricula logging. `model_8497` fell 38 times on ice and 16 times elsewhere, with ten times less time on ice.

Runs (1.2–1.35 s per iteration here, so about 30 min per 1,500 iterations; console log in `logs/train_<run>.log`):

```bash
uv run isaaclab train --rl_library rsl_rl --task G1-Hazard-Challenge --num_envs 1024 --checkpoint pretrained/hazard_starter/model_5600.pt agent.max_iterations=3000 agent.run_name=from5600_fast11
uv run isaaclab train --rl_library rsl_rl --task G1-Hazard-Challenge --num_envs 1024 --checkpoint logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt agent.max_iterations=1500 agent.run_name=from8497
```

Watch `Curriculum/terrain_levels` in TensorBoard: ice and harder commands lower the tracking reward that promotion
needs (75%). If the average row is still below about 5 after 500 iterations, lower the ice share or the threshold.
A new checkpoint replaces `model_8497` only if it is at least as good on the trench, ramp and ice tests and the
benchmark has no new falls and is not slower. Expect to retune some pilot limits for it.

**First run from `model_5600` (2026-11-04, stopped at iteration 6350, folder `..._from5600`):** the terrain
curriculum climbed faster than in the `model_8497` run (average row 5.4 after 500 iterations against 4.8), and falls
settled at about 8% of episodes (the `model_8497` run: 5–7%). But the forward-speed curriculum never moved off
0.85 m/s: it needs the tracking reward averaged over *all* ended episodes at 78% of its maximum, and ice falls held
it at about 76%. The fast kind was capped by it, so the run never trained at the pilot's 1.0 m/s cruise. Fixed by
giving the fast kind its own 1.1 m/s ceiling (`fast_max`); restarted as `from5600_fast11`. Pushes also stay off
(they start only below a 5% fall rate); the challenge never pushes, so that was left alone.

**`from5600_fast11` finished (`model_8599`, 79 min).** End of run against the end of the `model_8497` run: average
terrain row 6.42 vs 6.43, falls 7.2% vs 5.0% of episodes, forward-speed curriculum 1.25 vs 1.35 m/s (it started
moving at iteration 7500), pushes 0.1 vs 0.8 m/s, action noise 1.31 vs 1.19 (rose all run; the `model_8497` run's
rose 1.15 to 1.19). Falls include the ice, which the `model_8497` run never had.

**Policy tests, `model_8599` vs `model_8497` (2026-11-04, default settings, same session; logs in
`logs/policy_tests/<m8497|m8599>/<test>.log`). It does not pass the gate (ice, tallest ramps):**

| Test | `model_8497` | `model_8599` |
|---|---|---|
| Trench 0.22 / 0.24 m at 0.9 m/s, crossed | 83–87% / 67–76% (fell 13–33%) | **99–100% / 98%** (fell 0–2%) |
| Trench 0.24 m at 0.7 m/s, crossed | 58–64% (fell 35–42%) | **94–97%** (fell 3–6%) |
| Ramp 0.29 / 0.30 m at 0.9 m/s, climbed | 100% / 100% | 99% / **81–84%** (stuck, no falls) |
| Ramp 0.30 m at 0.5 m/s, climbed | 10–13% | 28–29% |
| Ice, straight: 180° at 0.6 rad/s, fell | 0.6% | **19.6%** (falls while rotating) |
| Ice, straight: 180° at 0.45 / 90° at 0.3 rad/s, fell | 1.6% / 14.2% | **19.8% / 40.8%** (90°: mostly while standing after) |
| Ice corner (90°, walk out): done | 100% (2.79 s at 0.6 rad/s) | 100% (2.35 s, spins less) |
| Rough at 0.9 m/s: finished / robots snagged | 99.2–100% / 3.6–8.0% | 99.6–100% / 0.8–4.0% |
| Rough at 0.68 m/s: robots snagged | 22–30% | 9–15% |
| Sideways 0.2–0.5 m/s, reached | 48–66% | 50–62% (the sideways reward did not help) |
| Straight 0.5 / 1.0 m/s, reached | 89% / 88% | 101% / 89%, less heading drift |

So training on the hazards fixed trenches and helped rough floor, but the robot now falls when it *stays* on ice
(long rotations, standing), the case the pilot uses for turn-backs (about a third of its turning-on-the-spot time is on
ice), and stalls on the tallest steep ramps at the pilot's 0.9 m/s.

**Earlier checkpoints of the same run (2026-11-05, ice / ramp / trench tests; logs in `logs/policy_tests/m<iter>/`).
`model_7500` passes all three and beats `model_8497` on each:**

| Test | `model_8497` | 7000 | **7500** | 8000 | 8599 |
|---|---|---|---|---|---|
| Ice 180° at 0.6 / 0.45 rad/s, fell | 0.6% / 1.6% | 0% / 0.4% | **0% / 0%** | 2.2% / 11.2% | 19.6% / 19.8% |
| Ice 180° at 0.3 rad/s, fell (pilot doesn't turn this slowly on ice) | 11.4% | 21.6% | 16.6% | 49.8% | 85.2% |
| Ramp 0.30 m at 0.9 / 0.7 / 0.5 m/s, climbed | 100 / 51–56 / 10–13% | 100 / 99 / 56–57% | **100 / 100 / 100%** | 100 / 100 / 100% | 81–84 / 36–44 / 28–29% |
| Trench 0.24 m at 0.9 m/s, crossed | 67–76% | 90–94% | **98–100%** | 100% | 98% |
| Trench 0.24 m at 0.5 m/s, fell | 17–26% | 16–19% | **1–3%** | 3–7% | 4% |

The ice skill went bad between 7500 and 8000, when the forward-speed curriculum started widening (0.9 → 1.05 m/s by
7750) and pushes came in; the ramp skill went after 8000. Ice corner, rough floor and sideways tests on `model_7500`
and the maze benchmark are still to do. `logs/` is git-ignored: copy `model_7500.pt` to `pretrained/` if it is kept.

**`model_7500`, rest of the gate (2026-11-05).** Ice corner 100% (2.19 s at 0.6 rad/s, fastest yet), rough floor
better than `model_8497` (0.9 m/s: 1.2–3.6% of robots snagged; 0.68 m/s: 9.6–15.6%), but **sideways steps reach only
27–32% of the command** (`model_8497`: 48–66%) and it keeps creeping forward. **Maze benchmark (baseline pilot
snapshot `trench2`, i.e. without the seed-50 `lost` fix): L7 s1 59.9, s2 90.6, s3 80.2, s4 46.1, L9 s1 115.9, then
L9 s2 380.2 s with 3 resets**, all from the weak sideways steps:
1. t = 71.9 s, slalom (6, 7): strafe commanded v_y −0.35, v_x 0.06–0.12; measured v_y −0.01 to −0.14 and v_x +0.25,
   so it walked forward into the spike bed (`model_8497` at the same spot: v_x 0.12–0.21 and slowing, stopped short).
2. t = 180 s, (3, 7): strafe v_y +0.35 measured 0.02–0.07; strafe / unstuck loop until the 10 s stuck rule.
3. t = 236.1 s, spike lane (7, 4): staging strafe at about 40% of the command, then a spike hit.

Then L9 s3 183.3 s (spike lane (6, 1) at 29.1 s, plus the stuck rule firing 0.9 s after that reset) and L9 s5
196.3 s (spike lane (5, 3) at 27.9 s). **Set 1: 1152.5 s, 6 resets (bar 638.7 s, 0)**: five are spike / sideways-step
failures, one is the stuck-rule echo.

Sideways skill varies along the run (7500: 27–32%, 8599: 50–62%) while the ice skill goes the other way, so no
checkpoint of `from5600_fast11` has both. Any new checkpoint needs the sideways test in its gate.

**Fine-tune from `model_8497` (2026-11-05, run `logs/rsl_rl/g1_hazard_challenge/2026-11-05_13-27-46_from8497`,
1,500 iterations, 45 min).** Terrain row 6.7 after 500 iterations, forward-speed curriculum at 1.35 m/s by iteration
8800, pushes never started (falls 7–11%). Gate tests (logs in `logs/policy_tests/f<iter>/`):

| Test | `model_8497` | 7500 (from 5600) | f9000 | **f9500** | f9996 |
|---|---|---|---|---|---|
| Sideways reached, left / right | 63 / 51% | 32 / 29% | 55 / 54% | **62 / 60%** | 70 / 66% |
| Sideways forward creep | 0.065 m/s | 0.077 | 0.135 | 0.135 | 0.132 |
| Ice 90° at 0.3–0.6 rad/s, fell | 0–14% | 0% | 0% | 0–0.4% | 0–0.6% |
| Ice 180° at 0.6 / 0.45 rad/s, fell | 0.6 / 1.6% | 0 / 0% | 9.4 / 25.8% | **7.4 / 11.4%** | 70.8 / 97.4% |
| Ramp 0.30 m at 0.9 / 0.7 / 0.5, climbed | 100 / 54 / 12% | 100 / 100 / 100% | 100 / 100 / 100% | 100 / 100 / 100% | 100 / 100 / 100% |
| Trench 0.24 m at 0.9 / 0.7, crossed | 72 / 61% | 99 / 100% | 100 / 96% | 98 / 100% | 99 / 99% |
| Trench 0.24 m at 0.5, crossed / fell | 72 / 22% | 70 / 2% | 84 / 15% | 91 / 8% | 94 / 6% |

Long (180°) turns on the spot on ice get worse in every fine-tuned checkpoint of both runs except the early
from-5600 ones; 90° ice turns are fine everywhere. **In the 16 baseline benchmark runs the pilot turned on the spot on
ice 18 times, all about 90° or less; its 180° turn-backs were all on rough (4) or plain floor.** So `model_9500` (best
sideways / ice-180 balance, much better ramps and trenches) is the candidate for the maze benchmark; its doubled
forward creep while stepping sideways is the thing to watch in the slaloms.

**`model_9500` maze benchmark, set 1 (2026-11-05, baseline pilot snapshot): 1012.4 s, 5 resets.** Ice corner test
first: 100% everywhere (2.75–2.78 s at 0.6 rad/s). Clean mazes are faster than `model_8497`: L7 s1 57.1, s2 86.0, s3
75.9, s4 42.5, L9 s1 110.5 (271.0 → 261.5 s on L7, 115.9 → 110.5 s on L9 s1). But L9 s2 216.1 s (1 reset), L9 s3
238.3 s (3), L9 s5 186.0 s (1), all from **forward drift at slow and backward commands**:
- four slalom spike hits ((6, 7) in s2, (6, 1) twice in s3, (5, 3) in s5): commanded 0.28 m/s into the slalom, it
  walks 0.34–0.44 m/s, then while stepping sideways (commanded v_x 0.12 down to −0.09) it still goes forward at
  0.37 → 0.12–0.24 m/s and reaches the spike bed (`model_8497`: about 0.24 m/s in, 0.20 → 0.12 m/s while stepping);
- one wall crash in (3, 5) of s3: backing off a wall before a turn on the spot (commanded v_x −0.13 m/s), it crept
  *forward* at 0.02–0.11 m/s for 2.5 s until it touched the wall.
Set 2: **901.8 s, 3 resets** (L7 s1 56.5, s2 82.4, s3 76.1, s4 43.0, L9 s1 111.0, s2 203.5, s3 144.7, s5 184.6), the
three resets again the slalom spike hits at (6, 7), (6, 1) and (5, 3): every level 9 slalom failed in both sets. On
the mazes without a reset it is 4–5% faster than `model_8497` (L7: 258.0 vs 271.0 s; L9 s1: 111.0 vs 116.0 s).
Likely cause: the pilot-like commands rarely ask for backward walking (about 4% of the time against about 16% with
`G1-Hazard-Pro`'s uniform commands), and `track_lin_vel_xy_exp` (std 0.5) barely penalizes a 0.1 m/s error. Any new
checkpoint needs a slow / backward walking test in its gate (e.g. `STRAFE_WALK="0.28,-0.13"` with the strafe test).

**Sideways test** ([scripts/strafe_test.py](scripts/strafe_test.py), task `G1-Strafe-Test`: open flat floor, 500
robots, commands ramped in like the pilot's). `model_8497` (125 robots per row, 0 falls):

| Command | Reached | Drift |
|---|---|---|
| left 0.20 / 0.35 / 0.50 m/s | 0.13 / 0.22 / 0.30 m/s (61–66%) | creeps forward 0.08–0.10 m/s, turns left 4–5°/s |
| right 0.20 / 0.35 / 0.50 m/s | 0.10 / 0.18 / 0.27 m/s (48–54%) | forward 0.03–0.04 m/s, turns left 3°/s |
| straight 0.5 / 1.0 m/s | 0.45 / 0.88 m/s (87–89%) | 8.9 / 12.8 cm left per metre, heading +1.5 / +2.3°/m |

**Rough-floor test** ([scripts/rough_test.py](scripts/rough_test.py), task `G1-Rough-Test`: walled corridors of maze
rough cells, the same boxes every run; straight, lineup (the lane moves 0.48 m sideways 2 m in), and L corners
turning left / right; steered like the pilot's normal tracking). Snag = under 0.1 m progress along the lane in 1 s.
`model_8497`, 250 robots per row and speed:

| Speed | Finished | Fell | Robots that snagged | Time stalled per robot |
|---|---|---|---|---|
| 0.50 m/s | 84–97% | 0.4–4.0% | 46–56% | 1.0–2.0 s |
| 0.68 m/s (pilot's rough limit) | 98–100% | 0–1.6% | 20–28% | 0.3–0.4 s |
| 0.90 m/s | 100% | 0% | 4–8% | 0.02–0.07 s |

Like trenches and ramps, **slower is worse on rough floor**: at 0.9 m/s the robots snag a quarter as often as at the
pilot's 0.68 and none fell (corners included, turning at up to 1.1 rad/s like the pilot).

**Rough floor at 0.9 m/s in the maze (2026-11-04, in the working tree, not committed).** `ROUGH_SPEED = 0.9` in a
rough cell or into one; low ramps keep 0.68, and ice (before or after) keeps 0.55, because the ice branch comes first.
Traced benchmark:

| Maze | Run 1 | Run 2 | Extra runs |
|---|---|---|---|
| L7 seed 1 | 59.4 | 59.8 | |
| L7 seed 2 | 86.4 | 98.1 (spike hit) | 89.2 (fall), 87.2, 83.7 (fall), 85.9 |
| L7 seed 3 | 78.7 | 78.3 | |
| L7 seed 4, random ends | 46.2 | 45.7 | |
| L9 seed 1 | 116.0 | 115.8 | |
| L9 seed 2 | 89.2 | 88.0 | |
| L9 seed 3 | 84.4 | 83.8 | |
| L9 seed 5, random ends | 85.2 | 291.3 (fall) | 79.9, 82.5, 81.5, 83.2 |
| **Total** | **645.5 s, 0 falls** | 860.8 s, 2 falls | |

- **Clean runs are faster everywhere:** about 33 s per set (645.5 vs 678.6 s), 3–7 s per maze. Mean speed on rough
  cells 0.59 m/s (was 0.50). Stalls of 1 s or more: 1 in 24 runs, 2.4 s (was 4 in 16 runs, 8.0 s).
- **Every fall was at a rough corner turning into ice, or just after one.** Three were stumbles in the rough cell
  while curving into the ice cell, at the ice approach limit of 0.55 m/s: L7 seed 2 (3, 5) twice (also the baseline's
  one fall, and 2 of 3 runs before that), and L9 seed 5 (6, 4) once (not seen in 8 earlier runs; the robot came into
  that cell at 0.86 m/s and braked to 0.55 in the corner). The fourth, the L7 seed 2 spike hit, followed the same
  corner: the robot headed for (5, 5), turned back late at the east edge of ice cell (4, 5), stepped sideways beside
  the bed and hit it turning toward the lane. Baseline runs do the same manoeuvre there (unchanged code).
- **What the 0.9 limit leaves out:** a rough cell whose next cell is ice is still driven at 0.55 m/s, the speed at
  which the rough test snags about half the robots. Next step: keep `ROUGH_SPEED` through such a rough cell and slow
  to 0.55 only just before the ice edge.
- **One fall can cost 200+ s:** after L9 seed 5's fall the planner abandoned the short route, wandered nearby
  cells, then drove the whole long route (section 5, item 4).

Not yet at the bar's "no new falls": keep it only together with the rough-before-ice fix, after a new benchmark.

**Turn-backs on rough floor (2026-11-04, in the working tree, not committed).** The user saw level 7 seed 2 crash
into a wall after turning back from the dead end at (5, 0). The dead end is found from the rough cell (4, 0), 0.5 m
before its border, and the robot turned on the spot right there, on the loose boxes. Over all traces, turns on the
spot moved the robot 0.26 m on average on rough floor and over 0.3 m in 32% of 133 turns (chest bobbing, feet on
boxes), against 0.11 m and none of 84 on plain floor. At (4, 0) the robot moved 0.3–1.0 m in every run, came within
0.23–0.75 m of a wall, and crashed once.
- **Fix (`PIVOT_ROUGH_EXIT = 0.35`, `_plan_rough_exit`):** a turn-back decided in a rough cell, past its centre and
  facing the way it walked, keeps walking into the cell ahead when that is plain floor behind a known opening. It
  brakes from 0.35 m past the border and rotates there. Otherwise it brakes at once, as before.
- **Result (20 traced runs):** all 6 level 7 seed 2 runs rotated on (5, 0) at x = 9.7–9.8, moving about 0.1 m, with
  the nearest wall 0.82–0.84 m away. Rough-floor turns on the spot went from 23 in 24 runs to 4 in 20 (left: L9
  seed 5 (5, 5) and L7 seed 4 (4, 4), where the cell ahead isn't plain). Cost: L7 seed 2 is about 3 s slower (walks
  ~0.8 m further and back); a smaller `PIVOT_ROUGH_EXIT` could cut that.
- **Benchmark with both changes:** 698.9 s and 708.7 s, 1 fall each. Neither fall touches the turn-back code: a turn
  on the spot on ice at L9 seed 3 (6, 0) (an ice cell next to the slalom, which keeps the old turn point), and a
  stumble on the 0.20 m trench line at L9 seed 2 (4, 0), crossed at 0.9 m/s coming from rough floor (that approach is
  0.9 instead of 0.68 because of `ROUGH_SPEED`). One extra L7 seed 2 run fell on ice (4, 5) right after the rough
  corner (3, 5).

**Rough corners into ice, trench approach, stuck guard (2026-11-04, in the working tree, not committed).** Three more
changes on top of the two above, each benchmarked:
1. **Rough cell before ice (`ICE_SLOW_DIST = 0.4`):** keep `ROUGH_SPEED` (the corner's curve limit still applies) and
   drop to the 0.55 ice limit only within 0.4 m of the ice edge, measured along the new heading. The robot is down to
   0.6 m/s within 0.25–0.46 m of the command dropping. The rough-corner stumbles had all been 0.56–0.86 m before the
   edge, at 0.55. Since then: no stumble in a rough corner before ice in 40 runs; L7 seed 2 (3, 5) fell once in 6
   runs (a snag at the same spot at 0.9; was 7 falls in 20), L9 seed 5 (6, 4) 0 in 6 (was 1 in 8).
2. **Trench after a rough cell: the old 0.68.** L9 seed 2's trench (4, 0) comes right after the rough cell (3, 0).
   It never fell in 32 crossings with the old pilot. With `ROUGH_SPEED` it fell in 3 of 14 (stumbles on the trench
   line, same speed, heading and lane there). Now a rough cell whose target is a trench, or the way into a rough cell
   whose next cell straight on is a trench, keeps 0.68 (`rough_before_trench`). **Limit:** the trench is classified
   only about 1.6 m ahead, so the first half of (3, 0) still runs at 0.9 (0.9 → 0.68 → 0.9); the latest 8 crossings
   still had 1 fall. Running rough cells at 0.68 until the next cell is known would restore the old approach fully.
3. **Stuck guard sees snags:** `_stuck_guard` predicts from the measured velocity (`lin_vel_b`), not the command.
   After the (5, 0) turn-back, a robot snagged on a box near its t = 10 s position was predicted 1 m clear and got a
   STUCK reset at t = 20 s (2 in 12 L7 seed 2 runs). Since the fix: 0 in 6, the guard stepping it off the spot in 3.

**Benchmark with all five changes (traced):**

| Maze | Run 1 | Run 2 | Bar (685 s runs) |
|---|---|---|---|
| L7 seed 1 | 59.5 | 59.2 | 63.1 / 62.9 |
| L7 seed 2 | 86.5 | 86.7 | 91.8 / 90.6 |
| L7 seed 3 | 78.8 | 78.6 | 82.0 / 81.6 |
| L7 seed 4, random ends | 45.6 | 46.6 | 49.7 / 50.5 |
| L9 seed 1 | 115.9 | 116.0 | 123.6 / 122.1 |
| L9 seed 2 | 86.7 | 87.1 | 93.9 / 96.2 |
| L9 seed 3 | 85.7 | 83.6 | 94.2 / 93.2 |
| L9 seed 5, random ends | 80.0 | 83.2 | 86.9 / 87.8 |
| **Total** | **638.7 s, 0 falls** | **641.0 s, 0 falls** | 685 / 685 s, 0 falls |

Faster in every maze, about 45 s per set, and no falls in either set. Six extra L9 seed 2 runs: 86.6–90.7 s, one fall
at the (4, 0) trench (166.5 s). Stalls of 1 s or more: 2 in 22 runs, both lining up for a spike lane at 0.50 m/s.

### Late turn that ends in the next cell (L9 seed 50, 2026-11-05, in the working tree)

**Problem.** L9 seed 50 (start (0, 0), goal (8, 8)): walking east at 1 m/s toward (4, 0), a route re-plan at x = 6.64
(0.64 m past the centre of (3, 0)) chose north (3, 1). That is a late turn, so the pilot brakes and turns on the spot;
the brake from 1 m/s carried the robot 0.6 m on, into (4, 0). It rotated there and then followed the corner curve laid
out from (3, 0), which aimed diagonally past the end of the wall between (3, 1) and (4, 1): lidar 0.27 m and a crash in
one run, 0.19 m in the other (the same pattern as L7 seed 2's (5, 0) turn-back).

**Fix (`act()`, after `arrived`):** a robot in neither its decision cell nor its target (and not in the goal cell)
re-decides from the cell it is in (`lost`); a turn on the spot under way rotates to the new direction. Test:
`test_robot_outside_origin_and_target_redecides`.

**Result (traced, `model_8497`).** L9 seed 50, 3 runs: 161.1 / 160.8 / 160.4 s, 0 falls. Each time the rule fired at
12.3 s in (4, 0), sent the robot back west into (3, 0) and round the corner from there; lowest lidar in (3, 0) / (4, 0)
0.42–0.73 m (was 0.19–0.27 m). All three runs took the long route (the route-choice problem, section 5 item 2; the
short route took about 89 s in the earlier run that crashed). In the 16 benchmark runs the rule fired only at L7 seed 2's
(5, 0) turn-back, where `PIVOT_ROUGH_EXIT` walks into (5, 0) on purpose: it re-decided (4, 0), the same direction, and
the turn went on unchanged.

| Maze | Run 1 | Run 2 | Bar |
|---|---|---|---|
| L7 seed 1 | 59.9 | 59.9 | 59.5 / 59.2 |
| L7 seed 2 | 88.4 | 86.9 | 86.5 / 86.7 |
| L7 seed 3 | 78.5 | 79.8 | 78.8 / 78.6 |
| L7 seed 4, random ends | 47.6 | 47.2 | 45.6 / 46.6 |
| L9 seed 1 | 115.7 | 116.1 | 115.9 / 116.0 |
| L9 seed 2 | 93.3 (slow steep-ramp climb, 9.0 s vs 6.7) | 174.5, 1 fall | 86.7 / 87.1 |
| L9 seed 3 | 82.8 | 83.9 | 85.7 / 83.6 |
| L9 seed 5, random ends | 79.6 | 80.1 | 80.0 / 83.2 |
| **Total** | **645.8 s, 0 falls** | 728.4 s, 1 fall | 638.7 / 641.0 s, 0 falls |

The run 2 fall is the trench in (8, 1) at 37.9 s: out of the corner (8, 0) → (8, 1) the robot was still accelerating
and met the 0.20 m trench at about 0.74 m/s (the trench test: 3–8% falls at 0.7 m/s, 0% at 0.9). The `lost` rule did
not fire in that run. It is the same "trench right after a corner or rough cell" risk as (4, 0) (section 5, item 1).

### Lining up for a spike cell from the bed's side: hold back (2026-11-07, committed `14f11b2`)

**Problem (forced short route, L9 seed 1, 3 of 3 runs hit).** Off the ice heading east into the rough corner (4, 4),
the slalom (4, 5) has its entry lane on the outside of the turn (x = 8.48); bed 1 covers x 7.08–7.94 from y = 9.1.
Staging (sideways steps toward the lane, section "Spike-lane corners") held the robot at u = −1.4 before the slalom's
centre, so while stepping across it also crept forward at up to 0.12 m/s: from u = −1.96 to −1.47. Staging ends 0.15 m
short of the centreline, so it stopped 0.57 m before the bed and still in line with it (x = 7.87); the diagonal toward
the lane started with a 35° right turn while walking at 0.25–0.47 m/s, and the left foot hit the bed's corner (40.8 /
45.7 / 41.1 s; run 2 first stalled 3 s against it).

**Fix:** `STAGE_HOLD_U = 1.9` (was 1.4): the staging point sits near the previous cell's centre, so the robot steps
across without creeping on and the diagonal starts with 1.15 m of room instead of 0.72 m. Test:
`test_slalom_staging_steps_across_without_creeping_toward_the_bed`.

**Result.** Forced L9 seed 1: 116.7 / 116.1 / 115.4 s, 0 falls (staging held y at 8.02–8.20; in the gap the pelvis
stayed at u −0.16…+0.03; lowest lidar 0.31–0.34 m, at the east wall beside the lane). Forced L7 seed 3: 60.2 / 56.0 s,
0 falls (the old entry: 113.0 s / 1 fall and 96.3 s / 2 falls). Benchmark (normal routes):

| Maze | Run 1 | Run 2 | Bar |
|---|---|---|---|
| L7 seed 1 | 58.9 | 59.4 | 59.5 / 59.2 |
| L7 seed 2 | 86.8 | 86.9 | 86.5 / 86.7 |
| L7 seed 3 | 79.5 | 79.7 | 78.8 / 78.6 |
| L7 seed 4, random ends | 47.0 | 47.0 | 45.6 / 46.6 |
| L9 seed 1 | 115.5 | 115.8 | 115.9 / 116.0 |
| L9 seed 2 | 89.1 | 89.4 | 86.7 / 87.1 |
| L9 seed 3 | 84.4 | 83.9 | 85.7 / 83.6 |
| L9 seed 5, random ends | 78.1 | 86.9 | 80.0 / 83.2 |
| **Total** | **639.3 s, 0 falls** | **649.0 s, 0 falls** | 638.7 / 641.0 s, 0 falls |

**Cost:** the longer diagonal (pre-alignment runs at ≤ 0.5 m/s) adds about 1.5–2 s at each outside-lane spike corner
the normal routes use: L9 seed 2 (8, 4) → (7, 4) 7.6 → 9.6 s of pre-alignment, L7 seed 4 3.8 → 5.0 s. Run 2's L9 seed
5 (+3.7 s) was more exploring (12.4 s vs 9–11) and a late turn, not the spike entry. The rough-floor wall crashes of
the rejected longer-staging attempts did not appear.

### Slalom sweep, ice turns at the edge, slalom exit, unstuck reflex in spike cells (2026-11-07, in the working tree)

The final level has a slalom on **both** routes and random ends, so every way into a slalom matters.
`tools/slalom_scan.py` (git-ignored, reads the real map) builds the challenge mazes offline (no simulator; matches the
runs exactly) and lists how each slalom is entered: `uv run python tools/slalom_scan.py 9 1 60` (list for seeds 1–60,
with and without random ends, in `logs/slalom_scan_L9.txt`). Of 117 slaloms: 35% entered straight, 40% from a corner
with the entry lane on the outside of the turn, 25% inside; almost all from rough floor, a few from ice; every one
exits straight.

**Sweep (forced short route, one run each):** L9 seed 5 (rough, right, outside), seed 2 random ends (rough, right,
outside, after a steep ramp), seed 9 (rough, right, inside), seed 15 (rough, left, inside), seed 3 random ends (ice,
left, inside), seed 22 (ice, right, outside), seed 4 (rough, left, outside), seed 50 (rough, right, inside): **7
slalom crossings, 0 spike hits** (10–13 s each, nearest 0.32–0.36 m). Three resets elsewhere led to the fixes below.

1. **Ice turns always at the ice edge (removed `ice_wait`).** An ice corner next to any spike cell used to wait and
   turn 0.5 m from the centre, on the middle of the ice (the 7–10% case of the ice-corner test). Seed 4 (1, 5) and
   seed 2 random ends (4, 2) fell there; they had rotated 0.10–0.15 m from the centre. Now they rotate 0.54–0.65 m from
   it: 0 falls in 4 runs. The reason for the wait (lining up from the inside of the turn hit L9 seed 3's slalom (6, 1))
   is gone with `STAGE_HOLD_U`: L9 seed 3 (normal route) 87.8 / 86.5 / 89.1 / 87.2 / 87.5 s, 0 falls, at about +3 s.
2. **Slalom stage 2 only inside the gap (`u_rel < 0.30`, was 0.95).** Seed 3 random ends, snagged on rough floor at
   the slalom's exit (pelvis u ≈ +0.95, this happens in about half of its runs): the unstuck reflex pushed it back to
   u = +0.61, 0.28 m off the exit lane, and stage 2 then strafed it back toward the gap along bed 2 into the spikes.
   Past the gap, stage 3 now walks on toward the exit lane (away from that bed).
3. **Unstuck reflex does not turn in a spike lane / slalom cell.** The same snag again: backing up while turning 40°
   swung the rear into bed 2 (its momentum, −0.32 m/s, carried on after the reflex). Now it backs up straight and
   steps away from the cones: seed 3 random ends 87.6 / 85.6 / 94.4 (snag at the exit, backed up straight, went on) /
   87.1 s, 0 falls (was 2 spike hits in 3 runs with a snag).
Tests: `test_ice_corner_turn_starts_at_the_ice_edge` (slalom case flipped),
`test_slalom_past_the_gap_off_lane_walks_on_not_back_along_the_bed`, `test_unstuck_reflex_does_not_turn_in_a_spike_cell`.

**Benchmark (normal routes):**

| Maze | 1 + 2, run 1 | 1 + 2, run 2 | 1 + 2 + 3 | Bar |
|---|---|---|---|---|
| L7 seed 1 | 59.4 | 59.5 | 59.0 | 59.5 / 59.2 |
| L7 seed 2 | 86.0 | 85.5 | 85.6 | 86.5 / 86.7 |
| L7 seed 3 | 79.1 | 78.5 | 79.0 | 78.8 / 78.6 |
| L7 seed 4, random ends | 47.2 | 47.3 | 46.8 | 45.6 / 46.6 |
| L9 seed 1 | 116.1 | 116.5 | 116.0 | 115.9 / 116.0 |
| L9 seed 2 | 88.8 | 87.7 | 92.5 | 86.7 / 87.1 |
| L9 seed 3 | 89.1 | 87.2 | 87.5 | 85.7 / 83.6 |
| L9 seed 5, random ends | 80.1 | 82.4 | 89.2 (known rough snag before the slalom) | 80.0 / 83.2 |
| **Total** | **645.8 s, 0 falls** | **644.6 s, 0 falls** | **655.6 s, 0 falls** | 638.7 / 641.0 s |

No new falls; about 4–7 s per set slower than the bar, from the safer spike entries (L9 seed 2 about +2 s, L9 seed 3
about +3 s). The removed failures cost 100+ s each when they happen, and the final level has more slaloms.

**Trenches (analysis only, no change).** 115 trench crossings in the recent `model_8497` traces: 5 falls (4%), 4 of
them at L9 seed 2's two 0.20 m trenches after rough floor ((4, 0), (8, 1)), one at L9 seed 22's 0.22 m trench (4, 5)
right after a trench and a corner. Corner or straight before makes no difference (3 / 59 vs 2 / 56). Speed at the line
looks protective (≥ 0.8 m/s: 1 / 74; < 0.8: 4 / 41), but the measured speed includes the stumble itself, and running
the rough cell before (4, 0) at 0.9 m/s had given 3 falls in 14. No clean pilot lever; the policy is the limit
(trench test: 0.22 m at 0.9 m/s, 13–16% falls).

### Route hints at junctions (2026-11-08, in the working tree)

**Idea (the user's):** the generator puts trenches, steep ramps, slaloms, spike lanes and low ramps only on the
start–goal routes, fills the short route's plain cells with loose boxes, and gives cells off both routes only a few
extra ice / rough cells. So a branch that shows such a hazard is not a dead end. `tools/route_stats.py` (git-ignored,
offline) over 117 level 9 mazes: short-route cells rough 58%, hazards 32%; long-route cells plain 64%; off-route cells
plain 87%, ice 9%, rough 4%, never a hazard. At 770 junctions into a dead-end branch, the branch's first cell was plain
87% of the time; the route's next cell was a hazard in 26%, rough (vs a plain branch) in another 20%. The pilot's
floor scan reaches 1.6 m, so at a junction it usually knows the near half of each neighbouring cell (not a trench line
in the middle of a side cell).

**Rule (`choose_direction` → `_route_hint`, short mode only):** a cost credit for a direction whose next cell holds one
of those hazards (`ROUTE_HINT_BONUS = 3.0`) or loose boxes (`ROUTE_HINT_ROUGH = 1.0`), half for the cell beyond it in
a straight line. No credit for going back, for a cell with a death / stall penalty, for a cell already visited
(`self.visited`, kept across resets), or for a step that does not bring the robot closer to the goal (Manhattan). The
last two came from runs: v1 sent L9 seed 5 back over a low ramp it had crossed (8 s) and L9 seed 10 random ends into a
spike lane on the long route's stretch back to the start (12 s); a route runs both ways.
Tests: `test_junction_prefers_the_branch_that_shows_a_route_hazard`,
`test_route_hint_never_overrides_death_penalties_or_points_back`.

**Six extra mazes (one run each, exploring time from `tools/time_breakdown.py`):**

| Maze | No hints | v1 | v3 (final) |
|---|---|---|---|
| L9 seed 50 | 160.5 s / 51.6 | 142.8 / 31.8 | **143.0 / 31.4** |
| L9 seed 5 | 172.0 / 87.5 | 176.7 / 83.1 | **167.6 / 76.8** |
| L9 seed 6 random ends | 123.5 / 47.4 | 126.4 / 48.7 | 126.5 / 52.6 |
| L9 seed 8 random ends | 110.7 / 24.3 | 113.8 / 30.1 | 214.0, 2 resets / 28.5 |
| L9 seed 10 random ends | 145.7 / 42.2 | 158.0 / 50.7 | 146.9 / 42.2 |
| L9 seed 31 random ends | 157.0 / 24.9 | 126.0 / 10.6 | 142.1 / 25.2 |

v3 explored 256.7 s against 277.9 s (−8%); the five mazes without a reset took 726.1 s against 758.7 s (−4%).
Seed 8's two resets were not routing: a 0.21 m trench right after ice (crossed at 0.66 m/s) and, on the long route
after that, a stumble on rough floor that slid the robot back into a spike bed.

**Benchmark:** 835.2 s (2 falls) / 653.2 s (0 falls). The hints change **no decision** in 7 of the 8 mazes (decision
sequences identical to the runs without hints); L9 seed 5 random ends takes (5, 4) → (6, 4) instead of (5, 5),
81.9 / 81.5 s. The two falls (L9 seed 3, run 1) came on the unchanged path: a stumble in the rough corner (8, 4) at
walking pace (0–0.25 m/s while turning), and, after that reset, sideways staging on the rough cell (5, 6) that
stumbled into the wall; run 2 of that maze was clean (85.2 s). L9 seed 2 run 2 (97.7 s) had a snag and the unstuck
reflex on the same route.

### Slalom sweep 2: ramps taken for spike cells, turn direction (2026-11-08, committed `040977b`)

**Sweep.** 36 level 9 mazes not run before, forced onto the short route (`tools/force_route.py`, one run each),
chosen from `logs/slalom_scan_L9.txt` to cover every way into a slalom: straight (rough, ice, after a trench / steep
ramp / spike lane), corners from rough or ice with the entry lane on the outside or the inside of the turn, left and
right. Summary per run: scratchpad `sweep_summary.py` (time in the slalom, sideways-step time, nearest lidar, exit).
**31 slalom crossings, 0 spike hits** (9.6–13.5 s each, nearest 0.30–0.37 m); with the first sweep 38 / 38. Five runs
never reached the slalom: `force_route.py` forces only the first step and the pilot turned back after one cell in 3
(seed 1 / 13 / 31 random ends), and 2 lost the forced start in a reset. Six falls elsewhere:

| Run | Cell | Cause |
|---|---|---|
| seed 11, seed 16 | trenches (3, 7), (4, 0) | the policy's trench risk (section 3, "Slalom sweep, ...": ~4% per crossing) |
| seed 20 | ice (3, 8) after the slalom exit | **the steep ramp ahead (4, 8) taken for a spike lane, then a slalom**: lined up sideways on the ice and fell |
| seed 47 random ends | steep ramp (7, 6) | **the ramp taken for a slalom and kept as one while on it**: climbed at slalom speed (0.28–0.42), stalled, the unstuck reflex backed it off and it fell |
| seed 15 random ends | spike lane (7, 1) | **turn on the spot rotated the long way inside the lane** (below) |
| seed 44 random ends | rough (4, 1) after the slalom exit | snagged while lining up for the spike lane (4, 2); the unstuck reflex (turning, rough cell) backed into the wall |

**Why ramps looked like spike cells.** `tools/classify_replay.py` (git-ignored, reads the real map) rebuilds the maze
mesh offline, recreates the 33 × 33 scan at every traced pose and replays `_classify_cell_from_scan`; it reproduced
both runs to within 0.2 s. Seed 20: yawed 0.6 rad on the ice, the scan caught only one corner of the ramp cell (the
slope's foot, u −0.80…−0.45 on +v): elevated hits on one side → spike lane. Once straight, 4 points of the foot on −v
of the **same half** passed 0.11 m, and the slalom upgrade fired because the earlier safe offset had the other sign
(both halves get the same offset from a view of one half). Seed 47 the same from rough, and the ramp test never
overwrote a slalom. The generator rules out both: a spike bed never reaches past the corridor's middle (reach
−0.1…0.0 m) and the two slalom beds lie in opposite halves, even if the axis is wrong.

**Fixes (`_classify_cell_from_scan`).**
1. `elev_sides` (per cell, kept): sides with ≥ 2 elevated hits seen so far in each half (reset when the cell's axis
   changes). Both sides in one half → ramp, and that overrules an earlier slalom.
2. The spike-lane → slalom upgrade needs the opposite side seen in the **other** half (from `elev_sides`), not an
   opposite safe-offset sign.
3. **Smooth slope → ramp from the first view** (`_smooth_slope`, `SLOPE_FIT_RMS = 0.002`, `SLOPE_MIN_POINTS = 8`,
   `SLOPE_RANGE`): fit z = a + b |u| + c v to the in-cell hits on slopes / cones (0.02 < z < 0.42, |u| > 0.25). With
   fixes 1–2 only, seed 20 still fell in 2 of 2 runs: no second view came, the ramp stayed a "spike lane" for 2.6 s
   while the robot stepped sideways on the ice (it slid the other way), then it spun out on the run-up turning
   0.65 rad/s at 0.65 m/s. `tools/plane_stats.py` over 44 runs (views with ≥ 8 such hits): 14,140 ramp views fit to
   0.000 m (an exact plane); 25,392 spike lane / slalom views none below 0.0065 m, none below 0.0147 m with a
   ramp-like slope and tilt.

**Turn direction (`act()`, decide block).** Seed 15 random ends: turning on the spot toward (8, 0) in the rough corner
(7, 0), right after the spike lane (7, 1), the loose boxes threw the robot 0.9 m back into the lane; the re-decide
there changed the pivot's goal to south, 40° clockwise, but `pivot_dir` kept the counter-clockwise sign chosen for the
old goal and it swept 330° inside the lane into the spikes. A changed goal heading now resets `pivot_dir` (the
shortest way is picked again).

Tests: `test_ramp_foot_seen_one_side_at_a_time_is_a_ramp_not_a_slalom`, `test_ramp_slope_seen_from_one_side_is_a_ramp_at_once`,
`test_false_slalom_is_corrected_by_a_view_of_the_ramp`, `test_pivot_retargeted_by_a_redecide_turns_the_short_way`
(fail on the old pilot), `test_slalom_seen_one_half_at_a_time_still_becomes_a_slalom`,
`test_spike_cones_are_never_a_smooth_slope` (guards). `test_option_b.py` reuses one cell for every hazard type and now
clears `elev_sides` with `cell_type` between cases.

**Offline audit** (`tools/classify_audit.py`: old and new classifier on the same recreated scans along all 44 sweep
runs, the robot's cell and its open neighbours, every 2nd step):

| True type | Cells | Final type right, old → new | Time taken for a spike lane / slalom, old → new |
|---|---|---|---|
| Spike lane | 105 | 105 → 105 | identical (4.6 s plain / rough at first sight, both) |
| Slalom | 38 | 38 → 38 | identical (159 s "spike lane" before the second bed is seen, both) |
| Steep ramp | 77 | 54 → **77** | 44.0 s → **0.1 s** |
| Low ramp | 38 | 15 → **32** | 24.9 s → **0** |

No spike cell was ever called a ramp. The old classifier also relabelled many ramps as spike lanes **after** passing
them (a backward view of one side), so a revisit after a reset would have climbed them at spike-lane speed (seed
47's stall). Still open: 6 low ramps end as rough (their foot is below 0.11 m; 45 s in both versions).

**Forced reruns:**

| Maze | Before | Fixes 1–2 + turn | + smooth slope |
|---|---|---|---|
| L9 seed 20 | 199.6 s, 1 fall | 200.8 / 199.2 s, 1 fall each | **67.4 / 67.6 s, 0** |
| L9 seed 47, random ends | 236.9 s, 1 fall | 112.7 / 113.2 s, 0 | 112.2 / 115.5 s, 0 |
| L9 seed 15, random ends | 193.2 s, spike hit | 109.3 / 104.2 s, 0 (the throw-back did not recur) | — |

**Benchmark (normal routes):**

| Maze | Fixes 1–2 + turn, run 1 | run 2 | Final (+ smooth slope), run 1 | run 2 | Bar |
|---|---|---|---|---|---|
| L7 seed 1 | 59.6 | 60.0 | 59.4 | 59.4 | 59.5 / 59.2 |
| L7 seed 2 | 88.5 (stuck-check reset) | 86.1 | 85.9 | 90.8 (stuck-check reset) | 86.5 / 86.7 |
| L7 seed 3 | 78.8 | 78.5 | 79.2 | 79.4 | 78.8 / 78.6 |
| L7 seed 4, random ends | 47.3 | 48.0 | 47.5 | 47.4 | 45.6 / 46.6 |
| L9 seed 1 | 116.0 | 116.1 | 115.9 | 116.0 | 115.9 / 116.0 |
| L9 seed 2 | 90.2 | 91.0 | 214.1 (2 falls, below) | 89.6 | 86.7 / 87.1 |
| L9 seed 3 | 88.0 | 87.0 | 85.0 | 87.1 | 85.7 / 83.6 |
| L9 seed 5, random ends | 80.4 | 81.0 | 78.9 | 80.1 | 80.0 / 83.2 |
| **Total** | **648.8 s, 1 reset** | **647.7 s, 0** | **765.9 s, 2 falls** | **649.8 s, 1 reset** | 638.7 / 641.0 s |

None of the falls or resets came from these changes (each traced): L9 seed 2 run 1 fell at the 0.20 m trench (4, 0)
after the rough cell (3, 0) (classified correctly, crossed at 0.9 m/s; the maze's known trench risk), then, on the
long route, stumbled while stepping sideways on the rough cell (4, 6) to line up for the real spike lane (4, 5) (the
rough-floor staging problem, as L9 seed 3 in "Route hints"). Both L7 seed 2 resets are the stuck check below. The
other runs are within ±1.5 s of each other and of the bar; L9 seed 2's ramps are now steep ramps throughout.

**L7 seed 2's two resets: the 10 s stuck check, not a fall and not from these changes.** Every version walks into the
dead end east of (4, 0) at t ≈ 10 s and passes the t = 10 s spot again at t ≈ 20 s (0.37–0.9 m clear in 5 earlier
runs). In 2 of the last 4 runs a snag on the rough cell (4, 0) slowed it (all 4 re-entered (4, 0) at 17.0–17.1 s; the
classifier changes never touch rough cells); the stuck guard backed it away from the spot (it always steps away on
the same side), momentum carried it to 0.14 m and it ended 0.21 m from the spot (limit 0.25 m). Fixed in the next
section (walking on through at the commanded speed; a fixed 0.5 m/s would not have been enough).

### Stuck guard: walk on through the last check's spot (2026-11-08, committed `040977b`)

**Problem** (the two L7 seed 2 resets above; traces `v5_bench` run 1 and `v6_bench` run 2 of the slalom sweep 2
session). The guard's command ramps from the 0.9 m/s rough-floor speed toward −0.25 m/s at 2 m/s², but the robot keeps
its speed for about 0.6 s after the guard wakes (run 1: 0.38–0.43 m/s until 19.1 s), then backs at only 0.12–0.15 m/s.
Run 1: 0.36 m short of the spot at 0.38 m/s → carried to 0.13 m, ended 0.22 m. Run 2: snagged 0.18 m short (0 m/s),
backed off only 0.07 m, ended 0.248 m. The "stop" branch (beyond 0.35 m) has the same flaw. Walking on at a fixed
0.5 m/s would also have failed run 1 (~0.24 m past the spot); at the 0.9 m/s already commanded it ends ~0.5 m past.

**Rule (`_stuck_guard`).** The old command (stop beyond 0.35 m, else step away on the robot's side) stays the default.
A "through" command replaces it when its predicted distance from the spot at check time is larger: along the travel
axis at the speed already commanded (at least `STUCK_GUARD_THROUGH` = 0.5 m/s), scaled (not clipped) into the trained
ranges so it stays on the axis, no yaw. Prediction: the measured velocity for `STUCK_GUARD_LAG` = 0.6 s, then the
command times `STUCK_GUARD_GAIN` (0.83 forward, 0.55 backward, 0.5 sideways). It gives 0.25 / 0.26 m for the old rule
in the two reset traces (real: 0.22 / 0.248 m) and 0.52 / 0.54 m for walking on. Through only when (a) facing within
45° of the travel direction (walking, not a sideways step), (b) in a plain / rough cell, and (c) either stopping 0.45 m
short of the cell border ahead or with the side ahead known open onto a plain / rough cell (never into a wall, or into
a trench / ramp / spike / ice cell without the pilot's lining up and speed rules). Otherwise the old rule.

Tests: `test_stuck_guard_walks_on_through_the_last_check_position` (0.33 m short at 0.36 m/s, and 0.40 m short where
the old rule stopped; fails on the old pilot), `test_stuck_guard_does_not_walk_through_into_a_hazard` (trench ahead:
steps back), `test_stuck_guard_does_not_step_sideways_through` (mid-turn: the old rule's stop; fails without the 45°
limit). The two older guard tests pass unchanged.

**Benchmark** (runs 1–2: first version, without the 45° limit; runs 3–4: final):

| Maze | Run 1 | Run 2 | Final run 3 | Final run 4 | Before (sweep 2 final) |
|---|---|---|---|---|---|
| L7 seed 1 | 59.5 | 60.2 | 60.1 | 59.3 | 59.4 / 59.4 |
| L7 seed 2 | 85.7 | 85.5 | 85.3 | 86.7 | 85.9 / 90.8 (stuck-check reset) |
| L7 seed 3 | 78.4 | 78.5 | 79.6 | 78.7 | 79.2 / 79.4 |
| L7 seed 4, random ends | 47.0 | 47.0 | 46.8 | 47.3 | 47.5 / 47.4 |
| L9 seed 1 | 115.6 | 115.8 | 116.3 | 116.2 | 115.9 / 116.0 |
| L9 seed 2 | 90.0 | 213.9 (3 falls, below) | 89.4 | 89.9 | 214.1 (2 falls) / 89.6 |
| L9 seed 3 | 86.7 | 88.5 | 86.8 | 86.6 | 85.0 / 87.1 |
| L9 seed 5, random ends | 88.4 | 79.6 | 83.5 | 79.9 | 78.9 / 80.1 |
| **Total** | **651.3 s, 0** | **769.0 s, 3 falls** | **647.8 s, 0** | **644.6 s, 0** | 765.9 s, 2 falls / 649.8 s, 1 reset |

The guard woke in only 2 of these 32 runs, both L9 seed 5 random ends at t = 48.5 s, mid-turn in the rough corner
(5, 4), 0.63 / 1.11 m from the spot. In run 2 the first version chose "through", which in that pose is a 0.35 m/s
sideways step on loose boxes (it ended 0.64 m from the spot, no harm), hence the 45° limit. In run 4 the final version
stopped, as the old rule did (0.55 m). In every other run the guard stayed idle, so those runs behave exactly like the
old pilot: run 2's L9 seed 2 falls were the trench (4, 0) at 15.8 s and two turns on the spot on ice ((4, 4) at 110 s,
(1, 1) at 124 s), and L9 seed 5 run 1 lost its time to dead-end exploration (17.1 s vs 8.7 s).

**L7 seed 2:** 16 runs with the new guard (10 first version, 6 final, the benchmark ones included), 85.2–88.4 s,
**0 resets**. But the deep snag never recurred: the robot slowed to only 0.38–0.70 m/s in (4, 0) and passed the
t = 10 s spot 0.87–1.38 m clear, so the guard never woke there (the two reset runs had 0.00–0.05 m/s). The new branch
is checked by the tests and the model check above, not yet in a closed-loop run. Traces and `guard_report.py` (lists
every guard episode per 10 s check, `uv run python guard_report.py <trace.csv> ...`):
`C:\Users\user\AppData\Local\Temp\claude\746074aed36627eb\68de3be3-5f37-4a3b-9bfe-a4792e8b4966\scratchpad\guard\`
(`v2\` = final code).

### Rough-corner line-up: sideways step vs a curve (2026-11-09/10, in the working tree, on since 2026-11-10)

**Problem.** Before a spike lane / slalom whose free lane is not straight ahead, the pilot lines up by stepping
sideways while turning to face the corridor (staging, `STAGE_HOLD_U`). Over the traced runs since 2026-11-07
(scratchpad `staging_stats.py` / `staging_corner.py`): rough corners 101 line-ups, **3 falls** (L9 seed 11 back into the
trench cell, L9 seed 2 (4, 6) and L9 seed 3 (5, 6) on the boxes), plus L9 seed 44 random ends (snag, then the unstuck
reflex into the wall); rough straight 27 / 0; plain corners 37 / 0. 99 of the 101 rough-corner cases have the lane on
the **outside** of the turn (0.63 m on along the way the robot walked in).

**Policy test** (new `scripts/corner_lineup_test.py`, task `G1-Rough-Test`'s L corridors: robots walk into the rough
corner cell, the cell after it is treated as a spike cell with the lane 0.48 m to the outside; pilot.py's controllers
copied; finish where the bed starts, u = −0.70):

| Maneuver | Robots | Fell / crashed | Stuck at 16 s | Snagged | Time | Lateral at u = −1.2 (min / median / max) |
|---|---|---|---|---|---|---|
| Sideways step while turning (pilot) | 1,250 | **4.0%** | 3.3% | 24% | 7.0 s | −0.60 / 0.28 / 0.79 |
| Walk on to the centreline, then the diagonal | 750 | 0.7% | 2.7% | 25% | 6.4 s | — (one at −0.53 at the bed) |
| Curve, ≤ 0.5 m/s | 750 | 0.7% | 1.9% | 26% | 6.0 s | — |
| Curve, ≤ 0.7 m/s | 1,250 | 0.2% | 0 | 6% | 4.5 s | 0.36 / 0.57 / 0.74 |
| **Curve, ≤ 0.9 m/s (rough speed)** | 1,250 | **0.2%** | 0 | **0.6%** | **3.7 s** | 0.42 / 0.59 / 0.71 |

The sideways step's falls cluster at the lane entry 0.74 m out (it overshoots toward the outer wall); slow line-ups on
boxes snag (as in the rough-floor test, slower is worse). "Curve" = pilot.py's pure-pursuit curve (1 m lookahead,
0.6 m/s² sideways limit, no v_y) along previous cell centre → corner centre moved out to the lane line → lane entry.

**Implemented** in `_compute_waypoint_and_speed` Case 2 (`lane_curve`: rough corner cell, 90° turn, still in the corner
cell, lane on the outside; `_polyline_point` factored out of `_pursuit_point`), tests
`test_rough_corner_into_outside_lane_is_walked_as_a_curve` (flag on; fails on the old pilot),
`test_rough_corner_curve_is_off_by_default`, `test_rough_corner_into_inside_lane_keeps_the_usual_approach`.

**Maze runs with the curve on** (stopped early by a session restart):

| Runs | Result |
|---|---|
| Benchmark run 1 (8 mazes) | 892.3 s, 2 falls, both unrelated: L9 seed 5 random ends turning on the spot in rough (6, 4), then an ice turn at (7, 8) sliding back into the spike lane (7, 7). L7 seed 4 random ends 43.8 s (was 47.0), L9 seed 2 87.4 s (was 89.4–89.9). |
| Benchmark run 2 (5 of 8) | L7 seeds 1–4 59.0 / 85.8 / 78.2 / 43.8 s, L9 seed 1 116.9 s, 0 falls |
| Forced short route, 11 runs (L9 seeds 1, 11, 24, 44 / 21 / 48 random ends, L7 seed 3) | 3 falls. L7 seed 3: an ice turn sliding back into a spike lane (unrelated). **L9 seed 11 run b:** an S-bend ((0, 7) → (1, 7) → (1, 6)), the robot entered the rough corner (1, 6) at 0.9 m/s still turning, the curve only bends 1 m before the moved corner point, it ran 0.33 m past the lane line (0.81 m off the centreline) and an arm hit the wall. **L9 seed 24 run b:** 0.22 m past the lane into the slalom (2, 5), stage 1 steered back over 0.9 m and left it 35° askew when stage 2 stepped across the gap: spike hit. Run a of both was clean (overshoot 0.14 m). |

So in the maze the curve overshoots outward more than in the test (straight 4 m run-ins only): **2 failures in about
two dozen curve corners**, against ~3% for the sideways step. So it was kept off (`ROUGH_CORNER_CURVE = False`) at first.

**Second try (2026-11-10): corner point inside the lane line.** `corner_lineup_test.py` got an S-bend run-in
(`CORNER_SBEND=0.45`: a 0.45 rad/s yaw-rate bias the wrong way in the last 1.2 m before the corner cell, as at L9 seed
11), the farthest point out before the bed (pelvis > 0.70 m off the centreline = an arm near the wall face at 0.925),
and `curveVVi<cm>` modes (corner point that far inside the lane line). 250 robots each, S-bend run-in, 0 falls in all:

| Variant | > 0.70 m out | Farthest out | Lateral at u = −1.2 (min / median) | Snagged | Time |
|---|---|---|---|---|---|
| ≤ 0.9 m/s, corner on the lane line (first try) | 5.2% | 0.76 m | 0.42 / 0.57 | 0% | 3.79 s |
| **≤ 0.9 m/s, corner 0.20 m inside** | **0.4%** | 0.70 m | 0.37 / **0.48** | 0% | 3.65 s |
| ≤ 0.7 m/s, corner 0.15 m inside | 0% | 0.63 m | 0.31 / 0.46 | 6.4% | 4.42 s |

`LANE_CURVE_INSET = 0.20`, `LANE_CURVE_SPEED = 0.9`. Maze runs with the curve on, 2026-11-10 (then switched on by
default, `ROUGH_CORNER_CURVE = True`):

| Runs | Result |
|---|---|
| Forced short route, L9 seed 11 × 2, L9 seed 24 × 2 | 95.4 / 96.3 s and 119.7 (1 fall) / 86.1 s. Seed 11's S-bend corner (1, 6) (the wall crash before) and seed 24's corner into the slalom (2, 5) (the spike hit before) clean in every run that reached them; seed 24 run a fell at the trench (1, 4) earlier (a normal corner into a trench, not this curve) |
| Benchmark set 1 | L9 seed 2 83.9, L9 seed 5 random ends 84.5, L7 seed 4 random ends 44.3, L9 seed 3 87.0, L7 seed 1 59.6, L7 seed 2 105.9 (1 fall: an ice turn on the spot at (4, 3) after the spike lane, the known ice-turn problem), L7 seed 3 78.6, L9 seed 1 116.7: **660.5 s, 1 fall** |
| Benchmark set 2 | L9 seed 2 86.7, L9 seed 5 random ends 82.1, L7 seed 4 random ends 44.0, L9 seed 3 88.1, L7 seed 1 59.5, L7 seed 2 85.4, L7 seed 3 78.2, L9 seed 1 116.4: **640.4 s, 0 falls** (committed pilot: 647.8 / 644.6 s; bar 638.7 / 641.0 s) |

The curve ran 12 times in these runs (L9 seed 2, L7 seed 4 random ends, the forced mazes), 0 falls within 4 s of one; no
sideways line-up on rough floor happened.

**Check with the curve on by default (2026-11-11):** 9 more forced short-route runs of mazes whose slalom is entered
from a rough corner with the lane outside: L9 seed 24 × 3 (84.4 / 84.4 / 87.2 s), seed 1 × 2 (109.9 / 113.3), seeds
7, 17, 40, 42 (105.8, 80.4, 88.4, 127.3 s). **0 falls; 9 slalom crossings, 0 spike hits** (nearest 0.29–0.34 m, as
before); 13 curves, entering the next cell 0.39–0.66 m off its centreline (lane 0.48, wall face 0.925), heading within
−15° to +5° (scratchpad `curve_check.py`). Seed 24's slalom corner, reached in only 1 of 2 runs before, was clean in
3 of 3. In total **25 curves in the maze, 0 falls**. Where comparable, as fast or faster than the sideways step:
seed 24 84–87 s (92.3 in sweep 2), seed 1 110–113 s (112–116), seed 17 80.4 s (86.6). Faster where it ran: L9 seed 2 83.9 / 86.7 s (89.4–89.9 before), L7 seed 4
random ends 44.3 / 44.0 s (47.0–47.5).

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
- **Turning on the spot on rough floor moves the robot** (0.26 m on average, up to 1.0 m: a foot catches a box), far
  more than on plain floor (0.11 m). Wall-clearing during the rotation (0.15 m/s) is too slow for that lurch. Avoid
  rotating on loose boxes where there is a choice (`_plan_rough_exit`).
- **Newton uses the larger friction of two touching shapes** (MuJoCo's rule) unless one has a higher `mjc:priority`.
  Randomizing the feet's friction below the floor's 1.0 does nothing; slippery ground has to be a separate mesh with
  priority, like the maze's ice (`HazardTerrainImporter`). `G1-Hazard-Pro`'s 0.2–1.0 foot friction never made ice.
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
   and sometimes falls. L9 seed 5 snags before its slalom. See item 7. **Lining up sideways at a rough corner before a
   spike cell falls in ~3% of those corners** (4.0% of robots in `scripts/corner_lineup_test.py`). A curve instead
   (`ROUGH_CORNER_CURVE`, **on** since 2026-11-10; corner point 0.20 m inside the lane, `LANE_CURVE_INSET`) passed the
   test, the forced runs and a 2 × 8 benchmark (660.5 s with 1 unrelated ice-turn fall / 640.4 s, 0 falls); 25 curves
   in the maze, 0 falls, L9 seed 24's slalom corner clean 3 of 3 on 2026-11-11 (section 3, "Rough-corner line-up"). The rough-floor test (section 3) says slower
   is worse: 20–28% of robots snag at the pilot's 0.68 m/s, 4–8% at 0.9 m/s with no falls. `ROUGH_SPEED = 0.9` in the
   maze (section 3, "Rough floor at 0.9 m/s in the maze"), with the follow-up fixes for rough corners into ice, the
   trench approach and the stuck guard: 638.7 / 641.0 s, 0 falls (section 3, "Rough corners into ice, trench
   approach, stuck guard"). **Open:** L9 seed 2's trenches right after a rough cell or a corner: (4, 0) (1 fall in
   the latest 8 crossings, 0 in 32 with the old pilot) and (8, 1) (1 fall on 2026-11-05, met at 0.74 m/s while still
   accelerating out of the corner); and the L7 seed 2 (3, 5) snag spot (a box on the curve's path, at any speed).
   *Fixed 2026-11-03:*
   - **Steep-ramp stalls** (0.29–0.30 m; L9 seed 2 stalled in 4 of 16 runs, about 120 s each) came from climbing at
     0.70 m/s. Steep ramps are now climbed at 0.9 m/s (section 3, "Ramp test").
   - **Ice turn-on-the-spot falls** (5% of ice turns) came from rotating on the middle of the ice. The turn is now
     decided on entering the ice cell, so the robot rotates near the edge (section 3, "Ice turn test"). Since
     2026-11-07 also next to spike cells (they had kept the old place and fell in 2 of 8 forced runs).
2. **Route choice at the fork.** In 4 of the 8 benchmark mazes (L7 seeds 1–3, L9 seed 1) the robot drives the long
   route, because `choose_direction` breaks the tie toward the cell it faces (unknown cells are priced as nearly plain).
   Clean short-route times with the route forced: L7 seed 1 ~54 s, seed 2 ~45 s, seed 3 ~58–62 s, L9 seed 1 ~130 s,
   against 63 / 91 / 82 / 122 s on the long route. Counting a ~10–18 s detour to look into the fork first, the gain is
   about 17 s per 8-maze set. The safe entry at corners whose spike lane is on the outside of the turn now exists
   (`STAGE_HOLD_U`, 2026-11-07): forced short routes L7 seed 3 56.0–60.2 s (long route ~79 s) and L9 seed 1
   115.4–116.7 s (long route ~116 s), 0 falls in 5 runs. The pilot may only use what it senses
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
6. **Slalom crossing speed:** the policy strafes at only about 0.18 m/s when commanded 0.30–0.35 (sideways test:
   48–66% of the command, weaker to the right). `G1-Hazard-Challenge` training did not fix it without breaking slow
   and backward walking (section 3, "Policy training setup"). If training is tried again: add a backward-walking
   command kind and a tighter forward-speed reward, and gate on a slow / backward walking test as well as the
   sideways, ice (180°), ramp and trench tests.
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
9. **Stuck guard on the way back from a dead end** (`_stuck_guard`): it always stepped away from the last check's spot
   on the side the robot was on, even when the robot was walking toward and through that spot; backing at −0.25 m/s
   against the momentum left L7 seed 2 0.21 m from the spot (reset at t = 20 s, section 3, "Slalom sweep 2").
   *Done 2026-11-08 (committed `040977b`):* the guard walks on through at the commanded speed (≥ 0.5 m/s) when that is
   predicted to end farther (a fixed 0.5 m/s would not have cleared that run), facing the travel direction, on plain /
   rough floor, never into a wall or hazard cell; benchmark 647.8 / 644.6 s, 0 falls (section 3, "Stuck guard: walk on
   through"). **Open:** not yet seen in a closed-loop run (the deep snag did not recur in 16 L7 seed 2 runs); if a
   stuck-check reset happens again, trace it and check what the guard chose.

## 6. Before submitting

- [ ] `uv run pytest tests` passes. `test_pilot_passes_server_upload_checks` checks the map-reading ban.
- [ ] 8-maze benchmark, at least 2 runs, no new falls compared with the table above.
- [ ] `PILOT_MODE` default is `short` in `pilot.py`.
- [ ] Upload `model_8497.pt` (as `policy.pt`) and `pilot.py` only. **Not** `tools/`, which reads the real map.
