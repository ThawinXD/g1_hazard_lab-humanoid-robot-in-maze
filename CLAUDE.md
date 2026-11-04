# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Isaac Lab 3 + Newton project (`g1_maze_lab` package) for a Unitree G1 humanoid course: Lab 7.1 (curriculum
fine-tuning), Lab 7.2 (maze autopilot) and **Challenge 7.5, the G1 Hazard Maze**, which is the active work. The
deliverable is a walking policy (`policy.pt`) plus a navigation pilot ([pilot.py](pilot.py)).

Read before starting work:
- [handoff.md](handoff.md): current state, the best checkpoint, the benchmark bar, gotchas and the prioritized open
  problems. Keep it up to date when the state changes.
- [develop.md](develop.md): full write-up of methods and results (curriculum fix, pilot layers, dead-end work).
- [README.md](README.md): course instructions for all labs and the challenge rules/server.
- [HAZARD_CHALLENGE_PLAN.md](HAZARD_CHALLENGE_PLAN.md): exact hazard geometry, scan signatures and crash rules.

## Commands

Run from this folder. On Windows always set `PYTHONIOENCODING=utf-8` (Isaac Lab crashes when output is redirected
otherwise).

```bash
uv run pytest tests                                   # no simulator needed
uv run pytest tests/test_pilot_review.py -k <name>    # single test
uv run ruff check . && uv run ruff format .           # line length 120
```

Run the challenge locally (add `--viz newton` to watch, `--random-ends`, `CHALLENGE_SHOW_MAZE=1` to print the real maze):

```bash
PILOT_MODE=short PYTHONIOENCODING=utf-8 uv run python scripts/challenge.py --level 9 --seed 3 --pilot pilot.py --policy logs/rsl_rl/g1_hazard_pro/2026-10-31_17-46-45_skill_curr1500/model_8497.pt
```

Benchmark (8 mazes; snapshot `pilot.py` first because every run re-loads it):

```bash
cp pilot.py /tmp/snap_pilot.py && PILOT=/tmp/snap_pilot.py OUT=/tmp/bench_a SPECS="7:1 7:2 7:3 7:4:--random-ends 9:1 9:2 9:3 9:5:--random-ends" bash tools/bench.sh
```

Trace a fall with [tools/trace_pilot.py](tools/trace_pilot.py) (wrapper pilot: `PILOT_SRC=pilot.py TRACE_OUT=... --pilot tools/trace_pilot.py`, see handoff.md §3).
Split a run's time by cause with `uv run python tools/time_breakdown.py <trace.csv> ...`.

Trench / ramp crossing success rate per width or height and speed (500 robots in walled corridors, ~90 s;
`TRENCH_SPEEDS`, `TRENCH_YAW`, `TRENCH_START`, `TRENCH_TIME`, `TRENCH_REPEATS` env vars apply to both):

```bash
PYTHONIOENCODING=utf-8 uv run python scripts/trench_test.py --checkpoint <model.pt>
PYTHONIOENCODING=utf-8 uv run python scripts/trench_test.py --task G1-Ramp-Test --checkpoint <model.pt>
```

Turns on the spot on ice like the pilot's (`ICE_RATES`, `ICE_TURNS`, `ICE_SPEED`, `ICE_BRAKE`, `ICE_BRAKE_U`,
`ICE_REPEATS`; the corner version turns at an ice corner with plain / rough / ramp / spike cells after it):

```bash
PYTHONIOENCODING=utf-8 uv run python scripts/ice_turn_test.py --task G1-Ice-Corner-Test --checkpoint <model.pt>
```

Rough-floor snags (corridors of maze rough cells: straight, lining up, L corners; `ROUGH_SPEEDS`, `ROUGH_TIME`,
`ROUGH_REPEATS`) and sideways steps / straight-walking drift on open floor (`STRAFE_SPEEDS`, `STRAFE_WALK`,
`STRAFE_TIME`, `STRAFE_SETTLE`, `STRAFE_REPEATS`):

```bash
PYTHONIOENCODING=utf-8 uv run python scripts/rough_test.py --checkpoint <model.pt>
PYTHONIOENCODING=utf-8 uv run python scripts/strafe_test.py --checkpoint <model.pt>
```

Training: `uv run isaaclab train --rl_library rsl_rl --task G1-Hazard-Challenge --num_envs 1024 --checkpoint <ckpt>`
(`G1-Hazard-Pro` trained `model_8497`; `G1-Hazard-Challenge` adds real ice, pilot-like commands, a sideways reward and
harder hazards, see handoff.md §3). Checkpoints land in `logs/rsl_rl/<experiment>/<run>/` (git-ignored). Judge a new
checkpoint with the trench / ramp / ice / rough / sideways tests against `model_8497` before the maze benchmark.

## Architecture

- **Runner** [scripts/challenge.py](scripts/challenge.py): builds the hazard maze env (`--level 7` → `G1-Maze-Hard`,
  `9` → `G1-Maze-Hard9`), loads the policy and the pilot, enforces the rules (fall / body crash / spike bed / 10 s
  stuck → reset to start, clock keeps running) and prints a `RESULT` line. `DefaultPilot` there defines the pilot
  interface: `Pilot(num_envs, num_cells, cell_size, device, start_cell, goal_cell)`, `act(obs)` returning velocity
  commands, `reset(env_ids)`. `challenge_server.py` / `challenge_runner.py` are the instructor upload/leaderboard side.
  **Don't change the rules in these scripts.**
- **Pilot** [pilot.py](pilot.py): one `Pilot` class built on `MazeAutopilot` ([src/g1_maze_lab/autopilot.py](src/g1_maze_lab/autopilot.py)).
  Layers: perception (height scan + lidar wall votes → walls and hazard class per cell), memory kept across resets
  (walls, cell types/costs, death penalties), planning (cost-weighted flood fill to the goal, `COSTS_SHORT` /
  `COSTS_AUTO` by `PILOT_MODE`), and control (waypoint/speed per hazard, pivot turn-back, stuck guard, unstuck
  reflex). All tunables are module constants at the top. The method table is in handoff.md §2.
- **Maze generation** `src/g1_maze_lab/terrains/` (`maze.py`, `hazard_maze.py`): procedural maze with a short
  (hard hazards) and a long (easier) route from the start fork.
- **Training tasks** `src/g1_maze_lab/tasks/` registered via the `isaaclab.tasks` entry point: `curriculum/`
  (Lab 7.1, incl. the `terrain_levels_skill` curriculum term in `mdp/curriculums.py`), `maze/`,
  `hazard_starter/` (`G1-Hazard-Starter`, `G1-Hazard-Pro`; training terrain has no walls) and `hazard_challenge/`
  (`G1-Hazard-Challenge`: maze ice through the hazard importer, `PilotLikeVelocityCommand`, `track_lin_vel_y_exp`).
  `trench_test/` holds the policy test tasks (`G1-Trench-Test`, `-Ramp-`, `-Ice-`, `-Ice-Corner-`, `-Rough-`,
  `-Strafe-Test`).
- **Friction:** Newton uses the larger friction of two touching shapes unless one has a higher `mjc:priority`, so
  slippery ground must be a separate prioritized mesh (as the maze's ice); lowering the feet's friction does nothing.
- **Tests** `tests/`: simulator-free. `test_pilot_review.py` has a regression test per pilot fix with synthetic
  scan/lidar/wall helpers; `test_challenge_upload.py` mirrors the server's upload checks.

## Rules and constraints

- **Upload ban:** `pilot.py` (and any helper it imports) must not mention `GENERATED`, `HAZARDS`, `HARD_CELLS` or
  `g1_maze_lab.terrains`, or the upload is refused. Never copy code from `tools/` (it reads the real map) into
  `pilot.py`. `test_pilot_passes_server_upload_checks` guards this.
- `policy.pt` must keep the exact actor layer shapes of `pretrained/g1_rough/model_pretrained.pt`; the policy's
  observations and height scanner can't change.
- Keep the user's constants: `PURSUIT_LOOKAHEAD = 1`, `TURN_DECIDE_DIST = 1.2`, `ap.max_acc = 2`, `PILOT_MODE`
  default `short` (the official runner sets no env vars).
- Simulator runs are not repeatable (the same seed can differ by 50+ s). Judge a pilot change only with the 8-maze
  benchmark, at least 2 runs, against the table in handoff.md: no new falls, total time not worse.
- Killing a background benchmark leaves `challenge.py` simulator processes running; find and kill them
  (`Get-CimInstance Win32_Process | ? { $_.CommandLine -like '*challenge.py*' }`).
- The wall map latch is permanent, so new wall-sensing code must never produce false walls (check the trace's
  `# MAP` lines).
- `logs/` is git-ignored: the best checkpoint exists only on this machine.
- Ask before committing.
- Code style in `pilot.py`: an end-of-line comment explaining nearly every line; match it.
