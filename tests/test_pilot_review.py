"""Regression tests for issues found in the pilot.py review (no simulator, only torch).

Each test reproduces one problem and is EXPECTED TO FAIL on the current pilot.py.
When you fix the problem, its test should pass. Run:

    uv run pytest tests/test_pilot_review.py -v

Scans are built in code, like tests/test_option_b.py: a 33 x 33 grid every 0.1 m around the robot
(axis-aligned, robot facing +x). Hazard shapes copy the defaults in g1_maze_lab/terrains/hazard_maze.py:
cell 2.0 m, wall 0.15 m (corridor half-width 0.925 m), ramp slopes 0.75 m long with a flat top for |u| <= 0.25,
spike beds from the wall to just short of the middle, slalom beds at |u| in [0.3, 0.9].
"""

from __future__ import annotations

import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pilot import (  # noqa: E402
    CELL_ICE,
    CELL_LOW_RAMP,
    CELL_SPIKE_LANE,
    CELL_SPIKE_SLALOM,
    CELL_PLAIN,
    CELL_ROUGH,
    CELL_STEEP_RAMP,
    CELL_TRENCH,
    PIVOT_BRAKE_ACC_ICE,
    ICE_SLOW_DIST,
    PIVOT_ROUGH_EXIT,
    PIVOT_TURN_RATE_ICE,
    RAMP_RUNUP,
    ROUGH_SPEED,
    STAGE_HOLD_U,
    STEEP_RAMP_SPEED,
    TRENCH_CLEAR_U,
    TRENCH_SPEED,
    Pilot,
)
from g1_maze_lab.autopilot import DIRS  # noqa: E402

CELL = 2.0
HW = 0.925  # half corridor width = 0.5 * (cell_size - wall_thickness)
DT = 0.02


# ----------------------------------------------------------------------------- helpers
def _pilot(monkeypatch, mode: str = "auto") -> Pilot:
    monkeypatch.setenv("PILOT_MODE", mode)
    monkeypatch.delenv("PILOT_SPEED", raising=False)
    return Pilot(num_envs=1, num_cells=7, cell_size=CELL, device="cpu")


def _grid(cx: float, cy: float) -> torch.Tensor:
    """Height-scanner hit XY (1089, 2): 3.2 m x 3.2 m every 0.1 m around (cx, cy)."""
    xs = torch.linspace(cx - 1.6, cx + 1.6, 33)
    ys = torch.linspace(cy - 1.6, cy + 1.6, 33)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def _uv(xy: torch.Tensor, ci: int, cj: int, axis: int = 0):
    """(along, across) coordinates relative to the centre of cell (ci, cj)."""
    dx, dy = xy[:, 0] - ci * CELL, xy[:, 1] - cj * CELL
    return (dx, dy) if axis == 0 else (dy, dx)


def _ramp(xy, ci, cj, height, axis=0):
    u, v = _uv(xy, ci, cj, axis)
    z = height - (height + 0.01) * (u.abs() - 0.25).clamp(min=0.0) / 0.75
    return torch.where((u.abs() < 1.0) & (v.abs() < 1.0), z, torch.zeros_like(z))


def _bed(xy, ci, cj, side, u0, u1, height=0.30, edge=0.05, axis=0):
    """Spike bed as a solid block: from the wall on ``side`` (+1 / -1) to ``edge`` short of the middle."""
    u, v = _uv(xy, ci, cj, axis)
    lateral = side * v
    return torch.where((u > u0) & (u < u1) & (lateral > edge) & (lateral < HW), height, 0.0)


def _trench(xy, ci, cj, width=0.22, axis=0):
    u, v = _uv(xy, ci, cj, axis)
    return torch.where((u.abs() < 0.5 * width) & (v.abs() < HW), -1.0, 0.0)


def _rough(xy, ci, cj, axis=0):
    """A few 5 cm boxes; the two near the -u edge are visible from the previous cell's centre."""
    u, v = _uv(xy, ci, cj, axis)
    z = torch.zeros_like(u)
    for uc, vc in ((-0.6, -0.4), (-0.6, 0.3), (0.0, 0.0), (0.5, -0.3), (0.6, 0.4)):
        z = torch.where(((u - uc).abs() < 0.15) & ((v - vc).abs() < 0.15), 0.05, z)
    return z


def _obs(pos, z_fn, t, yaw=0.0, segments=None, lin_vel=None) -> dict:
    """Sensor dict like challenge.py. ``segments`` = wall segments ((x0, y0), (x1, y1)) seen by the 72-ray lidar."""
    xy = _grid(*pos)
    obs = {
        "pos_xy": torch.tensor([pos], dtype=torch.float32),
        "yaw": torch.tensor([yaw], dtype=torch.float32),
        "t": t,
        "dt": DT,
        "scan_xy": xy.unsqueeze(0),
        "scan_z": z_fn(xy).unsqueeze(0),
    }
    if segments is not None:
        dist, ang = _lidar(pos, segments, yaw)
        obs["lidar_dist"], obs["lidar_yaw"] = dist.unsqueeze(0), ang.unsqueeze(0)
    if lin_vel is not None:
        obs["lin_vel_b"] = torch.tensor([[lin_vel[0], lin_vel[1], 0.0]])
    return obs


def _lidar(pos, segments, yaw=0.0, max_range=6.0):
    """72 horizontal rays every 5 deg (world yaw), ray-cast against 2D wall segments; misses read 1e6."""
    ang = torch.tensor([yaw + math.radians(5 * k) for k in range(72)])
    dist = torch.full((72,), 1e6)
    px, py = pos
    for (x0, y0), (x1, y1) in segments:
        ex, ey = x1 - x0, y1 - y0
        for k in range(72):
            ux, uy = math.cos(float(ang[k])), math.sin(float(ang[k]))
            den = ux * ey - uy * ex
            if abs(den) < 1e-9:
                continue
            t = ((x0 - px) * ey - (y0 - py) * ex) / den  # Distance along the ray
            s = ((x0 - px) * uy - (y0 - py) * ux) / den  # Position along the segment (0..1)
            if 0.0 < t < max_range and 0.0 <= s <= 1.0 and t < dist[k]:
                dist[k] = t
    return dist, torch.atan2(torch.sin(ang), torch.cos(ang))


def _walls_z(segments):
    """Height-scan z function: 1.0 m within 0.075 m of any wall segment (wall thickness 0.15 m), floor elsewhere."""

    def z_fn(xy):
        z = torch.zeros(len(xy))
        for (x0, y0), (x1, y1) in segments:
            lo_x, hi_x = min(x0, x1) - 0.075, max(x0, x1) + 0.075
            lo_y, hi_y = min(y0, y1) - 0.075, max(y0, y1) + 0.075
            inside = (xy[:, 0] >= lo_x) & (xy[:, 0] <= hi_x) & (xy[:, 1] >= lo_y) & (xy[:, 1] <= hi_y)
            z = torch.where(inside, torch.ones_like(z), z)
        return z

    return z_fn


def _close_wall(p: Pilot, ci: int, cj: int, d: int):
    di, dj = DIRS[d]
    p.ap.walls[0, ci, cj, d] = 1
    p.ap.walls[0, ci + di, cj + dj, (d + 2) % 4] = 1


def _set_plan(p: Pilot, origin, target, heading: int):
    """Pretend the robot already decided in ``origin`` to walk to ``target``."""
    p.ap.origin[0] = torch.tensor(origin)
    p.ap.target[0] = torch.tensor(target)
    p.ap.heading[0] = heading
    p.ap.need_decision[0] = False


def _count_plans(p: Pilot) -> list:
    calls = []
    original = p.choose_direction

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    p.choose_direction = counting
    return calls


# ----------------------------------------------------------------------------- review #1
def test_stuck_penalty_still_changes_the_plan(monkeypatch):
    """Review #1: the +15 'stuck' penalty is overwritten by _set_cell_hazard() on the next frame.

    Robot at the centre of (1, 0), north wall closed, rough cell (2, 0) ahead. Without a penalty, east costs 11.8
    and turning back west costs 13.8, so it goes east. The first stall 1 m before (2, 0) is a snag on this cell's
    floor and keeps the plan; after a second stall in the same cell, east should cost 26.8 and lose to west.
    """
    p = _pilot(monkeypatch)
    _close_wall(p, 1, 0, 1)
    pos = (2.0, 0.0)
    p.pos_anchor[0] = torch.tensor(pos)  # robot has not moved since t = 0

    p.act(_obs(pos, lambda xy: _rough(xy, 2, 0), t=1.0))
    assert int(p.ap.heading[0]) == 0, "setup: without a penalty the robot should head east"

    p.act(_obs(pos, lambda xy: _rough(xy, 2, 0), t=10.0))  # stuck since t = 0 -> reflex, no penalty yet
    assert float(p.unstuck_until[0]) > 10.0, "setup: the unstuck reflex should have fired"
    p.act(_obs(pos, lambda xy: _rough(xy, 2, 0), t=10.0 + DT))
    assert int(p.ap.heading[0]) == 0, "one snag 1 m before an ordinary cell changed the route"

    p.act(_obs(pos, lambda xy: _rough(xy, 2, 0), t=14.0))  # stuck again in the same cell -> penalty on (2, 0)
    assert float(p.unstuck_until[0]) > 14.0, "setup: the second unstuck reflex should have fired"
    p.act(_obs(pos, lambda xy: _rough(xy, 2, 0), t=14.0 + DT))  # replans with the penalty
    assert int(p.ap.heading[0]) != 0, (
        f"still heading east into the cell it got stuck on; its cost is back to "
        f"{float(p.cell_cost[0, 2, 0]):.2f} (the +15 was overwritten when the cell was re-classified)"
    )


# ----------------------------------------------------------------------------- review #2
def test_steep_ramp_seen_from_previous_cell_is_not_low(monkeypatch):
    """Review #2: from the previous cell's centre only the near slope is visible.

    For a 0.24 m ramp the highest visible point is ~0.19 m, below the 0.205 m threshold, so the ramp is
    stored as LOW_RAMP (cost 1.45 instead of 22). Leaving it unknown until the top is seen would also be fine.
    """
    p = _pilot(monkeypatch)
    xy = _grid(2.0, 0.0)
    p._classify_cell_from_scan(0, 2, 0, xy, _ramp(xy, 2, 0, height=0.24), 0)
    assert int(p.cell_type[0, 2, 0]) != CELL_LOW_RAMP, "0.24 m (steep) ramp classified as LOW_RAMP"


def test_steep_ramp_is_not_downgraded_by_a_later_partial_view(monkeypatch):
    """Review #2: a full view says STEEP_RAMP, a later partial view must not overwrite it."""
    p = _pilot(monkeypatch)
    full = _grid(4.0, 0.0)
    p._classify_cell_from_scan(0, 2, 0, full, _ramp(full, 2, 0, height=0.24), 0)
    assert int(p.cell_type[0, 2, 0]) == CELL_STEEP_RAMP, "setup: full view should give STEEP_RAMP"

    part = _grid(2.0, 0.0)  # e.g. the robot backed out to the previous cell
    p._classify_cell_from_scan(0, 2, 0, part, _ramp(part, 2, 0, height=0.24), 0)
    assert int(p.cell_type[0, 2, 0]) == CELL_STEEP_RAMP, (
        f"steep ramp downgraded to type {int(p.cell_type[0, 2, 0])} by a partial view"
    )


# ----------------------------------------------------------------------------- review #3
def test_slalom_does_not_advance_into_second_bed_before_crossing(monkeypatch):
    """Review #3: past u = +0.25 the slalom code jumps to stage 3 even if the lane change is not finished.

    Slalom in (2, 0): bed 1 (u < -0.3) on +y, bed 2 (u > +0.3) on -y, so the exit lane is +y. The robot is at
    u = +0.27 but still at y = -0.40, right in front of bed 2. The waypoint must not move forward into bed 2.
    """
    p = _pilot(monkeypatch)
    p.cell_type[0, 2, 0] = CELL_SPIKE_SLALOM
    p.cell_axis[0, 2, 0] = 0
    p.safe_offset[0, 2, 0, 0] = -0.48  # u < 0 half: free lane on -y
    p.safe_offset[0, 2, 0, 1] = 0.48  # u > 0 half: free lane on +y
    _set_plan(p, origin=(2, 0), target=(3, 0), heading=0)

    x, y = 4.27, -0.40
    xy = _grid(x, y)
    z = _bed(xy, 2, 0, side=+1, u0=-0.9, u1=-0.3) + _bed(xy, 2, 0, side=-1, u0=0.3, u1=0.9)
    wp, _, _ = p._compute_waypoint_and_speed(0, torch.tensor([x, y]), 0.8, torch.tensor([2, 0]), xy, z)

    u_aim = float(wp[0]) - 2 * CELL
    assert u_aim <= 0.30, (
        f"waypoint at u = {u_aim:+.2f} (inside bed 2, which starts at u = +0.30) while the robot is still "
        f"{0.48 - y:.2f} m from the exit lane"
    )


def test_slalom_past_the_gap_off_lane_walks_on_not_back_along_the_bed(monkeypatch):
    """Level 9 seed 3 random ends (forced short route): snagged at the slalom's exit, the unstuck reflex pushed the
    robot back to u = +0.61 and 0.28 m off the exit lane. Stage 2 then strafed it back toward the gap (u = -0.10)
    along bed 2 into the spikes. Past the gap it must walk on toward the exit lane instead."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 2, 0] = CELL_SPIKE_SLALOM
    p.cell_axis[0, 2, 0] = 0
    p.safe_offset[0, 2, 0, 0] = -0.48  # u < 0 half: free lane on -y (bed 1 on +y)
    p.safe_offset[0, 2, 0, 1] = 0.48  # u > 0 half: free lane on +y (bed 2 on -y)
    _set_plan(p, origin=(2, 0), target=(3, 0), heading=0)
    x, y = 4.61, 0.20  # u = +0.61, 0.28 m off the exit lane toward bed 2
    xy = _grid(x, y)
    z = _bed(xy, 2, 0, side=+1, u0=-0.9, u1=-0.3) + _bed(xy, 2, 0, side=-1, u0=0.3, u1=0.9)
    wp, _, _ = p._compute_waypoint_and_speed(0, torch.tensor([x, y]), 0.3, torch.tensor([2, 0]), xy, z)
    assert not bool(p.strafe[0]), "strafing back toward the gap along bed 2"
    assert float(wp[0]) - 2 * CELL >= 0.95, f"waypoint {wp.tolist()} is not ahead past the bed"
    assert abs(float(wp[1]) - 0.48) < 0.01, f"waypoint {wp.tolist()} is not on the exit lane"


# ----------------------------------------------------------------------------- review #4
def test_unstuck_reflex_never_strafes_toward_spike_bed(monkeypatch):
    """Review #4: the unstuck reflex strafes +-0.20 m/s, flipping side every time, whatever is next to it.

    Robot in the free lane of a spike cell (bed on +y), stuck twice. Neither reflex may strafe toward the bed.
    """
    p = _pilot(monkeypatch)
    side = +1
    pos = (4.0, -0.45)
    _set_plan(p, origin=(2, 0), target=(3, 0), heading=0)
    p.pos_anchor[0] = torch.tensor(pos)

    vys = []
    for t in (10.0, 14.0):  # second event: 4 s after the first, reflex (1.2 s) already over
        cmd = p.act(_obs(pos, lambda xy: _bed(xy, 2, 0, side=side, u0=-0.7, u1=0.7), t=t))
        assert float(p.unstuck_until[0]) > t, "setup: the unstuck reflex should have fired"
        vys.append(float(cmd[0, 1]))

    assert all(vy * side <= 0.0 for vy in vys), (
        f"unstuck v_y commands {vys} (robot frame, facing +x): at least one strafes toward the spike bed on +y"
    )


def test_unstuck_reflex_does_not_turn_in_a_spike_cell(monkeypatch):
    """Level 9 seed 3 random ends: snagged at a slalom's exit, the reflex turned 40 deg while backing up and the rear
    swung into the bed beside the lane. In a spike cell it backs up facing down the corridor, stepping away from the
    cones; elsewhere it still turns."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 2, 0] = CELL_SPIKE_LANE
    pos = (4.0, -0.45)  # In the free lane of (2, 0), bed on +y
    _set_plan(p, origin=(2, 0), target=(3, 0), heading=0)
    p.pos_anchor[0] = torch.tensor(pos)
    cmd = p.act(_obs(pos, lambda xy: _bed(xy, 2, 0, side=+1, u0=-0.7, u1=0.7), t=10.0))
    assert float(p.unstuck_until[0]) > 10.0, "setup: the unstuck reflex should have fired"
    assert float(cmd[0, 0]) < 0.0 and float(cmd[0, 1]) <= 0.0, f"reflex {cmd[0].tolist()} not backing away from the bed"
    assert float(cmd[0, 2]) == 0.0, f"reflex turns at {float(cmd[0, 2]):.2f} rad/s in a spike cell"

    p = _pilot(monkeypatch)  # Plain cell: the reflex still turns to free the feet
    _set_plan(p, origin=(2, 0), target=(3, 0), heading=0)
    p.pos_anchor[0] = torch.tensor(pos)
    cmd = p.act(_obs(pos, lambda xy: torch.zeros(len(xy)), t=10.0))
    assert abs(float(cmd[0, 2])) == 0.75, "reflex no longer turns on plain floor"


def _choose(p: Pilot, cell, heading: int, open_dirs) -> int:
    """choose_direction for robot 0 in ``cell`` facing ``heading`` with the given open sides (DIRS indices)."""
    sides = torch.zeros(1, 4, dtype=torch.bool)
    sides[0, list(open_dirs)] = True
    return int(p.choose_direction(torch.tensor([cell]), sides, torch.tensor([heading]))[0])


def test_junction_prefers_the_branch_that_shows_a_route_hazard(monkeypatch):
    """Short mode, junction (2, 2) facing north: north (2, 3) unknown and east (3, 2) a spike lane are equally far from
    the goal (6, 6). The plain cost picks north (spike lanes cost more); the generator only puts spike lanes, trenches,
    slaloms and ramps on start-goal routes, so east cannot be a dead end and is taken. Auto mode keeps avoiding it."""
    for mode, expect in (("short", 0), ("auto", 1)):
        p = _pilot(monkeypatch, mode=mode)
        p._set_cell_hazard(0, 3, 2, CELL_SPIKE_LANE)
        assert _choose(p, (2, 2), 1, (0, 1)) == expect, f"{mode} mode: wrong branch at the junction"

    p = _pilot(monkeypatch, mode="short")  # Loose boxes: the short route's fill, a weaker hint
    p._set_cell_hazard(0, 3, 2, CELL_ROUGH)
    assert _choose(p, (2, 2), 1, (0, 1)) == 0, "rough branch (short route fill) not preferred over unknown"


def test_route_hint_never_overrides_death_penalties_or_points_back(monkeypatch):
    """No credit for a hazard cell the robot already fell in (+50), nor for the way back (the route behind leads to
    the start)."""
    p = _pilot(monkeypatch, mode="short")
    p._set_cell_hazard(0, 3, 2, CELL_SPIKE_LANE)
    p.cell_deaths[0, 3, 2] = 1.0
    p.cell_cost[0, 3, 2] += 50.0
    assert _choose(p, (2, 2), 1, (0, 1)) == 1, "went back into the spike lane it fell in"

    p = _pilot(monkeypatch, mode="short")
    p._set_cell_hazard(0, 2, 1, CELL_SPIKE_LANE)  # Behind the robot (it came from the south)
    assert _choose(p, (2, 2), 1, (1, 3)) == 1, "turned back toward a hazard behind the robot"

    # Level 9 seed 10 random ends: a route hazard on a step away from the goal may be the route's way back to the start
    p = _pilot(monkeypatch, mode="short")
    p._set_cell_hazard(0, 1, 2, CELL_SPIKE_LANE)  # West of (2, 2), one step farther from the goal (6, 6)
    assert _choose(p, (2, 2), 1, (1, 2)) == 1, "followed a route hazard away from the goal"

    # Level 9 seed 5: back out of a dead end, facing the way it came: a hazard it already crossed is no clue
    p = _pilot(monkeypatch, mode="short")
    p._set_cell_hazard(0, 3, 2, CELL_LOW_RAMP)
    p.visited[0, 3, 2] = True
    assert _choose(p, (2, 2), 1, (0, 1)) == 1, "went back over a hazard it had already crossed"


# ----------------------------------------------------------------------------- review #5
def test_planner_does_not_rerun_every_step_toward_a_chosen_hazard(monkeypatch):
    """Review #5: hazard_surprise is true on every step while walking toward a hazard the planner picked.

    Short mode, north wall closed: crossing the trench in (2, 0) costs 12.95, turning back costs 13.8, so the
    robot keeps going. One replan when the trench first appears is fine; one per step is not.
    """
    p = _pilot(monkeypatch, mode="short")
    _close_wall(p, 1, 0, 1)
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    calls = _count_plans(p)

    pos = (2.5, 0.0)  # the trench at the centre of (2, 0) has just come into view
    for k in range(5):
        p.act(_obs(pos, lambda xy: _trench(xy, 2, 0), t=1.0 + k * DT))

    assert int(p.ap.heading[0]) == 0, "setup: in short mode the robot should still cross the trench"
    assert len(calls) <= 1, f"planner ran {len(calls)} times in 5 steps with nothing new to react to"


# ----------------------------------------------------------------------------- review #8
def test_robot_past_target_centre_gets_a_new_target(monkeypatch):
    """Review #8: 'arrived' needs the robot within turn_radius (0.22 m near hazards) of the target centre.

    Here it is 0.6 m past the centre of the ramp cell (2, 0) and 0.25 m to the side (e.g. after a slip).
    No decision fires, so the waypoint stays behind it and the robot would turn around.
    """
    p = _pilot(monkeypatch)
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    p.act(_obs((4.6, 0.25), lambda xy: _ramp(xy, 2, 0, height=0.28), t=1.0))
    assert tuple(p.ap.target[0].tolist()) != (2, 0), "robot is past the centre of its target but never re-targets"


# ----------------------------------------------------------------------------- curved turns
def test_corner_is_decided_early_and_walked_as_a_curve(monkeypatch):
    """Corner cell (2, 0): east closed, north open. 0.8 m before its centre the robot must already pick (2, 1)
    and aim at a lookahead point inside the corner (a curve), not at the corner cell's centre."""
    p = _pilot(monkeypatch)
    _close_wall(p, 2, 0, 0)  # east wall of the corner cell
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    p.act(_obs((3.2, 0.0), lambda xy: torch.zeros(len(xy)), t=1.0))
    assert tuple(p.ap.target[0].tolist()) == (2, 1), "turn not decided before the corner cell's centre"
    assert bool(p.on_curve[0]), "turn not tracked as a curve"
    p.act(_obs((3.5, 0.0), lambda xy: torch.zeros(len(xy)), t=1.02))  # within the 0.7 m lookahead of the bend
    wx, wy = p.last_wp[0].tolist()
    assert 3.9 < wx < 4.05 and 0.05 < wy < 0.6, f"lookahead point ({wx:.2f}, {wy:.2f}) is not around the corner"


# ----------------------------------------------------------------------------- dead ends + turn-back on the spot


def _dead_end_east_of_1_0(p: Pilot):
    """(2, 0) closed east and north (south is the maze border), (1, 0) closed north: the only way on is back west."""
    _close_wall(p, 2, 0, 0)
    _close_wall(p, 2, 0, 1)
    _close_wall(p, 1, 0, 1)


def test_dead_end_target_turns_back_before_entering(monkeypatch):
    """Robot in (1, 0) walking to (2, 0). When (2, 0) turns out to be a dead end, it must re-decide right away
    (back west) while still in (1, 0), and turn back with the on-the-spot maneuver."""
    p = _pilot(monkeypatch)
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    p.act(_obs((2.6, 0.0), lambda xy: torch.zeros(len(xy)), t=1.0))  # Records the cost of the chosen target
    _dead_end_east_of_1_0(p)  # ... which is now seen to be a dead end
    p.act(_obs((2.6, 0.0), lambda xy: torch.zeros(len(xy)), t=1.02))
    assert tuple(p.ap.target[0].tolist()) == (0, 0), "did not turn back when the target became a dead end"
    assert int(p.pivot_state[0]) > 0, "turn-back is not done with the on-the-spot maneuver"


def test_robot_inside_dead_end_decides_immediately(monkeypatch):
    """Robot just inside the dead-end cell (2, 0): decide now instead of walking on to the cell's centre."""
    p = _pilot(monkeypatch)
    _dead_end_east_of_1_0(p)
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    p.act(_obs((3.2, 0.0), lambda xy: torch.zeros(len(xy)), t=1.0))
    assert tuple(p.ap.target[0].tolist()) == (1, 0), "kept walking into the dead end"


def test_lidar_maps_dead_end_corridor(monkeypatch):
    """Start (0, 0) facing east into a 3-cell corridor that ends at x = 5; north of the start is open.
    The lidar must map the corridor's far walls (beyond the height scan) so the planner goes north instead."""
    p = _pilot(monkeypatch)
    segs = [
        ((1.0, 1.0), (5.0, 1.0)),  # north walls of (1, 0) and (2, 0)
        ((5.0, -1.0), (5.0, 1.0)),  # east wall of (2, 0): the dead end
        ((-1.0, -1.0), (5.0, -1.0)),  # south border
        ((-1.0, -1.0), (-1.0, 3.0)),  # west border
        ((1.0, 1.0), (1.0, 3.0)),  # east wall of (0, 1); its north side stays open (the way on)
    ]
    for k in range(9):  # 3 sensing cycles (scan every 3rd step), robot standing at the start
        p.act(_obs((0.0, 0.0), _walls_z(segs), t=1.0 + k * DT, segments=segs))
    w = p.ap.walls[0]
    assert int(w[2, 0, 0]) == 1, "end wall of the corridor (x = 5) not mapped"
    assert int(w[2, 0, 1]) == 1, "north wall of (2, 0), 4 m away, not mapped"
    assert int(w[1, 0, 0]) == 0, "boundary between (1, 0) and (2, 0) not mapped as open"
    assert int(p.ap.heading[0]) == 1, "planner still walks into the dead-end corridor instead of going north"


def test_reverse_brakes_then_rotates_in_place(monkeypatch):
    """Walking east at 0.8 m/s when a turn-back is chosen: first brake to a stop without turning, then rotate on
    the spot (no forward or sideways motion)."""
    p = _pilot(monkeypatch)
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    p.act(_obs((2.6, 0.0), lambda xy: torch.zeros(len(xy)), t=1.0, lin_vel=(0.8, 0.0)))
    p.ap.cmd[0] = torch.tensor([0.8, 0.0, 0.0])
    _dead_end_east_of_1_0(p)
    v_prev, rotated = 0.8, False
    for k in range(80):
        cmd = p.act(_obs((2.6, 0.0), lambda xy: torch.zeros(len(xy)), t=1.02 + k * DT, lin_vel=(v_prev, 0.0)))
        v, vy, w = (float(x) for x in cmd[0])
        if v > 0.05:  # Still braking: no turning, no side-stepping
            assert abs(w) < 0.05 and abs(vy) < 0.05, f"turning ({w:+.2f} rad/s) while still walking at {v:.2f} m/s"
        if abs(w) > 0.4:
            assert abs(v) < 1e-6 and abs(vy) < 1e-6, f"rotating with v=({v:+.2f}, {vy:+.2f}) instead of on the spot"
            rotated = True
        v_prev = max(v, 0.0)  # The robot tracks the command
    assert rotated, "never rotated on the spot"


def _turn_back_from_rough(monkeypatch, here: int, ahead: int):
    """Walk east through (1, 0) (type ``here``) toward the dead end (2, 0) (type ``ahead``) at 0.8 m/s, the robot
    following the command; returns [(x, v, w)] per step after the turn-back is decided at x = 2.6."""
    p = _pilot(monkeypatch)
    flat = lambda xy: torch.zeros(len(xy))
    z_fn = (lambda xy: _rough(xy, 1, 0)) if here == CELL_ROUGH else flat
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    p.act(_obs((2.6, 0.0), z_fn, t=1.0, lin_vel=(0.8, 0.0)))
    p.ap.cmd[0] = torch.tensor([0.8, 0.0, 0.0])
    _dead_end_east_of_1_0(p)
    p.ap.walls[0, 1, 0, 0] = p.ap.walls[0, 2, 0, 2] = 0  # (1, 0) -> (2, 0) is a known opening
    x, v_prev, out = 2.6, 0.8, []
    for k in range(250):
        p.cell_type[0, 1, 0], p.cell_type[0, 2, 0] = here, ahead
        cmd = p.act(_obs((x, 0.0), z_fn, t=1.02 + k * DT, lin_vel=(v_prev, 0.0)))
        v, _, w = (float(c) for c in cmd[0])
        out.append((x, v, w))
        x += max(v, 0.0) * DT
        v_prev = max(v, 0.0)
    return out


def test_turn_back_in_rough_cell_rotates_on_the_plain_cell_ahead(monkeypatch):
    """Level 7 seed 2: the dead end (5, 0) is found from the rough cell (4, 0), 0.5 m before its border. Rotating on
    the loose boxes moved the robot up to 1 m (into the wall once). Walk on into the plain cell, brake from
    PIVOT_ROUGH_EXIT past the border and rotate there; on plain floor, or before a hazard cell, brake at once."""
    steps = _turn_back_from_rough(monkeypatch, CELL_ROUGH, CELL_PLAIN)
    line = 3.0 + PIVOT_ROUGH_EXIT
    for x, v, w in steps:
        if x < line - 0.02:
            assert v >= 0.4 - 1e-6 and abs(w) < 0.05, f"braking or turning at x = {x:.2f} before the plain cell"
    rotating = [x for x, v, w in steps if abs(w) > 0.4]
    assert rotating and min(rotating) > 3.0, "did not rotate on the plain cell"
    for here, ahead in ((CELL_PLAIN, CELL_PLAIN), (CELL_ROUGH, CELL_SPIKE_LANE)):
        x, v, w = _turn_back_from_rough(monkeypatch, here, ahead)[5]
        assert v < 0.7 and x < 2.8, f"did not brake at once (cell types {here} -> {ahead})"


def test_pivot_steps_away_from_close_wall_first(monkeypatch):
    """Turning back 0.3 m from a wall: step away from it (no rotation) before rotating."""
    p = _pilot(monkeypatch)
    segs = [((2.9, -1.0), (2.9, 1.0))]  # Wall 0.3 m in front of the robot
    _set_plan(p, origin=(1, 0), target=(0, 0), heading=2)
    p._start_pivot(0, 2)
    p.act(_obs((2.6, 0.0), _walls_z(segs), t=1.0, segments=segs, lin_vel=(0.0, 0.0)))  # Brake (already stopped)
    cmd = p.act(_obs((2.6, 0.0), _walls_z(segs), t=1.02, segments=segs, lin_vel=(0.0, 0.0)))
    v, vy, w = (float(x) for x in cmd[0])
    assert v < -0.1 and abs(w) < 1e-6, f"command ({v:+.2f}, {vy:+.2f}, {w:+.2f}) does not back away from the wall"


def test_pivot_does_not_trigger_unstuck(monkeypatch):
    """Standing still while turning on the spot is intended: the 3.5 s anti-stuck reflex must not fire."""
    p = _pilot(monkeypatch)
    _set_plan(p, origin=(1, 0), target=(0, 0), heading=2)
    p._start_pivot(0, 2)
    for k in range(225):  # 4.5 s; the test robot never actually turns, so the pivot stays active
        t = 1.0 + k * DT
        p.act(_obs((2.6, 0.0), lambda xy: torch.zeros(len(xy)), t=t, lin_vel=(0.0, 0.0)))
        assert float(p.unstuck_until[0]) < t, f"unstuck reflex fired during the turn-back at t = {t:.2f} s"


def test_normal_corner_does_not_pivot(monkeypatch):
    """A 90 deg corner is walked as a curve, never with the stop-and-rotate turn-back."""
    p = _pilot(monkeypatch)
    _close_wall(p, 2, 0, 0)
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    for k in range(5):
        p.act(_obs((3.2 + 0.02 * k, 0.0), lambda xy: torch.zeros(len(xy)), t=1.0 + k * DT))
        assert int(p.pivot_state[0]) == 0, "ordinary corner triggered the turn-back maneuver"


def test_stuck_guard_moves_off_last_check_position(monkeypatch):
    """challenge.py resets a robot that is within 0.25 m of where it was at the previous 10 s check. Standing (e.g.
    turning on the spot) at that same spot just before the next check, the pilot must move away along the corridor
    without turning; far from that spot it must leave the command alone."""
    p = _pilot(monkeypatch)
    _set_plan(p, origin=(1, 0), target=(0, 0), heading=2)
    p._start_pivot(0, 2)
    flat = lambda xy: torch.zeros(len(xy))
    p.act(_obs((2.6, 0.0), flat, t=10.0, lin_vel=(0.0, 0.0)))  # Check at t = 10 s records this position
    for k in range(20):  # 19.2 s .. 19.6 s: inside the guard window before the t = 20 s check
        cmd = p.act(_obs((2.6, 0.0), flat, t=19.2 + k * DT, lin_vel=(0.0, 0.0)))
    v, vy, w = (float(x) for x in cmd[0])
    assert math.hypot(v, vy) > 0.2 and abs(w) < 0.05, f"command ({v:+.2f}, {vy:+.2f}, {w:+.2f}) does not move away"
    assert p._stuck_guard(0, torch.tensor([4.0, 0.0]), 0.0) is None, "guard active although far from the check position"


def test_stuck_guard_sees_a_snagged_robot(monkeypatch):
    """Level 7 seed 2: walking back from a dead end, the robot snagged on a box 0.2 m from where the previous 10 s
    check saw it, still commanded 0.9 m/s. Predicted from the command it would be 1 m away by the next check, so the
    guard woke too late; predicted from the measured (zero) velocity it must act at once, 1.2 s before the check."""
    p = _pilot(monkeypatch)
    _set_plan(p, origin=(1, 0), target=(0, 0), heading=2)
    flat = lambda xy: torch.zeros(len(xy))
    p.act(_obs((2.2, 0.0), flat, t=10.0, yaw=math.pi, lin_vel=(0.9, 0.0)))  # Check at t = 10 s records this position
    p.ap.cmd[0] = torch.tensor([0.9, 0.0, 0.0])
    p.sim_time = 18.8  # 1.2 s before the t = 20 s check
    assert p._stuck_guard(0, torch.tensor([2.4, 0.0]), math.pi) is None, "setup: commanded speed predicts clearance"
    guard = p._stuck_guard(0, torch.tensor([2.4, 0.0]), math.pi, torch.tensor([0.0, 0.0, 0.0]))
    assert guard is not None and math.hypot(guard[0], guard[1]) > 0.2, "guard idle although the robot is not moving"


def test_no_new_target_inside_goal_cell(monkeypatch):
    """GOAL needs the robot within 0.6 m of the goal cell's centre. Entering the goal cell (6, 6) from the south,
    the pilot must keep walking to its centre, not pick a next cell and turn back out (ping-pong at the goal)."""
    p = _pilot(monkeypatch)
    _close_wall(p, 6, 5, 2)  # (6, 5) closed west: the corridor leads straight into the goal
    _set_plan(p, origin=(6, 5), target=(6, 6), heading=1)
    for k in range(5):
        p.act(_obs((12.0, 11.1 + 0.02 * k), lambda xy: torch.zeros(len(xy)), t=1.0 + k * DT, yaw=math.pi / 2))
    assert tuple(p.ap.target[0].tolist()) == (6, 6), "picked a new target inside the goal cell"
    assert int(p.pivot_state[0]) == 0, "turning back inside the goal cell"


def test_robot_outside_origin_and_target_redecides(monkeypatch):
    """Level 9 seed 50: a late turn north in (3, 0) braked from 1 m/s into (4, 0); from there the pilot aimed at the
    turn's target (3, 1), diagonally across the end of the wall between (3, 1) and (4, 1), and crashed. A robot in
    neither its decision cell nor its target must re-decide from where it is: the new target is a neighbour."""
    p = _pilot(monkeypatch)
    _close_wall(p, 2, 1, 2)  # (2, 1) | (1, 1): the diagonal is blocked by a wall, as in the maze
    _set_plan(p, origin=(1, 0), target=(1, 1), heading=1)
    p.act(_obs((4.25, 0.2), lambda xy: torch.zeros(len(xy)), t=1.0))  # Slid into (2, 0)
    ti, tj = (int(v) for v in p.ap.target[0])
    assert abs(ti - 2) + abs(tj) == 1, f"target ({ti}, {tj}) is not next to the robot's cell (2, 0)"
    assert tuple(p.ap.origin[0].tolist()) == (2, 0), "decision not made from the robot's cell"


def test_late_replan_turn_rotates_on_the_spot(monkeypatch):
    """Past the centre of (2, 0), walking east, the target (3, 0) turns out to be a dead end and the only way on
    is north: too late for a curve, so the 90 deg turn is done by stopping and rotating on the spot."""
    p = _pilot(monkeypatch)
    _set_plan(p, origin=(2, 0), target=(3, 0), heading=0)
    flat = lambda xy: torch.zeros(len(xy))
    p.act(_obs((4.3, 0.0), flat, t=1.0))  # Records the cost of the chosen target
    _close_wall(p, 3, 0, 0)  # (3, 0): closed east and north (south is the border) -> dead end
    _close_wall(p, 3, 0, 1)
    p.act(_obs((4.3, 0.0), flat, t=1.02))
    assert int(p.ap.heading[0]) == 1, "did not replan north"
    assert int(p.pivot_state[0]) > 0, "late 90 deg turn not done on the spot"


def test_spike_lane_lined_up_before_the_bed_when_far_off_lane(monkeypatch):
    """Heading south from (4, 5) into spike-lane cell (4, 4) (bed on the east side, safe lane x = 7.52) while
    0.9 m east of the lane (e.g. after sliding on ice): first line up with the lane well before the bed (y >= 9.3),
    not aim straight at the bed's entry corner."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 4, 4] = CELL_SPIKE_LANE
    p.cell_axis[0, 4, 4] = 1
    p.safe_offset[0, 4, 4] = -0.48  # Bed on +x, safe lane on -x
    _set_plan(p, origin=(4, 5), target=(4, 4), heading=3)
    xy = _grid(8.9, 9.9)
    wp, _, _ = p._compute_waypoint_and_speed(0, torch.tensor([8.9, 9.9]), -1.57, torch.tensor([4, 5]), xy, torch.zeros(len(xy)))
    assert abs(float(wp[0]) - 7.52) < 0.05 and float(wp[1]) >= 9.3, f"pre-align point {wp.tolist()} cuts toward the bed"
    assert bool(p.strafe[0]), "lines up by turning toward the point instead of stepping sideways"


def test_slalom_staging_steps_across_without_creeping_toward_the_bed(monkeypatch):
    """Forced short route, level 9 seed 1: off the ice heading east into corner (4, 4), the slalom (4, 5) north of it
    has its entry lane on the outside of the turn (x = 8.48) and bed 1 at x < 7.94 from y = 9.1. Staging must hold the
    robot near the corner cell's centre while it steps across (it used to creep to y = 8.55, 0.57 m before the bed,
    and the turn toward the lane put a foot on the bed's corner in 3 of 3 runs)."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 4, 5] = CELL_SPIKE_SLALOM
    p.cell_axis[0, 4, 5] = 1
    p.safe_offset[0, 4, 5, 0] = 0.48  # Entry half (y < 10): bed on -x, lane on +x
    p.safe_offset[0, 4, 5, 1] = -0.48  # Exit half: bed on +x, lane on -x
    _set_plan(p, origin=(4, 4), target=(4, 5), heading=1)
    pos = torch.tensor([7.09, 8.03])  # Just off the ice, 0.99 m on the bed's side of the slalom's centreline
    xy = _grid(7.09, 8.03)
    wp, _, _ = p._compute_waypoint_and_speed(0, pos, 0.0, torch.tensor([4, 4]), xy, torch.zeros(len(xy)))
    assert bool(p.strafe[0]), "not lining up sideways from the bed's side"
    assert abs(float(wp[0]) - 8.48) < 0.01, f"staging point {wp.tolist()} is not on the entry lane"
    assert abs(float(wp[1]) - (10.0 - STAGE_HOLD_U)) < 0.01, f"staging point {wp.tolist()} creeps toward the bed"
    assert 10.0 - STAGE_HOLD_U <= 8.15, "staging hold point not near the corner cell's centre"


def test_no_lane_staging_when_off_lane_on_the_wall_side(monkeypatch):
    """0.19 m off the safe lane but on the wall side (away from the bed) is safe: no sideways staging (on rough
    floor the small strafe stalled and triggered the unstuck reflex)."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 4, 4] = CELL_SPIKE_LANE
    p.cell_axis[0, 4, 4] = 1
    p.safe_offset[0, 4, 4] = -0.48  # Bed on +x, safe lane x = 7.52
    _set_plan(p, origin=(4, 5), target=(4, 4), heading=3)
    xy = _grid(7.33, 9.9)  # 0.19 m west of the lane = toward the west wall
    p._compute_waypoint_and_speed(0, torch.tensor([7.33, 9.9]), -1.57, torch.tensor([4, 5]), xy, torch.zeros(len(xy)))
    assert not bool(p.strafe[0]), "staging sideways although the robot is on the safe side of the lane"


def test_slalom_exit_waypoint_stays_ahead(monkeypatch):
    """In the slalom's exit lane near the cell border, the waypoint must stay well ahead of the robot (a fixed
    point at u = 0.95 collapsed onto the robot and made it spin)."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 2, 0] = CELL_SPIKE_SLALOM
    p.cell_axis[0, 2, 0] = 0
    p.safe_offset[0, 2, 0, 0] = -0.48
    p.safe_offset[0, 2, 0, 1] = 0.48
    _set_plan(p, origin=(2, 0), target=(3, 0), heading=0)
    pos = torch.tensor([4.93, 0.48])  # u = +0.93 in the exit lane
    xy = _grid(4.93, 0.48)
    wp, _, _ = p._compute_waypoint_and_speed(0, pos, 0.0, torch.tensor([2, 0]), xy, torch.zeros(len(xy)))
    assert float(torch.linalg.norm(wp - pos)) > 0.5, f"exit waypoint {wp.tolist()} is on top of the robot"


def test_stall_blames_target_only_when_it_is_the_obstacle(monkeypatch):
    """Level 9 seed 5: a snag on a box 0.6 m before a slalom cell put +15 on the slalom and sent the robot back to the
    long route. Only a ramp / trench the feet reach, the cell the robot is in, or a repeated stall blames the target."""
    p = _pilot(monkeypatch)
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)  # Target's near edge at x = 3.0
    p.stall_count[0] = 1
    here = torch.tensor([1, 0])

    def blames(x: float, cell=here) -> bool:
        return p._stall_blames_target(0, cell, torch.tensor([x, 0.0]))

    p.cell_type[0, 2, 0] = CELL_SPIKE_SLALOM
    assert not blames(2.4), "a snag 0.6 m before a slalom blamed the slalom (the level 9 seed 5 case)"
    assert not blames(2.8), "spike beds never block the robot physically"
    p.cell_type[0, 2, 0] = CELL_STEEP_RAMP
    assert not blames(2.4), "ramp blamed although the feet do not reach it"
    assert blames(2.6), "stalled at the foot of a steep ramp without blaming it"
    p.cell_type[0, 2, 0] = CELL_ROUGH
    assert blames(3.5, torch.tensor([2, 0])), "stalled inside the target cell without blaming it"
    p.stall_count[0] = 2
    assert blames(2.0), "a repeated stall in the same cell should blame the target"

    p.stall_count[0] = 1  # Walking west: target (0, 0), its near edge at x = 1.0
    _set_plan(p, origin=(1, 0), target=(0, 0), heading=2)
    p.cell_type[0, 0, 0] = CELL_STEEP_RAMP
    assert blames(1.4) and not blames(1.6), "edge distance wrong when walking west"


def _speed_at(p: Pilot, x: float, cell: tuple[int, int]) -> float:
    return _speed_at_xy(p, (x, 0.0), cell)


def _speed_at_xy(p: Pilot, pos_xy: tuple[float, float], cell: tuple[int, int]) -> float:
    xy = _grid(*pos_xy)
    pos, z = torch.tensor(pos_xy), torch.zeros(len(xy))
    _, speed, _ = p._compute_waypoint_and_speed(0, pos, 0.0, torch.tensor(cell), xy, z)
    return speed


def test_trench_crossed_at_trench_speed_then_next_cells_limit(monkeypatch):
    """Trench in cell (2, 0) (line at x = 4.0), ice after it: TRENCH_SPEED until the pelvis is TRENCH_CLEAR_U past the
    line, then the ice limit. Coming from rough floor, the rough limit holds until the trench cell."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 1, 0], p.cell_type[0, 2, 0], p.cell_type[0, 3, 0] = CELL_ROUGH, CELL_TRENCH, CELL_ICE
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    assert _speed_at(p, 2.6, (1, 0)) == 0.68, "rough floor before a trench should keep 0.68 (2 falls in 6 at 0.9)"
    p.cell_type[0, 0, 0] = CELL_PLAIN  # Toward the rough cell from plain floor: already 0.68, no 0.9 -> 0.68 -> 0.9
    _set_plan(p, origin=(0, 0), target=(1, 0), heading=0)
    assert _speed_at(p, 0.6, (0, 0)) == 0.68, "sped up toward a rough cell that leads straight into a trench"
    assert _speed_at(p, 3.7, (2, 0)) == TRENCH_SPEED, "not at trench speed before the trench line"
    _set_plan(p, origin=(2, 0), target=(3, 0), heading=0)
    assert _speed_at(p, 4.0 + TRENCH_CLEAR_U - 0.1, (2, 0)) == TRENCH_SPEED, "slowed for the ice before clearing"
    assert _speed_at(p, 4.0 + TRENCH_CLEAR_U + 0.1, (2, 0)) == 0.55, "ice limit not applied after the trench"


def test_trench_speed_both_directions_and_from_plain_floor(monkeypatch):
    """Walking west across the trench of cell (2, 0): the line is passed going -x. From plain floor the approach
    already uses TRENCH_SPEED; steep ramps keep their own STEEP_RAMP_SPEED limit."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 3, 0], p.cell_type[0, 2, 0], p.cell_type[0, 1, 0] = CELL_PLAIN, CELL_TRENCH, CELL_ROUGH
    _set_plan(p, origin=(3, 0), target=(2, 0), heading=2)
    assert _speed_at(p, 5.5, (3, 0)) == TRENCH_SPEED, "approach from plain floor not at trench speed"
    assert _speed_at(p, 4.3, (2, 0)) == TRENCH_SPEED, "not at trench speed before the line going west"
    _set_plan(p, origin=(2, 0), target=(1, 0), heading=2)
    rough = min(ROUGH_SPEED, p.fast_speed)
    assert _speed_at(p, 4.0 - TRENCH_CLEAR_U - 0.1, (2, 0)) == rough, "next cell's limit not applied going west"
    p.cell_type[0, 2, 0] = CELL_STEEP_RAMP
    assert _speed_at(p, 4.0, (2, 0)) == min(STEEP_RAMP_SPEED, p.fast_speed), "steep ramp limit changed"


def test_steep_ramp_climbed_fast_after_a_run_up(monkeypatch):
    """Steep ramp in cell (2, 0), slope foot at x = 3.0 going east. At 0.70 m/s about half the robots stalled on a
    0.30 m ramp (scripts/trench_test.py --task G1-Ramp-Test), at 0.9-1.1 none: reach STEEP_RAMP_SPEED in the last
    RAMP_RUNUP before the slope, also from rough floor, keep 0.70 before that and on ice (no speeding up on ice)."""
    p = _pilot(monkeypatch)
    fast = min(STEEP_RAMP_SPEED, p.fast_speed)
    p.cell_type[0, 1, 0], p.cell_type[0, 2, 0], p.cell_type[0, 3, 0] = CELL_ROUGH, CELL_STEEP_RAMP, CELL_PLAIN
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    assert _speed_at(p, 3.0 - RAMP_RUNUP - 0.1, (1, 0)) == 0.70, "sped up on rough floor before the run-up"
    assert _speed_at(p, 3.0 - RAMP_RUNUP + 0.1, (1, 0)) == fast, "no run-up on the rough floor before the ramp"
    _set_plan(p, origin=(2, 0), target=(3, 0), heading=0)
    assert _speed_at(p, 4.0, (2, 0)) == fast, "not at climbing speed on the ramp"
    p.cell_type[0, 1, 0] = CELL_ICE
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    assert _speed_at(p, 2.8, (1, 0)) == 0.70, "sped up on ice before the ramp"
    p.cell_type[0, 3, 0] = CELL_PLAIN  # Walking west: the slope foot is at x = 5.0
    _set_plan(p, origin=(3, 0), target=(2, 0), heading=2)
    assert _speed_at(p, 5.0 + RAMP_RUNUP - 0.1, (3, 0)) == fast, "no run-up walking west"
    assert _speed_at(p, 5.0 + RAMP_RUNUP + 0.1, (3, 0)) == 0.70, "run-up too early walking west"


def test_rough_floor_limit_is_its_own(monkeypatch):
    """Rough cells (in or into) get ROUGH_SPEED (scripts/rough_test.py: slower robots snag more); low ramps keep 0.68,
    and ice, before or after the rough cell, keeps its 0.55."""
    p = _pilot(monkeypatch)
    rough = min(ROUGH_SPEED, p.fast_speed)
    p.cell_type[0, 1, 0], p.cell_type[0, 2, 0], p.cell_type[0, 3, 0] = CELL_PLAIN, CELL_ROUGH, CELL_PLAIN
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    assert _speed_at(p, 2.6, (1, 0)) == rough, "approach to a rough cell not at the rough limit"
    _set_plan(p, origin=(2, 0), target=(3, 0), heading=0)
    assert _speed_at(p, 4.0, (2, 0)) == rough, "not at the rough limit inside the rough cell"
    p.cell_type[0, 3, 0] = CELL_LOW_RAMP
    assert _speed_at(p, 4.0, (2, 0)) == 0.68, "rough cell before a low ramp should keep the low-ramp limit"
    p.cell_type[0, 3, 0] = CELL_ICE  # Ice edge at x = 5.0
    assert _speed_at(p, 5.0 - ICE_SLOW_DIST - 0.1, (2, 0)) == rough, "slowed for the ice too early on the boxes"
    assert _speed_at(p, 5.0 - ICE_SLOW_DIST + 0.1, (2, 0)) == 0.55, "not at the ice limit just before the ice"
    p.cell_type[0, 2, 0] = CELL_PLAIN
    assert _speed_at(p, 4.0, (2, 0)) == 0.55, "plain floor before ice should keep the ice limit"
    p.cell_type[0, 1, 0], p.cell_type[0, 2, 0] = CELL_ICE, CELL_ROUGH
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    assert _speed_at(p, 2.6, (1, 0)) == 0.55, "sped up on ice before a rough cell"


def test_rough_corner_into_ice_keeps_rough_speed_until_the_ice(monkeypatch):
    """Level 7 seed 2: rough corner (3, 5) turning from north to east into ice (4, 5). Walked at the ice limit, the
    robot stumbled at the end of the turn, 0.56-0.67 m before the ice: the corner keeps ROUGH_SPEED (its curve speed
    limit still applies) and slows only within ICE_SLOW_DIST of the ice edge (x = 7.0, along the new heading)."""
    p = _pilot(monkeypatch)
    rough = min(ROUGH_SPEED, p.fast_speed)
    p.cell_type[0, 3, 5], p.cell_type[0, 4, 5] = CELL_ROUGH, CELL_ICE
    _set_plan(p, origin=(3, 5), target=(4, 5), heading=0)
    p.prev_heading[0] = 1  # Came in heading north
    assert _speed_at_xy(p, (5.95, 9.2), (3, 5)) == rough, "slowed for the ice at the start of the corner"
    assert _speed_at_xy(p, (6.4, 9.7), (3, 5)) == rough, "slowed where the robot stumbled (0.6 m before the ice)"
    assert _speed_at_xy(p, (6.75, 9.9), (3, 5)) == 0.55, "not at the ice limit 0.25 m before the ice"


def test_turn_on_ice_is_done_on_the_spot(monkeypatch):
    """Arriving at an ice corner, a 90 deg turn is done by stopping and rotating, not walked."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 2, 0] = CELL_ICE
    _close_wall(p, 2, 0, 0)  # Corner: east closed, north open
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    ice = lambda xy: torch.full((len(xy),), 0.01)
    p.act(_obs((3.7, 0.0), ice, t=1.0))
    assert int(p.ap.heading[0]) == 1, "setup: should turn north"
    assert int(p.pivot_state[0]) > 0, "90 deg turn on ice was walked instead of done on the spot"


def test_ice_corner_turn_starts_at_the_ice_edge(monkeypatch):
    """Just inside an ice corner cell (0.05 m past its border) with its walls known: decide and start braking now, so
    the robot rotates near the ice edge. Deciding 0.5 m from the centre rotated on the middle of the ice and 5% of those
    turns fell (scripts/ice_turn_test.py --task G1-Ice-Corner-Test: 0 of 1,000 when braking from the border)."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 2, 0] = CELL_ICE
    p.ap.walls[0, 2, 0] = torch.tensor([1, 0, 0, 1])  # East and south closed, north and west (back) open
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    ice = lambda xy: torch.full((len(xy),), 0.01)
    p.act(_obs((3.05, 0.0), ice, t=1.0))
    assert int(p.ap.heading[0]) == 1, "turn not decided on entering the ice corner"
    assert int(p.pivot_state[0]) > 0, "no turn on the spot started at the ice edge"

    # Also with a slalom after the corner: lining up from the inside of the turn holds back (STAGE_HOLD_U), while
    # waiting to turn 0.5 m from the centre rotated on the middle of the ice and fell (2 of 8 forced runs)
    p = _pilot(monkeypatch)
    p.cell_type[0, 2, 0], p.cell_type[0, 2, 1] = CELL_ICE, CELL_SPIKE_SLALOM
    p.ap.walls[0, 2, 0] = torch.tensor([1, 0, 0, 1])
    _set_plan(p, origin=(1, 0), target=(2, 0), heading=0)
    p.act(_obs((3.05, 0.0), ice, t=1.0))
    assert int(p.ap.heading[0]) == 1 and int(p.pivot_state[0]) > 0, "waited to turn on the middle of the ice"


def test_ice_turn_brakes_and_rotates_at_the_ice_limits(monkeypatch):
    """Turning on the spot on ice: brake at PIVOT_BRAKE_ACC_ICE and rotate at up to PIVOT_TURN_RATE_ICE, no faster."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 1, 0] = CELL_ICE
    _set_plan(p, origin=(1, 0), target=(1, 1), heading=1)
    ice = lambda xy: torch.full((len(xy),), 0.01)
    p.sim_time = 1.0  # Pivot starts at the first step's time (the brake phase times out after 1.5 s)
    p._start_pivot(0, 1)  # Facing east, turn north
    p.ap.cmd[0] = torch.tensor([0.55, 0.0, 0.0])  # Walking at the ice speed limit
    v_prev, w_peak = 0.55, 0.0
    for k in range(60):
        cmd = p.act(_obs((2.0, 0.0), ice, t=1.0 + k * DT, lin_vel=(v_prev, 0.0)))
        v, vy, w = (float(x) for x in cmd[0])
        if v_prev > 0.05:
            assert v_prev - v <= PIVOT_BRAKE_ACC_ICE * DT + 1e-5, f"braking faster than the ice limit ({v_prev} -> {v})"
            assert v_prev - v >= PIVOT_BRAKE_ACC_ICE * DT - 1e-5 or v == 0.0, f"braking slower than the ice limit ({v})"
        w_peak = max(w_peak, abs(w))
        v_prev = max(v, 0.0)  # The robot tracks the command
    assert int(p.pivot_state[0]) == 3, "never reached the rotate phase"
    assert abs(w_peak - PIVOT_TURN_RATE_ICE) < 1e-5, f"peak yaw rate on ice {w_peak:.2f} rad/s"


def test_no_lane_staging_from_the_centreline(monkeypatch):
    """Coming along the centreline toward a spike lane is the normal case: keep the usual diagonal pre-alignment
    (no sideways staging)."""
    p = _pilot(monkeypatch)
    p.cell_type[0, 4, 4] = CELL_SPIKE_LANE
    p.cell_axis[0, 4, 4] = 1
    p.safe_offset[0, 4, 4] = -0.48
    _set_plan(p, origin=(4, 5), target=(4, 4), heading=3)
    xy = _grid(8.0, 9.9)
    wp, _, _ = p._compute_waypoint_and_speed(0, torch.tensor([8.0, 9.9]), -1.57, torch.tensor([4, 5]), xy, torch.zeros(len(xy)))
    assert not bool(p.strafe[0]), "staging sideways from the centreline"
    assert abs(float(wp[1]) - 8.75) < 0.01, f"pre-align point {wp.tolist()} is not the usual u = -0.75 entry point"
