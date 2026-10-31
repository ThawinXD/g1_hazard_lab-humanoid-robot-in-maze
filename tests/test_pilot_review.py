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

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pilot import CELL_LOW_RAMP, CELL_SPIKE_SLALOM, CELL_STEEP_RAMP, Pilot  # noqa: E402
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


def _obs(pos, z_fn, t, yaw=0.0) -> dict:
    xy = _grid(*pos)
    return {
        "pos_xy": torch.tensor([pos], dtype=torch.float32),
        "yaw": torch.tensor([yaw], dtype=torch.float32),
        "t": t,
        "dt": DT,
        "scan_xy": xy.unsqueeze(0),
        "scan_z": z_fn(xy).unsqueeze(0),
    }


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
    and turning back west costs 13.8, so it goes east. After 3.5 s stuck, east should cost 26.8 and lose to west.
    """
    p = _pilot(monkeypatch)
    _close_wall(p, 1, 0, 1)
    pos = (2.0, 0.0)
    p.pos_anchor[0] = torch.tensor(pos)  # robot has not moved since t = 0

    p.act(_obs(pos, lambda xy: _rough(xy, 2, 0), t=1.0))
    assert int(p.ap.heading[0]) == 0, "setup: without a penalty the robot should head east"

    p.act(_obs(pos, lambda xy: _rough(xy, 2, 0), t=10.0))  # stuck since t = 0 -> reflex + penalty on (2, 0)
    assert float(p.unstuck_until[0]) > 10.0, "setup: the unstuck reflex should have fired"

    p.act(_obs(pos, lambda xy: _rough(xy, 2, 0), t=10.0 + DT))  # replans with the penalty
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
