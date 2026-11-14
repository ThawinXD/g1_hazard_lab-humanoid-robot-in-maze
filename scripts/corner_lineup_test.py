"""Corner line-up test (headless): a spike lane right after a rough corner, walked like pilot.py.

Uses ``G1-Corner-Test`` (walled L corridors turning left / right at a cell of maze rough boxes, the maze's crash rule;
the cells before and after the corner cell are flat, as spike cells are). Each robot walks east into the rough corner
cell, where pilot.py decides the turn (on entering the cell). The cell after the corner is taken to be a spike lane:
its free lane is 0.48 m off the centreline, on the OUTSIDE of the turn (``CORNER_LANE=outside``, 98% of the line-ups
at rough corners in traced runs) or on the INSIDE (``CORNER_LANE=inside``). The robot walks on along the lane through
the whole bed, as pilot.py's Case 1 (normal tracking toward 0.65 m ahead on the lane line, <= 0.48 m/s), until 0.78 m
past the spike cell's centre. There are no cones; a SPIKE is challenge.py's rule: any body below the cone tips
(0.37 m) inside the bed, which reaches from the wall to 0-0.10 m short of the centreline (random per robot, as
hazard_maze.py). pilot.py's cone shield (cones within 0.38 m sideways and 0.65 m
ahead: sidestep away at 0.22 m/s, <= 0.38 m/s) is applied with the bed's cone positions.

``CORNER_RUNIN`` [m] makes the cell before the corner a spike lane too, its lane that far toward the inside of the
turn (0.48: level 7 seed 4's (3, 4) -> (4, 4) -> (4, 5)): the robot walks that lane at 0.48 m/s to 0.78 m past that
cell's centre, then normally (0.9 m/s) into the corner; that bed counts for SPIKE as well.

Maneuvers after the turn decision (``CORNER_MODES``), steered like pilot.py:

* ``strafe``: pilot.py's Case 2 approach: normal tracking to the entry point (u = -0.75 m, on the lane) at <= 0.5 m/s;
  while more than 0.15 m into the bed's half and short of u = -1.0, step sideways toward the lane at u = -1.9 m
  while turning to face down the new corridor (v_x <= 0.12, v_y <= 0.35, yaw rate <= 1.1). This is what pilot.py
  does for a lane on the INSIDE; with ``ROUGH_CORNER_CURVE`` on, an OUTSIDE lane is walked as ``curve09i20``.
* ``walk_on``: keep facing the way it came and walk on (normal tracking, <= 0.5 m/s) to the same 0.15 m line, then
  normal tracking to the entry point.
* ``walk_to_lane``: walk on until 0.30 m before the lane line, then normal tracking to the entry point.
* ``curve05`` / ``curve07`` / ``curve09``: walk the corner as pilot.py's pure-pursuit curve (1 m lookahead, speed from
  the 0.6 m/s^2 sideways-acceleration limit, no v_y), around a corner point moved to the lane line, then on along
  the lane; at most 0.5 / 0.7 / 0.9 m/s. ``curve09i20`` = corner point 0.20 m short of the lane line.

Per mode it prints the share of robots that got through the bed, hit a spike, fell or crashed (walls count), or were
still on the way; their lateral position where the bed starts (u = -0.70 m; measured toward the lane, lane centre =
0.48 m, bed edge at 0.0 to -0.10 m); how far they drifted into the bed's half before it; snags (progress < 0.1 m in
1 s); and the mean time from the corner cell's border to 0.78 m past the spike cell's centre.

Usage (from the project folder):
    PYTHONIOENCODING=utf-8 uv run python scripts/corner_lineup_test.py --checkpoint <model.pt>
Env vars: CORNER_MODES (default "strafe,walk_on,curve05,curve07,curve09"), CORNER_LANE (outside / inside, default
outside), CORNER_RUNIN (default 0), CORNER_TIME (trial [s], default 20), CORNER_REPEATS (trials per mode, default 2),
CORNER_SBEND (yaw-rate bias [rad/s] the wrong way in the last 1.2 m before the corner cell, as after an S-bend;
default 0), CORNER_SHIELD (0: without the cone shield).
"""

import os
import sys

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

import g1_maze_lab.tasks  # noqa: E402, F401  (registers G1-Rough-Test)
from g1_maze_lab.tasks.trench_test.terrain import rough_path  # noqa: E402

MODES = os.environ.get("CORNER_MODES", "strafe,walk_on,curve05,curve07,curve09").split(",")
LANE_SIDE = os.environ.get("CORNER_LANE", "outside")
RUNIN = float(os.environ.get("CORNER_RUNIN", 0.0))
TRIAL = float(os.environ.get("CORNER_TIME", 20.0))
REPEATS = int(os.environ.get("CORNER_REPEATS", 2))
MIN_SPEED, MAX_TURN = 0.25, 1.1  # pilot.py ap.min_speed [m/s], ap.max_turn [rad/s]
LIMITS = (2.0, 2.0, 2.8)  # pilot.py ap.max_acc (v_x, v_y) [m/s^2] and ap.max_ang_acc [rad/s^2]
ROUGH_SPEED, APPROACH_SPEED, LANE_SPEED = 0.9, 0.5, 0.48  # pilot.py rough floor, Case 2 pre-align, Case 1 lane [m/s]
LANE, STAGE_HOLD_U, ENTRY_U, BED_U = 0.48, 1.9, 0.75, 0.70  # pilot.py lane offset, staging hold, entry point, bed end
STAGE_BAND = 0.15  # pilot.py: staging while more than this into the bed's half [m]
THROUGH_U = 0.78  # pilot.py keeps the lane until this far past the spike cell's centre [m]
SPIKE_TOP = 0.37  # challenge.py: cone tips 0.35 m + 0.02
EDGE_MAX, CONE_PITCH, CONE_RAD, WALL = 0.10, 0.15, 0.45 * 0.15, 0.925  # Bed edge short of the centreline, cone grid [m]
SHIELD_SPEED, SHIELD_VY = 0.38, 0.22  # pilot.py cone shield
SHIELD = os.environ.get("CORNER_SHIELD", "1") != "0"  # 0: without pilot.py's cone shield
STALL_WINDOW, STALL_PROGRESS = 1.0, 0.1
SBEND = float(os.environ.get("CORNER_SBEND", 0.0))  # Yaw-rate bias [rad/s] the wrong way in the last 1.2 m before the
# corner cell: the robot arrives still turning out of a previous corner (an S-bend, level 9 seed 11)
WALL_NEAR = 0.70  # Pelvis this far off the centreline (wall face 0.925 m) = an arm near the wall [m]
CELLS = ("before the corner", "in the corner cell", "in the spike cell")  # Where falls / spike hits happened


def _find_env(step):
    for cell in step.__closure__ or ():
        obj = cell.cell_contents
        if hasattr(obj, "unwrapped") and hasattr(obj.unwrapped, "command_manager"):
            return obj.unwrapped
    raise RuntimeError("Could not find the environment in the play loop.")


def _yaw(q):
    x, y, z, w = q.unbind(-1)  # xyzw
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def _assign_corridors(env):
    terrain = env.scene.terrain
    rows, cols = terrain.terrain_origins.shape[:2]
    ids = torch.arange(env.num_envs, device=env.device)
    terrain.terrain_levels[:] = ids % rows
    terrain.terrain_types[:] = (ids // rows) % cols
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    return ids % rows


def _track(pos, yaw, wp_xy, vmax, vy_bias=0.0):
    """pilot.py normal tracking toward a waypoint: (v_x, v_y, w_z); ``vmax`` / ``vy_bias`` scalars or (n,)."""
    to = wp_xy - pos
    err = _wrap(torch.atan2(to[:, 1], to[:, 0]) - yaw)
    lat_b = -torch.sin(yaw) * to[:, 0] + torch.cos(yaw) * to[:, 1]
    v = MIN_SPEED + (vmax - MIN_SPEED) * torch.cos(err).clamp(min=0.0) ** 2
    v = torch.where(err.abs() > 0.7, torch.full_like(v, MIN_SPEED), v)
    lat = torch.where(err.abs() < 0.45, (0.45 * lat_b).clamp(-0.22, 0.22), torch.zeros_like(v))
    vy = (vy_bias + lat).clamp(-0.28, 0.28)
    return torch.stack((v, vy, (1.6 * err).clamp(-MAX_TURN, MAX_TURN)), -1)


def _pursuit(pos, pts, lookahead=1.0):
    """Point ``lookahead`` along the polyline ``pts`` (list of (n, 2)) past the point nearest to ``pos``."""
    best_d = torch.full((pos.shape[0],), 1e9, device=pos.device)
    best_s = torch.zeros_like(best_d)
    acc = torch.zeros_like(best_d)
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        seg = b - a
        length = seg.norm(dim=-1)
        t = (((pos - a) * seg).sum(-1) / (length * length)).clamp(0.0, 1.0)
        d = (a + t.unsqueeze(-1) * seg - pos).norm(dim=-1)
        closer = d < best_d
        best_d = torch.where(closer, d, best_d)
        best_s = torch.where(closer, acc + t * length, best_s)
        acc = acc + length
    s = best_s + lookahead
    out = pts[-1].clone()
    acc = torch.zeros_like(best_d)
    done = torch.zeros_like(best_d, dtype=torch.bool)
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        length = (b - a).norm(dim=-1)
        here = ~done & (s <= acc + length)
        frac = ((s - acc) / length).clamp(0.0, 1.0).unsqueeze(-1)
        out = torch.where(here.unsqueeze(-1), a + (b - a) * frac, out)
        done |= here
        acc = acc + length
    return out


def _cap(v, shield):
    """Speed limit ``v`` per robot, cut to the cone shield's while it is on (n,)."""
    return torch.where(shield, min(v, SHIELD_SPEED), v)


def _curve(pos, yaw, wp_xy, vmax, vy_bias=0.0):
    """pilot.py curve tracking (pure pursuit): speed from the sideways-acceleration limit, v_y only the shield's."""
    to = wp_xy - pos
    err = _wrap(torch.atan2(to[:, 1], to[:, 0]) - yaw)
    kappa = 2.0 * torch.sin(err).abs() / to.norm(dim=-1).clamp(min=0.3)
    v = torch.minimum(torch.sqrt(0.6 / kappa.clamp(min=1e-3)), torch.as_tensor(vmax, device=pos.device))
    vy = torch.as_tensor(vy_bias, device=pos.device).expand_as(v).clamp(-0.28, 0.28)
    out = torch.stack((v.clamp(min=MIN_SPEED), vy, (1.6 * err).clamp(-MAX_TURN, MAX_TURN)), -1)
    return torch.where((err.abs() < 1.2).unsqueeze(-1), out, _track(pos, yaw, wp_xy, vmax, vy_bias))


def _strafe(pos, yaw, wp_xy, heading):
    """pilot.py staging / slalom strafe: face ``heading``, step sideways toward the waypoint."""
    to = wp_xy - pos
    c, s = torch.cos(yaw), torch.sin(yaw)
    fwd_b, lat_b = c * to[:, 0] + s * to[:, 1], -s * to[:, 0] + c * to[:, 1]
    head = _wrap(heading - yaw)
    return torch.stack(
        ((0.8 * fwd_b).clamp(-0.15, 0.12), (1.2 * lat_b).clamp(-0.35, 0.35), (1.6 * head).clamp(-MAX_TURN, MAX_TURN)),
        -1,
    )


def _cones(centre, along, across, side, edge):
    """Cone centres (n, K, 2) of a bed: ``along`` / ``across`` unit vectors (n, 2), bed on the ``-side`` half (side
    (n,) = +-1 toward the lane), from ``edge`` (n,) short of the centreline to the wall; and which exist (n, K)."""
    u = torch.arange(-BED_U + CONE_RAD, BED_U - CONE_RAD + 1e-6, CONE_PITCH, device=centre.device)
    k = torch.arange(int(WALL // CONE_PITCH) + 1, device=centre.device)
    v = -(edge.unsqueeze(1) + CONE_RAD + CONE_PITCH * k)  # (n, nk), toward the bed
    ok = (v > -(WALL - CONE_RAD)).unsqueeze(1).expand(-1, len(u), -1)
    uu = u.view(1, -1, 1).expand(v.shape[0], -1, v.shape[1])
    vv = v.unsqueeze(1).expand(-1, len(u), -1) * side.view(-1, 1, 1)
    pts = centre.view(-1, 1, 1, 2) + uu.unsqueeze(-1) * along.view(-1, 1, 1, 2)
    pts = pts + vv.unsqueeze(-1) * across.view(-1, 1, 1, 2)
    return pts.flatten(1, 2), ok.flatten(1, 2)


def _shield(pos, yaw, cones, ok):
    """pilot.py cone shield: (active (n,), v_y bias (n,)) from cones within 0.38 m sideways and 0.65 m ahead."""
    rel = cones - pos.unsqueeze(1)
    c, s = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
    rx, ry = rel[..., 0] * c + rel[..., 1] * s, -rel[..., 0] * s + rel[..., 1] * c
    near = ok & (rx > -0.15) & (rx < 0.65) & (ry.abs() < 0.38)
    count = near.sum(1)
    mean_ry = (ry * near).sum(1) / count.clamp(min=1)
    return count >= 2, torch.where(mean_ry > 0.0, -SHIELD_VY, SHIELD_VY)


def _in_bed(points, centre, along, across, side, edge, z):
    """challenge.py spike rule for one bed: any of ``points`` (n, B, 2) inside it below the cone tips -> (n,)."""
    rel = points - centre.unsqueeze(1)
    u = (rel * along.unsqueeze(1)).sum(-1)
    v = (rel * across.unsqueeze(1)).sum(-1) * side.unsqueeze(1)
    return ((u > -BED_U) & (u < BED_U) & (v < -edge.unsqueeze(1)) & (z < SPIKE_TOP)).any(1)


def _trial(env, step, dt, geo, used, mode):
    w0, c, u1, u2 = geo
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("base_velocity")
    n, dev = env.num_envs, env.device
    env.reset()
    origins = env.scene.env_origins
    heading0 = torch.atan2(u1[:, 1], u1[:, 0])
    pose = robot.data.default_root_pose.torch.clone()
    pose[:, :2] = origins[:, :2] + RUNIN * u2  # On the run-in lane line (0 = the centreline)
    pose[:, 2] += origins[:, 2]
    zero = torch.zeros_like(heading0)
    pose[:, 3:7] = torch.stack((zero, zero, torch.sin(0.5 * heading0), torch.cos(0.5 * heading0)), -1)
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))

    spike = c + 2.0 * u2  # Centre of the (pretend) spike cell after the corner
    side = torch.full((n,), 1.0 if LANE_SIDE == "outside" else -1.0, device=dev)  # Lane along +u1 (outside) or -u1
    turn_sign = torch.sign(u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0])  # +1 left turn, -1 right
    heading2 = torch.atan2(u2[:, 1], u2[:, 0])
    entry = spike - ENTRY_U * u2 + side.unsqueeze(1) * LANE * u1
    edge = torch.rand(n, device=dev) * EDGE_MAX  # Bed edge short of the centreline
    cones, cone_ok = _cones(spike, u2, u1, side, edge)
    runin_c = c - 2.0 * u1  # Centre of the run-in spike cell (the cell before the corner)
    r_side = torch.full((n,), 1.0 if RUNIN >= 0.0 else -1.0, device=dev)  # Run-in lane along +u2 (inside) or -u2
    edge2 = torch.rand(n, device=dev) * EDGE_MAX
    if RUNIN:
        cones2, ok2 = _cones(runin_c, u1, u2, r_side, edge2)
        cones, cone_ok = torch.cat((cones, cones2), 1), torch.cat((cone_ok, ok2), 1)

    command = torch.zeros(n, 3, device=dev)
    lim = torch.tensor(LIMITS, device=dev) * dt
    active = used.clone()
    outcome = torch.full((n,), 2, dtype=torch.long, device=dev)  # 0 through, 1 fell / crashed, 2 on the way, 3 spike
    lat_at = {u: torch.full((n,), float("nan"), device=dev) for u in (1.2, 0.9, BED_U)}  # Toward the lane, at u
    v_max = torch.full((n,), -9.0, device=dev)  # Farthest toward the lane-side wall before the bed
    v_min = torch.full((n,), 9.0, device=dev)  # Farthest into the bed's half before the bed
    t_border = torch.full((n,), float("nan"), device=dev)
    t_done = torch.full((n,), float("nan"), device=dev)
    decided = torch.zeros(n, dtype=torch.bool, device=dev)
    walking_on = torch.zeros(n, dtype=torch.bool, device=dev)
    window = max(int(round(STALL_WINDOW / dt)), 1)
    hist = torch.zeros(window, n, 2, device=dev)
    stalled = torch.zeros(n, dtype=torch.bool, device=dev)
    snags = torch.zeros(n, device=dev)
    end_uv = torch.full((n, 6), float("nan"), device=dev)  # (u, v toward the lane, bed 1 after / 2 before the corner,
    # along the first corridor from the corner cell's centre, shield on for [s], turn +1 left / -1 right) where a robot
    # fell / hit a spike
    shield_t = torch.zeros(n, device=dev)
    wall_from = -9.0 if LANE_SIDE == "outside" else -2.0 + WALL  # Inner wall of the new corridor starts past the corner
    for k in range(int(TRIAL / dt)):
        pos, yaw = robot.data.root_pos_w.torch[:, :2], _yaw(robot.data.root_quat_w.torch)
        rel = pos - spike
        u_now, v_now = (rel * u2).sum(-1), (rel * u1).sum(-1)  # Along the new corridor / across it (+ = outside)
        v_lane = v_now * side  # Across it, toward the lane
        newly = ~decided & (v_now >= -1.0)  # Entered the corner cell: pilot.py decides the turn there
        t_border = torch.where(newly, torch.full_like(t_border, k * dt), t_border)
        decided |= newly
        in_runin = ~decided & (v_now < -2.0 + THROUGH_U) if RUNIN else torch.zeros_like(decided)  # Run-in lane
        in_spike = decided & (u_now > -1.0)  # In the spike cell: pilot.py Case 1
        staging = (v_lane < -STAGE_BAND) & (u_now < -1.0)
        shield, vy_b = _shield(pos, yaw, cones, cone_ok)
        shield &= (decided | in_runin) & SHIELD  # pilot.py: the current or the target cell is a spike cell
        shield_t = torch.where(shield, shield_t + dt, torch.zeros_like(shield_t))  # How long it has been on
        vy_b = torch.where(shield, vy_b, torch.zeros_like(vy_b))

        target = _track(pos, yaw, c, ROUGH_SPEED)  # First leg: aim at the corner cell's centre
        if RUNIN:  # pilot.py: Case 2 toward the lane's entry point, then Case 1 in the cell (0.65 m ahead, the border)
            u_r = v_now + 2.0  # Along the run-in spike cell from its centre
            in_cell = u_r > -1.0
            aim = torch.where(in_cell, (u_r + 0.65).clamp(max=0.90), torch.full_like(u_r, -ENTRY_U))
            vmax = torch.where(in_cell, _cap(LANE_SPEED, shield), _cap(APPROACH_SPEED, shield))
            runin_wp = runin_c + aim.unsqueeze(-1) * u1 + RUNIN * u2
            target = torch.where(in_runin.unsqueeze(-1), _track(pos, yaw, runin_wp, vmax, vy_b), target)
        if SBEND:
            late = ~decided & (v_now > -2.2)
            biased = (target[:, 2] - SBEND * turn_sign).clamp(-MAX_TURN, MAX_TURN)
            target[:, 2] = torch.where(late, biased, target[:, 2])
        lane_v = side.unsqueeze(1) * LANE * u1
        if mode == "strafe":
            hold = spike - STAGE_HOLD_U * u2 + lane_v
            t_st = _strafe(pos, yaw, hold, heading2)
            t_go = _track(pos, yaw, entry, _cap(APPROACH_SPEED, shield), vy_b)
            after = torch.where(staging.unsqueeze(-1), t_st, t_go)
        elif mode.startswith("curve"):
            rest = mode[5:]
            vmax = float(rest[:2]) / 10.0
            inset = float(rest.split("i")[1]) / 100.0 if "i" in rest else 0.0  # Corner point short of the lane line [m]
            path = [w0, c + side.unsqueeze(1) * (LANE - inset) * u1, entry, spike + 2.0 * u2 + lane_v]
            after = _curve(pos, yaw, _pursuit(pos, path), _cap(vmax, shield), vy_b)
        else:
            stop = -STAGE_BAND if mode == "walk_on" else LANE - 0.30
            walking_on = (walking_on | (decided & staging)) & (v_lane < stop)
            ahead = spike + u_now.clamp(-2.4, -1.6).unsqueeze(-1) * u2 + lane_v  # On the lane line, straight ahead
            t_on = _track(pos, yaw, ahead, _cap(APPROACH_SPEED, shield), vy_b)
            t_go = _track(pos, yaw, entry, _cap(APPROACH_SPEED, shield), vy_b)
            after = torch.where(walking_on.unsqueeze(-1), t_on, t_go)
        lane_wp = spike + (u_now + 0.65).clamp(max=0.90).unsqueeze(-1) * u2 + lane_v
        after = torch.where(in_spike.unsqueeze(-1), _track(pos, yaw, lane_wp, _cap(LANE_SPEED, shield), vy_b), after)
        target = torch.where(decided.unsqueeze(-1), after, target)
        target[~active] = 0.0
        command += (target - command).clamp(-lim, lim)
        cmd.time_left[:] = 1.0e9
        cmd.vel_command_b[:] = command
        with torch.inference_mode():
            step()

        done = active & env.termination_manager.terminated  # Already reset to the spawn point: last position before
        end_uv[done] = torch.stack((u_now, v_lane, torch.zeros_like(u_now), v_now, shield_t, turn_sign), -1)[done]
        outcome[done] = 1
        active &= ~done
        pos = robot.data.root_pos_w.torch[:, :2]
        rel = pos - spike
        u_now, v_now = (rel * u2).sum(-1), (rel * u1).sum(-1)
        v_lane = v_now * side
        bodies = robot.data.body_pos_w.torch
        z = bodies[..., 2] - origins[:, 2:3]
        hit1 = _in_bed(bodies[..., :2], spike, u2, u1, side, edge, z)
        hit2 = _in_bed(bodies[..., :2], runin_c, u1, u2, r_side, edge2, z) if RUNIN else torch.zeros_like(hit1)
        hit = active & (hit1 | hit2)
        end_uv[hit] = torch.stack((u_now, v_lane, torch.where(hit1, 1.0, 2.0), v_now, shield_t, turn_sign), -1)[hit]
        outcome[hit] = 3
        active &= ~hit
        for uu, arr in lat_at.items():
            cross = active & (u_now > -uu) & arr.isnan()
            arr[cross] = v_lane[cross]
        before_bed = active & decided & (u_now < -BED_U)
        lane_wall = before_bed & (u_now > wall_from)  # Where the lane-side wall stands (inside lane: past the corner)
        v_max = torch.where(lane_wall, torch.maximum(v_max, v_lane), v_max)
        v_min = torch.where(before_bed, torch.minimum(v_min, v_lane), v_min)
        through = active & (u_now > THROUGH_U)
        outcome[through] = 0
        t_done[through] = (k + 1) * dt - t_border[through]
        active &= ~through
        moved = (pos - hist[k % window]).norm(dim=-1)
        now_stalled = active & decided & (k >= window) & (moved < STALL_PROGRESS)
        snags += (now_stalled & ~stalled).float()
        stalled = now_stalled
        hist[k % window] = pos
    return outcome, lat_at[BED_U], snags, t_done, end_uv, lat_at[1.2], lat_at[0.9], v_max, v_min


def _pct(k, n):
    return f"{100.0 * k / n:5.1f}%" if n else "    -"


def _quantiles(a):
    q = torch.quantile(a, torch.tensor([0.0, 0.01, 0.05, 0.5], device=a.device)).tolist()
    return f"min {q[0]:+.2f}  p1 {q[1]:+.2f}  p5 {q[2]:+.2f}  median {q[3]:+.2f}  max {float(a.max()):+.2f}"


def corner_playback(step, *, dt, args_cli, env_cfg):
    env = _find_env(step)
    cmd = env.command_manager.get_term("base_velocity")
    cmd.cfg.heading_command = False
    cmd.is_heading_env[:] = False
    cmd.is_standing_env[:] = False
    gen = env_cfg.scene.terrain.terrain_generator
    sub = next(iter(gen.sub_terrains.values()))
    sub.size = tuple(gen.size)
    kinds = sub.kinds
    kind_idx = _assign_corridors(env)
    dev = env.device
    table = []
    for kind in kinds:
        p = rough_path(kind, sub)
        if len(p) == 2:
            p = [p[0], p[1], (p[1][0] + 1.0, p[1][1])]  # Unused (straight rows stand still)
        table.append(p)
    pts = torch.tensor(table, device=dev)[kind_idx] + env.scene.env_origins[:, :2].unsqueeze(1)
    w0, c, w2 = pts.unbind(1)
    u1 = (c - w0) / (c - w0).norm(dim=-1, keepdim=True)
    u2 = (w2 - c) / (w2 - c).norm(dim=-1, keepdim=True)
    used = torch.tensor([k in ("left", "right") for k in kinds], device=dev)[kind_idx]
    print(
        f"[CORNER] {int(used.sum())} robots in L corners, lane on the {LANE_SIDE} of the turn, "
        f"run-in lane {RUNIN:+.2f} m, modes {MODES}, {REPEATS} trial(s) of {TRIAL:.0f} s",
        flush=True,
    )
    print(
        f"{'mode':>13}{'n':>6}{'through':>9}{'SPIKE':>8}{'fell/crash':>11}{'on way':>8}{'lat mean':>10}{'lat min':>9}"
        f"{'< 0.25 m':>10}{'snagged':>9}{'t (s)':>8}",
        flush=True,
    )
    for mode in MODES:
        runs = [_trial(env, step, dt, (w0, c, u1, u2), used, mode) for _ in range(REPEATS)]
        outcome, lat, snags, t_done, end_uv, lat12, lat09, v_max, v_min = (
            torch.cat([r[i] for r in runs]) for i in range(9)
        )
        sel = used.repeat(REPEATS)
        n = int(sel.sum())
        o = outcome[sel]
        reached = sel & ~lat.isnan()  # Got to the bed's start
        lat_ok = lat[reached]
        nan = float("nan")
        lat_mean = float(lat_ok.mean()) if len(lat_ok) else nan
        lat_min = float(lat_ok.min()) if len(lat_ok) else nan
        ok = sel & (outcome == 0)
        t_mean = float(t_done[ok].mean()) if bool(ok.any()) else nan
        print(
            f"{mode:>13}{n:>6}{_pct(int((o == 0).sum()), n):>9}{_pct(int((o == 3).sum()), n):>8}"
            f"{_pct(int((o == 1).sum()), n):>11}{_pct(int((o == 2).sum()), n):>8}{lat_mean:>10.2f}{lat_min:>9.2f}"
            f"{_pct(int((lat_ok < 0.25).sum()), len(lat_ok)):>10}{_pct(int((snags[sel] > 0).sum()), n):>9}"
            f"{t_mean:>8.2f}",
            flush=True,
        )
        for name, arr in (("u=-1.2", lat12), ("u=-0.9", lat09), ("u=-0.7", lat)):
            a = arr[sel & ~arr.isnan()]
            if len(a):
                print(f"{'':15}toward the lane at {name}: {_quantiles(a)}", flush=True)
        dec = sel & (v_min < 9.0)
        if bool(dec.any()):
            vm, vn = v_max[dec], v_min[dec]
            near = _pct(int((vm > WALL_NEAR).sum()), len(vm))
            print(f"{'':15}before the bed, deepest into the bed's half: {_quantiles(vn)}", flush=True)
            print(
                f"{'':15}before the bed, farthest toward the lane-side wall: max {float(vm.max()):.2f}, over "
                f"{WALL_NEAR} m {near}",
                flush=True,
            )
        for code, name in ((3, "spike hits"), (1, "falls")):
            e = end_uv[sel & (outcome == code)]
            if len(e):
                where = torch.where(e[:, 0] >= -1.0, 2, torch.where(e[:, 3] >= -1.0, 1, 0))
                counts = ", ".join(f"{cell} {int((where == i).sum())}" for i, cell in enumerate(CELLS))
                spots = ", ".join(
                    f"({a:+.2f}, {b:+.2f}{', run-in bed' if int(w) == 2 else ''})" for a, b, w, *_ in e[:12].tolist()
                )
                shielded = int((e[:, 4] > 0.0).sum())
                print(
                    f"{'':15}{name} {len(e)}: {counts}; left / right turns {int((e[:, 5] > 0).sum())} / "
                    f"{int((e[:, 5] < 0).sum())}; cone shield on at that moment: {shielded}; at (u, v toward the "
                    f"lane): {spots}",
                    flush=True,
                )


play_rsl_rl.run_playback = corner_playback

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--task" not in argv:
        argv = ["--task", "G1-Corner-Test"] + argv
    if "--num_envs" not in argv:
        argv += ["--num_envs", "500"]
    play_rsl_rl.run(argv)
