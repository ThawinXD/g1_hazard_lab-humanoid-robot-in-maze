"""Corner line-up test (headless): lining up for a spike lane right after a rough corner, three ways.

Uses ``G1-Rough-Test`` (walled corridors of maze rough cells, the maze's crash rule): only the ``left`` / ``right``
L corridors are used (the other robots stand still). Each robot walks east into the rough corner cell, where pilot.py
decides the turn (on entering the cell). The cell after the corner is taken to be a spike lane whose free lane is on
the OUTSIDE of the turn, 0.48 m off the centreline (98% of the lining-up episodes at rough corners in traced runs),
so the robot, coming in on the corridor's centreline, is in the bed's half and must move 0.48-0.63 m across.

Maneuvers (``CORNER_MODES``), steered like pilot.py:

* ``strafe``: pilot.py's staging: step sideways toward the lane at u = -1.9 m before the spike cell's centre while
  turning to face down the new corridor (v_x <= 0.12, v_y <= 0.35, yaw rate <= 1.1), until it is within 0.15 m of the
  centreline on the bed's side; then normal tracking to the entry point (u = -0.75 m, in the lane) at <= 0.5 m/s.
* ``walk_on``: keep facing the way it came and walk on (normal tracking, <= 0.5 m/s) to the same 0.15 m line, then
  normal tracking to the entry point.
* ``walk_to_lane``: walk on until 0.30 m before the lane line, then normal tracking to the entry point.
* ``curve05`` / ``curve07`` / ``curve09``: walk the corner as pilot.py's pure-pursuit curve (1 m lookahead, speed from
  the 0.6 m/s^2 sideways-acceleration limit, no v_y), around a corner point moved out to the lane line, then on along
  the lane; at most 0.5 / 0.7 / 0.9 m/s.

Per mode it prints the share of robots that reached the bed's start (u = -0.70 m), fell or crashed (walls count), or
were still on the way; their lateral position there (lane centre = 0.48 m; the bed reaches to the centreline, so
under ~0.25 m a leg is over it); snags (progress < 0.1 m in 1 s); and the mean time from the corner cell's border.

Usage (from the project folder):
    PYTHONIOENCODING=utf-8 uv run python scripts/corner_lineup_test.py --checkpoint <model.pt>
Env vars: CORNER_MODES (default "strafe,walk_on,curve05,curve07,curve09"; ``curve09i20`` = corner point 0.20 m inside
the lane line), CORNER_TIME (trial [s], default 16), CORNER_REPEATS (trials per mode, default 2), CORNER_SBEND (yaw-rate
bias [rad/s] the wrong way in the last 1.2 m before the corner cell, as after an S-bend; default 0).
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
TRIAL = float(os.environ.get("CORNER_TIME", 16.0))
REPEATS = int(os.environ.get("CORNER_REPEATS", 2))
MIN_SPEED, MAX_TURN = 0.25, 1.1  # pilot.py ap.min_speed [m/s], ap.max_turn [rad/s]
LIMITS = (2.0, 2.0, 2.8)  # pilot.py ap.max_acc (v_x, v_y) [m/s^2] and ap.max_ang_acc [rad/s^2]
ROUGH_SPEED, APPROACH_SPEED = 0.9, 0.5  # pilot.py rough-floor limit; pre-align (Case 2) limit [m/s]
LANE, STAGE_HOLD_U, ENTRY_U, BED_U = 0.48, 1.9, 0.75, 0.70  # pilot.py lane offset, staging hold, entry point, bed start
STAGE_BAND = 0.15  # pilot.py: staging while more than this into the bed's half [m]
STALL_WINDOW, STALL_PROGRESS = 1.0, 0.1
SBEND = float(os.environ.get("CORNER_SBEND", 0.0))  # Yaw-rate bias [rad/s] the wrong way in the last 1.2 m before the
# corner cell: the robot arrives still turning out of a previous corner (an S-bend, level 9 seed 11)
WALL_NEAR = 0.70  # Pelvis this far off the centreline (wall face 0.925 m) = an arm near the outer wall [m]


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


def _track(pos, yaw, wp_xy, vmax):
    """pilot.py normal tracking toward a waypoint: (v_x, v_y, w_z)."""
    to = wp_xy - pos
    err = _wrap(torch.atan2(to[:, 1], to[:, 0]) - yaw)
    lat_b = -torch.sin(yaw) * to[:, 0] + torch.cos(yaw) * to[:, 1]
    v = MIN_SPEED + (vmax - MIN_SPEED) * torch.cos(err).clamp(min=0.0) ** 2
    v = torch.where(err.abs() > 0.7, torch.full_like(v, MIN_SPEED), v)
    vy = torch.where(err.abs() < 0.45, (0.45 * lat_b).clamp(-0.22, 0.22), torch.zeros_like(v)).clamp(-0.28, 0.28)
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


def _curve(pos, yaw, wp_xy, vmax):
    """pilot.py curve tracking (pure pursuit): speed from the sideways-acceleration limit, no v_y."""
    to = wp_xy - pos
    err = _wrap(torch.atan2(to[:, 1], to[:, 0]) - yaw)
    kappa = 2.0 * torch.sin(err).abs() / to.norm(dim=-1).clamp(min=0.3)
    v = torch.sqrt(0.6 / kappa.clamp(min=1e-3)).clamp(max=vmax).clamp(min=MIN_SPEED)
    out = torch.stack((v, torch.zeros_like(v), (1.6 * err).clamp(-MAX_TURN, MAX_TURN)), -1)
    return torch.where((err.abs() < 1.2).unsqueeze(-1), out, _track(pos, yaw, wp_xy, vmax))


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


def _trial(env, step, dt, geo, used, mode):
    w0, c, u1, u2 = geo
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("base_velocity")
    n, dev = env.num_envs, env.device
    env.reset()
    origins = env.scene.env_origins
    heading0 = torch.atan2(u1[:, 1], u1[:, 0])
    pose = robot.data.default_root_pose.torch.clone()
    pose[:, :2] = origins[:, :2]
    pose[:, 2] += origins[:, 2]
    zero = torch.zeros_like(heading0)
    pose[:, 3:7] = torch.stack((zero, zero, torch.sin(0.5 * heading0), torch.cos(0.5 * heading0)), -1)
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))

    spike = c + 2.0 * u2  # Centre of the (pretend) spike cell after the corner
    turn_sign = torch.sign(u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0])  # +1 left turn, -1 right
    v_max = torch.full((n,), -9.0, device=dev)  # Farthest out toward the outer wall before the bed
    heading2 = torch.atan2(u2[:, 1], u2[:, 0])
    entry = spike - ENTRY_U * u2 + LANE * u1
    command = torch.zeros(n, 3, device=dev)
    lim = torch.tensor(LIMITS, device=dev) * dt
    active = used.clone()
    outcome = torch.full((n,), 2, dtype=torch.long, device=dev)
    lat_at_bed = torch.full((n,), float("nan"), device=dev)
    lat_at = {u: torch.full((n,), float("nan"), device=dev) for u in (1.2, 0.9)}  # Lateral where the pelvis passes u
    t_border = torch.full((n,), float("nan"), device=dev)
    t_bed = torch.full((n,), float("nan"), device=dev)
    decided = torch.zeros(n, dtype=torch.bool, device=dev)
    walking_on = torch.zeros(n, dtype=torch.bool, device=dev)
    window = max(int(round(STALL_WINDOW / dt)), 1)
    hist = torch.zeros(window, n, 2, device=dev)
    stalled = torch.zeros(n, dtype=torch.bool, device=dev)
    snags = torch.zeros(n, device=dev)
    fall_uv = torch.full((n, 2), float("nan"), device=dev)
    for k in range(int(TRIAL / dt)):
        pos, yaw = robot.data.root_pos_w.torch[:, :2], _yaw(robot.data.root_quat_w.torch)
        rel = pos - spike
        u_now, v_now = (rel * u2).sum(-1), (rel * u1).sum(-1)  # Along the new corridor / across it (+ = outside)
        newly = ~decided & (v_now >= -1.0)  # Entered the corner cell: pilot.py decides the turn there
        t_border = torch.where(newly, torch.full_like(t_border, k * dt), t_border)
        decided |= newly
        staging = (v_now < -STAGE_BAND) & (u_now < -1.0)
        target = _track(pos, yaw, c, ROUGH_SPEED)  # First leg: aim at the corner cell's centre
        if SBEND:
            late = ~decided & (v_now > -2.2)
            biased = (target[:, 2] - SBEND * turn_sign).clamp(-MAX_TURN, MAX_TURN)
            target[:, 2] = torch.where(late, biased, target[:, 2])
        if mode == "strafe":
            hold = spike - STAGE_HOLD_U * u2 + LANE * u1
            t_st = _strafe(pos, yaw, hold, heading2)
            t_go = _track(pos, yaw, entry, APPROACH_SPEED)
            after = torch.where(staging.unsqueeze(-1), t_st, t_go)
        elif mode.startswith("curve"):
            rest = mode[5:]
            vmax = float(rest[:2]) / 10.0
            inset = float(rest.split("i")[1]) / 100.0 if "i" in rest else 0.0  # Corner point inside the lane line [m]
            path = [w0, c + (LANE - inset) * u1, entry, spike + 2.0 * u2 + LANE * u1]
            after = _curve(pos, yaw, _pursuit(pos, path), vmax)
        else:
            stop = -STAGE_BAND if mode == "walk_on" else LANE - 0.30
            walking_on = (walking_on | (decided & staging)) & (v_now < stop)
            ahead = spike + u_now.clamp(-2.4, -1.6).unsqueeze(-1) * u2 + LANE * u1  # On the lane line, straight ahead
            t_on = _track(pos, yaw, ahead, APPROACH_SPEED)
            t_go = _track(pos, yaw, entry, APPROACH_SPEED)
            after = torch.where(walking_on.unsqueeze(-1), t_on, t_go)
        target = torch.where(decided.unsqueeze(-1), after, target)
        target[~active] = 0.0
        command += (target - command).clamp(-lim, lim)
        cmd.time_left[:] = 1.0e9
        cmd.vel_command_b[:] = command
        with torch.inference_mode():
            step()

        done = active & env.termination_manager.terminated
        fall_uv[done] = torch.stack((u_now, v_now), -1)[done]
        outcome[done] = 1
        active &= ~done
        pos = robot.data.root_pos_w.torch[:, :2]
        rel = pos - spike
        u_now, v_now = (rel * u2).sum(-1), (rel * u1).sum(-1)
        for uu, arr in lat_at.items():
            hit = active & (u_now > -uu) & arr.isnan()
            arr[hit] = v_now[hit]
        v_max = torch.where(active & decided, torch.maximum(v_max, v_now), v_max)
        at_bed = active & (u_now > -BED_U)
        outcome[at_bed] = 0
        lat_at_bed[at_bed] = v_now[at_bed]
        t_bed[at_bed] = (k + 1) * dt - t_border[at_bed]
        active &= ~at_bed
        moved = (pos - hist[k % window]).norm(dim=-1)
        now_stalled = active & decided & (k >= window) & (moved < STALL_PROGRESS)
        snags += (now_stalled & ~stalled).float()
        stalled = now_stalled
        hist[k % window] = pos
    return outcome, lat_at_bed, snags, t_bed, fall_uv, lat_at[1.2], lat_at[0.9], v_max


def _pct(k, n):
    return f"{100.0 * k / n:5.1f}%" if n else "    -"


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
        f"[CORNER] {int(used.sum())} robots in L corners, modes {MODES}, {REPEATS} trial(s) of {TRIAL:.0f} s",
        flush=True,
    )
    print(
        f"{'mode':>13}{'n':>6}{'at bed':>9}{'fell/crash':>11}{'on way':>8}{'lat mean':>10}{'lat min':>9}"
        f"{'< 0.25 m':>10}{'snagged':>9}{'t (s)':>8}",
        flush=True,
    )
    for mode in MODES:
        runs = [_trial(env, step, dt, (w0, c, u1, u2), used, mode) for _ in range(REPEATS)]
        outcome, lat, snags, t_bed, fall_uv, lat12, lat09, v_max = (torch.cat([r[i] for r in runs]) for i in range(8))
        sel = used.repeat(REPEATS)
        n = int(sel.sum())
        o = outcome[sel]
        ok = sel & (outcome == 0)
        lat_ok = lat[ok]
        nan = float("nan")
        lat_mean = float(lat_ok.mean()) if len(lat_ok) else nan
        lat_min = float(lat_ok.min()) if len(lat_ok) else nan
        t_mean = float(t_bed[ok].mean()) if len(lat_ok) else nan
        near_bed = _pct(int((lat_ok < 0.25).sum()), len(lat_ok))
        print(
            f"{mode:>13}{n:>6}{_pct(int((o == 0).sum()), n):>9}{_pct(int((o == 1).sum()), n):>11}"
            f"{_pct(int((o == 2).sum()), n):>8}{lat_mean:>10.2f}{lat_min:>9.2f}{near_bed:>10}"
            f"{_pct(int((snags[sel] > 0).sum()), n):>9}{t_mean:>8.2f}",
            flush=True,
        )
        for name, arr in (("u=-1.2", lat12), ("u=-0.9", lat09), ("u=-0.7", lat)):
            a = arr[ok]
            if len(a):
                q = torch.quantile(a, torch.tensor([0.0, 0.01, 0.05, 0.5], device=a.device)).tolist()
                print(
                    f"{'':15}lateral at {name}: min {q[0]:.2f}  p1 {q[1]:.2f}  p5 {q[2]:.2f}  median {q[3]:.2f}"
                    f"  max {float(a.max()):.2f}",
                    flush=True,
                )
        vm = v_max[ok]
        if len(vm):
            p99 = float(torch.quantile(vm, 0.99))
            near = _pct(int((vm > WALL_NEAR).sum()), len(vm))
            far = float(vm.max())
            print(f"{'':15}farthest out before the bed: max {far:.2f}, p99 {p99:.2f}, over {WALL_NEAR} m {near}")
        fu = fall_uv[sel & (outcome == 1)]
        if len(fu):
            spots = ", ".join(f"({a:+.2f}, {b:+.2f})" for a, b in fu.tolist())
            print(f"{'':15}falls at (u, v): {spots}", flush=True)


play_rsl_rl.run_playback = corner_playback

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--task" not in argv:
        argv = ["--task", "G1-Rough-Test"] + argv
    if "--num_envs" not in argv:
        argv += ["--num_envs", "500"]
    play_rsl_rl.run(argv)
