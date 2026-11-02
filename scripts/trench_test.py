"""Trench crossing test (headless): many G1 robots, each walking down a walled corridor and across one trench.

The corridors have the hazard maze's dimensions (1.85 m between 1.0 m walls) and the challenge's crash rule (torso,
pelvis, elbows or palms touching anything), see ``src/g1_maze_lab/tasks/trench_test/``. Terrain rows are trench
widths (``TRENCH_WIDTHS``). Each robot starts TRENCH_START m before the trench line, turned by one of the TRENCH_YAW
headings and up to 0.1 m off the centreline, and is steered like pilot.py's normal tracking: pure pursuit to a point
1 m ahead on the centreline, with the pilot's acceleration and turn-rate limits.

For every trench width, start heading and speed it prints the share of robots that crossed, fell or crashed, or got
stuck, and two near-miss counts: stumble (pelvis more than 8 cm below its walking height near the trench) and slow
(more than twice the normal time within 0.5 m of the trench line).

Usage (from the project folder):
    PYTHONIOENCODING=utf-8 uv run python scripts/trench_test.py --checkpoint logs/rsl_rl/<exp>/<run>/model_XXXX.pt
Env vars:
    TRENCH_SPEEDS   forward speeds [m/s] (default "0.5,0.7,0.9"; pilot.py walks at 0.70 near trenches)
    TRENCH_YAW      start heading offsets [deg], spread over the robots with alternating sign (default "0,15")
    TRENCH_START    start distance before the trench line [m] (default 2.0)
    TRENCH_TIME     time per trial [s] (default 10)
    TRENCH_REPEATS  trials per speed (default 2)
"""

import math
import os
import sys

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

import g1_maze_lab.tasks  # noqa: E402, F401  (registers G1-Trench-Test)
from g1_maze_lab.tasks.trench_test.terrain import TRENCH_WIDTHS  # noqa: E402

SPEEDS = [float(v) for v in os.environ.get("TRENCH_SPEEDS", "0.5,0.7,0.9").split(",")]
YAWS = [float(v) for v in os.environ.get("TRENCH_YAW", "0,15").split(",")]
START = float(os.environ.get("TRENCH_START", 2.0))
TRIAL = float(os.environ.get("TRENCH_TIME", 10.0))
REPEATS = int(os.environ.get("TRENCH_REPEATS", 2))
LOOKAHEAD = 1.0  # pilot.py PURSUIT_LOOKAHEAD [m]
MAX_ACC, MAX_ANG_ACC, MAX_TURN = 2.0, 2.8, 1.1  # pilot.py ap.max_acc [m/s^2], ap.max_ang_acc, ap.max_turn
LATERAL = 0.1  # Random start offset from the centreline [m]
PAST = 1.0  # Crossed = pelvis this far past the trench line [m]
NEAR = 0.5  # Near-miss window around the trench line [m]
STUMBLE_DROP = 0.08  # Pelvis drop below its walking height that counts as a stumble [m]
OUTCOMES = ("crossed", "fell/crash", "stuck", "blowup")


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
    """Robot i -> row i % rows (trench width), spread over the columns; returns (width index, heading index, sign)."""
    terrain = env.scene.terrain
    rows, cols = terrain.terrain_origins.shape[:2]
    ids = torch.arange(env.num_envs, device=env.device)
    k = ids // rows
    terrain.terrain_levels[:] = ids % rows
    terrain.terrain_types[:] = k % cols
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    sign = torch.where((k // len(YAWS)) % 2 == 0, 1.0, -1.0)
    return ids % rows, k % len(YAWS), sign


def _trial(env, step, speed, dt, spawn_before, width_idx, yaw_idx, sign):
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("base_velocity")
    n, dev = env.num_envs, env.device
    env.reset()
    origins = env.scene.env_origins.clone()
    trench_x = origins[:, 0] + spawn_before  # The terrain's spawn point is spawn_before ahead of the trench line
    centre_y = origins[:, 1]

    # Place every robot: TRENCH_START before the line, a random lateral offset, its heading offset
    yaw0 = sign * torch.tensor(YAWS, device=dev)[yaw_idx] * math.pi / 180.0
    pose = robot.data.default_root_pose.torch.clone()
    pose[:, 0] = trench_x - START
    pose[:, 1] = centre_y + (torch.rand(n, device=dev) * 2.0 - 1.0) * LATERAL
    pose[:, 2] += origins[:, 2]
    pose[:, 3:7] = torch.stack(
        (torch.zeros_like(yaw0), torch.zeros_like(yaw0), torch.sin(0.5 * yaw0), torch.cos(0.5 * yaw0)), dim=-1
    )
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))

    command = torch.zeros(n, 3, device=dev)
    lim = torch.tensor([MAX_ACC * dt, MAX_ACC * dt, MAX_ANG_ACC * dt], device=dev)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    outcome = torch.full((n,), 2, dtype=torch.long, device=dev)  # stuck unless something else happens
    crash_u = torch.full((n,), float("nan"), device=dev)
    walk_z, walk_n = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    min_z = torch.full((n,), float("inf"), device=dev)
    t_near = torch.zeros(n, device=dev)
    blowup_term = "physics_blowup" in env.termination_manager.active_terms
    for _ in range(int(TRIAL / dt)):
        pos = robot.data.root_pos_w.torch
        u, v = pos[:, 0] - trench_x, pos[:, 1] - centre_y
        err = _wrap(torch.atan2(-v, torch.full_like(v, LOOKAHEAD)) - _yaw(robot.data.root_quat_w.torch))
        target = torch.stack(
            (torch.full_like(u, speed), torch.zeros_like(u), (1.6 * err).clamp(-MAX_TURN, MAX_TURN)), -1
        )
        target[~active] = 0.0
        command += (target - command).clamp(-lim, lim)
        cmd.time_left[:] = 1.0e9
        cmd.vel_command_b[:] = command
        with torch.inference_mode():
            step()

        # A terminated robot was already put back on its spawn point by the env: judge it by the state before
        done = active & env.termination_manager.terminated
        if bool(done.any()):
            blew = env.termination_manager.get_term("physics_blowup") if blowup_term else torch.zeros_like(done)
            outcome[done] = torch.where(blew[done], 3, 1)
            crash_u[done] = u[done]
            active &= ~done
        pos = robot.data.root_pos_w.torch
        u, z = pos[:, 0] - trench_x, pos[:, 2] - origins[:, 2]
        walking = active & (u < -0.8) & (u > -START + 0.5)  # Walking height before the trench, after starting up
        walk_z += torch.where(walking, z, 0.0)
        walk_n += walking.float()
        near = active & (u.abs() < NEAR)
        min_z = torch.where(near, torch.minimum(min_z, z), min_z)
        t_near += near.float() * dt
        crossed = active & (u > PAST)
        outcome[crossed] = 0
        active &= ~crossed

    nominal = torch.where(walk_n > 0, walk_z / walk_n.clamp(min=1.0), torch.nan)
    stumble = torch.isfinite(min_z) & (nominal - min_z > STUMBLE_DROP)
    slow = t_near > 2.0 * (2.0 * NEAR / speed)
    return outcome, crash_u, stumble, slow


def _pct(k, n):
    return f"{100.0 * k / n:5.1f}%" if n else "    -"


def trench_playback(step, *, dt, args_cli, env_cfg):
    env = _find_env(step)
    cmd = env.command_manager.get_term("base_velocity")
    cmd.cfg.heading_command = False
    cmd.is_heading_env[:] = False
    cmd.is_standing_env[:] = False
    spawn_before = env_cfg.scene.terrain.terrain_generator.sub_terrains["corridor_trench"].start_distance
    width_idx, yaw_idx, sign = _assign_corridors(env)
    print(
        f"[TRENCH] {env.num_envs} robots, widths {list(TRENCH_WIDTHS)} m, start {START} m before the line, "
        f"headings +-{YAWS} deg, {REPEATS} trial(s) of {TRIAL:.0f} s per speed",
        flush=True,
    )
    for speed in SPEEDS:
        rows = []
        for _ in range(REPEATS):
            rows.append(_trial(env, step, speed, dt, spawn_before, width_idx, yaw_idx, sign))
        outcome = torch.cat([r[0] for r in rows])
        crash_u = torch.cat([r[1] for r in rows])
        stumble = torch.cat([r[2] for r in rows])
        slow = torch.cat([r[3] for r in rows])
        w_idx, y_idx = width_idx.repeat(REPEATS), yaw_idx.repeat(REPEATS)
        print(f"\n[TRENCH] speed {speed:.2f} m/s", flush=True)
        print(
            f"{'width':>7}{'heading':>9}{'n':>6}{'crossed':>9}{'fell/crash':>11}{'stuck':>8}{'blowup':>8}"
            f"{'stumble':>9}{'slow':>8}   crashes before / on / after the trench",
            flush=True,
        )
        for wi, width in enumerate(TRENCH_WIDTHS):
            for yi, yaw in enumerate(YAWS):
                sel = (w_idx == wi) & (y_idx == yi)
                n = int(sel.sum())
                counts = [int((outcome[sel] == k).sum()) for k in range(len(OUTCOMES))]
                cu = crash_u[sel & (outcome == 1)]
                where = (int((cu < -0.3).sum()), int((cu.abs() <= 0.3).sum()), int((cu > 0.3).sum()))
                print(
                    f"{width:>7.2f}{yaw:>5.0f} deg{n:>6}{_pct(counts[0], n):>9}{_pct(counts[1], n):>11}"
                    f"{_pct(counts[2], n):>8}{_pct(counts[3], n):>8}{_pct(int(stumble[sel].sum()), n):>9}"
                    f"{_pct(int(slow[sel].sum()), n):>8}   {where[0]} / {where[1]} / {where[2]}",
                    flush=True,
                )


play_rsl_rl.run_playback = trench_playback

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--task" not in argv:
        argv = ["--task", "G1-Trench-Test"] + argv
    if "--num_envs" not in argv:  # The shared play mode caps the scene at 50 robots
        argv += ["--num_envs", "500"]
    play_rsl_rl.run(argv)
