"""Sideways stepping and straight-walking drift test (headless): many G1 robots on open flat floor.

``G1-Strafe-Test`` is flat floor without walls (``src/g1_maze_lab/tasks/trench_test/``) with the challenge's crash
rule. Every robot starts facing +x and gets one fixed command, ramped in with pilot.py's acceleration limits:

* left / right: v_x = 0, v_y = +-STRAFE_SPEEDS, w_z = 0 (the slalom and spike-lane line-ups strafe like this)
* straight: v_x = STRAFE_WALK, v_y = w_z = 0, open loop, so the drift that pilot.py's tracking has to cancel shows

For each command it prints the share that fell, the speed reached along the command (mean over the steady part, from
STRAFE_SETTLE s on) and its ratio to the command, and the drift: for sideways commands the forward speed [m/s] and
turn rate [deg/s], for straight walking the sideways drift [cm per m walked, + = to the robot's left] and the heading
change [deg per m, + = turning left].

Usage (from the project folder):
    PYTHONIOENCODING=utf-8 uv run python scripts/strafe_test.py --checkpoint logs/rsl_rl/<exp>/<run>/model_XXXX.pt
Env vars:
    STRAFE_SPEEDS   sideways speeds [m/s] (default "0.2,0.35,0.5"; pilot.py strafes at up to 0.35)
    STRAFE_WALK     straight speeds [m/s] (default "0.5,1.0")
    STRAFE_TIME     time per trial [s] (default 6)
    STRAFE_SETTLE   start of the measured part [s] (default 1.5)
    STRAFE_REPEATS  trials (default 2)
"""

import math
import os
import sys

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

import g1_maze_lab.tasks  # noqa: E402, F401  (registers G1-Strafe-Test)

SPEEDS = [float(v) for v in os.environ.get("STRAFE_SPEEDS", "0.2,0.35,0.5").split(",")]
WALK = [float(v) for v in os.environ.get("STRAFE_WALK", "0.5,1.0").split(",")]
TRIAL = float(os.environ.get("STRAFE_TIME", 6.0))
SETTLE = float(os.environ.get("STRAFE_SETTLE", 1.5))
REPEATS = int(os.environ.get("STRAFE_REPEATS", 2))
LIMITS = (2.0, 2.0, 2.8)  # pilot.py ap.max_acc (v_x, v_y) [m/s^2] and ap.max_ang_acc [rad/s^2]
SPECS = [("left", v) for v in SPEEDS] + [("right", v) for v in SPEEDS] + [("straight", v) for v in WALK]


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


def _targets(spec_idx, dev):
    """Command (n, 3) for each robot's spec."""
    table = []
    for kind, v in SPECS:
        table.append((0.0, v, 0.0) if kind == "left" else (0.0, -v, 0.0) if kind == "right" else (v, 0.0, 0.0))
    return torch.tensor(table, device=dev)[spec_idx]


def _trial(env, step, dt, spec_idx):
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("base_velocity")
    n, dev = env.num_envs, env.device
    env.reset()
    origins = env.scene.env_origins.clone()
    pose = robot.data.default_root_pose.torch.clone()
    pose[:, :2] = origins[:, :2]
    pose[:, 2] += origins[:, 2]
    pose[:, 3:7] = torch.tensor((0.0, 0.0, 0.0, 1.0), device=dev)  # Facing +x
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))

    target = _targets(spec_idx, dev)
    command = torch.zeros(n, 3, device=dev)
    lim = torch.tensor(LIMITS, device=dev) * dt
    active = torch.ones(n, dtype=torch.bool, device=dev)
    vel_sum, wz_sum, k_meas = torch.zeros(n, 2, device=dev), torch.zeros(n, device=dev), 0
    p_settle = yaw_settle = None
    for k in range(int(TRIAL / dt)):
        command += (torch.where(active.unsqueeze(1), target, 0.0) - command).clamp(-lim, lim)
        cmd.time_left[:] = 1.0e9
        cmd.vel_command_b[:] = command
        with torch.inference_mode():
            step()
        active &= ~env.termination_manager.terminated
        pos, yaw = robot.data.root_pos_w.torch[:, :2], _yaw(robot.data.root_quat_w.torch)
        if (k + 1) * dt < SETTLE:
            continue
        if p_settle is None:
            p_settle, yaw_settle = pos.clone(), yaw.clone()
        vel = robot.data.root_lin_vel_w.torch[:, :2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        vel_sum += torch.stack((c * vel[:, 0] + s * vel[:, 1], -s * vel[:, 0] + c * vel[:, 1]), -1)  # Yaw frame
        wz_sum += robot.data.root_ang_vel_w.torch[:, 2]
        k_meas += 1
    pos, yaw = robot.data.root_pos_w.torch[:, :2], _yaw(robot.data.root_quat_w.torch)
    move = pos - p_settle
    c, s = torch.cos(yaw_settle), torch.sin(yaw_settle)
    fwd, side = c * move[:, 0] + s * move[:, 1], -s * move[:, 0] + c * move[:, 1]  # In the heading at SETTLE
    return ~active, vel_sum / max(k_meas, 1), wz_sum / max(k_meas, 1), fwd, side, _wrap(yaw - yaw_settle)


def _pct(k, n):
    return f"{100.0 * k / n:5.1f}%" if n else "    -"


def strafe_playback(step, *, dt, args_cli, env_cfg):
    env = _find_env(step)
    cmd = env.command_manager.get_term("base_velocity")
    cmd.cfg.heading_command = False
    cmd.is_heading_env[:] = False
    cmd.is_standing_env[:] = False
    spec_idx = torch.arange(env.num_envs, device=env.device) % len(SPECS)
    print(
        f"[STRAFE] {env.num_envs} robots on open floor, {REPEATS} trial(s) of {TRIAL:.0f} s, measured from "
        f"{SETTLE:.1f} s",
        flush=True,
    )
    runs = [_trial(env, step, dt, spec_idx) for _ in range(REPEATS)]
    fell, vel, wz, fwd, side, dyaw = (torch.cat([r[i] for r in runs]) for i in range(6))
    idx = spec_idx.repeat(REPEATS)
    print(f"\n{'command':>16}{'n':>6}{'fell':>8}{'reached':>11}{'ratio':>8}   drift", flush=True)
    for si, (kind, v) in enumerate(SPECS):
        sel = (idx == si) & ~fell
        n_all, n = int((idx == si).sum()), int(sel.sum())
        if n == 0:
            print(f"{kind:>10} {v:4.2f}{n_all:>6}{_pct(n_all, n_all):>8}", flush=True)
            continue
        if kind == "straight":
            reached = float(vel[sel, 0].mean())
            per_m = fwd[sel].clamp(min=0.1)
            drift = (
                f"sideways {100.0 * float((side[sel] / per_m).mean()):+5.1f} cm/m, "
                f"heading {math.degrees(float((dyaw[sel] / per_m).mean())):+5.1f} deg/m"
            )
        else:
            sign = 1.0 if kind == "left" else -1.0
            reached = sign * float(vel[sel, 1].mean())
            drift = (
                f"forward {float(vel[sel, 0].mean()):+.3f} m/s, turn {math.degrees(float(wz[sel].mean())):+5.1f} deg/s"
            )
        print(
            f"{kind:>10} {v:4.2f}{n_all:>6}{_pct(n_all - n, n_all):>8}{reached:>7.3f} m/s{100.0 * reached / v:>7.0f}%"
            f"   {drift}",
            flush=True,
        )


play_rsl_rl.run_playback = strafe_playback

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--task" not in argv:
        argv = ["--task", "G1-Strafe-Test"] + argv
    if "--num_envs" not in argv:  # The shared play mode caps the scene at 50 robots
        argv += ["--num_envs", "500"]
    play_rsl_rl.run(argv)
