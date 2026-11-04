"""Rough-floor test (headless): G1 robots walking corridors of maze rough cells like pilot.py, counting snags.

``G1-Rough-Test`` (``src/g1_maze_lab/tasks/trench_test/``): walled corridors with the maze's dimensions and crash rule,
every 2 m cell holding the maze's 10 loose 2-8 cm boxes (the same boxes in every run). Rows are ``ROUGH_KINDS``:

* straight: a straight corridor
* lineup: the same, but 2 m in the lane moves 0.48 m sideways (half the robots each way), like lining up for a spike
  lane (where L9 seed 5 snags)
* left / right: an L turning in a rough corner cell (where L7 seed 2 snags)

Robots are steered like pilot.py's normal tracking: pure pursuit to a point 1 m ahead on the lane, yaw rate 1.6 x the
heading error (at most 1.1 rad/s), forward speed scaled by cos^2 of the heading error (at least 0.25 m/s, and 0.25
when the error is over 0.7 rad), sideways 0.45 x the lane offset in the body frame (at most 0.22 m/s) when lined up
and more than 1 m from a corner, with pilot.py's acceleration limits.

For each kind and speed it prints the share of robots that finished (1 m before the corridor's end), fell or crashed,
or were still on the way at the end of the trial; snags (progress along the lane under 0.1 m in 1 s, counted once per
stall) per robot and the share of robots with at least one; the time spent stalled; and the mean time to finish.

Usage (from the project folder):
    PYTHONIOENCODING=utf-8 uv run python scripts/rough_test.py --checkpoint logs/rsl_rl/<exp>/<run>/model_XXXX.pt
Env vars:
    ROUGH_SPEEDS    forward speeds [m/s] (default "0.5,0.68,0.9"; pilot.py's rough-floor limit is 0.68)
    ROUGH_TIME      time per trial [s] (default 25)
    ROUGH_REPEATS   trials per speed (default 2)
"""

import os
import sys

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

import g1_maze_lab.tasks  # noqa: E402, F401  (registers G1-Rough-Test)
from g1_maze_lab.tasks.trench_test.terrain import ROUGH_LINEUP, rough_path  # noqa: E402

SPEEDS = [float(v) for v in os.environ.get("ROUGH_SPEEDS", "0.5,0.68,0.9").split(",")]
TRIAL = float(os.environ.get("ROUGH_TIME", 25.0))
REPEATS = int(os.environ.get("ROUGH_REPEATS", 2))
LOOKAHEAD = 1.0  # pilot.py PURSUIT_LOOKAHEAD [m]
MIN_SPEED, MAX_TURN = 0.25, 1.1  # pilot.py ap.min_speed [m/s], ap.max_turn [rad/s]
LIMITS = (2.0, 2.0, 2.8)  # pilot.py ap.max_acc (v_x, v_y) [m/s^2] and ap.max_ang_acc [rad/s^2]
FINISH_BEFORE_END = 1.0  # Finished = this far before the corridor's end [m]
STALL_WINDOW, STALL_PROGRESS = 1.0, 0.1  # Snag = under STALL_PROGRESS m along the lane in STALL_WINDOW s
STARTUP = 1.5  # No snag counted while starting up [s]
OUTCOMES = ("finished", "fell/crash", "on the way")


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
    """Robot i -> row i % rows (corridor kind), spread over the columns; returns the kind index per robot."""
    terrain = env.scene.terrain
    rows, cols = terrain.terrain_origins.shape[:2]
    ids = torch.arange(env.num_envs, device=env.device)
    terrain.terrain_levels[:] = ids % rows
    terrain.terrain_types[:] = (ids // rows) % cols
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    return ids % rows


class _Lane:
    """Each robot's lane: two segments W0-W1-W2 (world xy), plus a sideways offset for the lineup robots."""

    def __init__(self, waypoints, lineup_sign, is_corner):
        self.w0, self.w1, self.w2 = waypoints.unbind(1)  # (n, 2) each
        self.l1 = (self.w1 - self.w0).norm(dim=-1)
        self.l2 = (self.w2 - self.w1).norm(dim=-1)
        self.u1 = (self.w1 - self.w0) / self.l1.unsqueeze(-1)
        self.u2 = (self.w2 - self.w1) / self.l2.unsqueeze(-1)
        self.total = self.l1 + self.l2
        self.lineup_sign, self.is_corner = lineup_sign, is_corner

    def offset(self, s):
        return self.lineup_sign * ROUGH_LINEUP[1] * (s >= ROUGH_LINEUP[0]).float()

    def point(self, s):
        """Lane point (n, 2) at arc length s, with the lineup offset to the left of the direction of travel."""
        on1 = (s < self.l1).unsqueeze(-1)
        u = torch.where(on1, self.u1, self.u2)
        p = torch.where(on1, self.w0 + self.u1 * s.unsqueeze(-1), self.w1 + self.u2 * (s - self.l1).unsqueeze(-1))
        normal = torch.stack((-u[:, 1], u[:, 0]), -1)
        return p + normal * self.offset(s).unsqueeze(-1)

    def progress(self, pos):
        """Arc length of the nearest point of the (offset-free) lane."""
        t1 = ((pos - self.w0) * self.u1).sum(-1).clamp(0.0, None).minimum(self.l1)
        t2 = ((pos - self.w1) * self.u2).sum(-1).clamp(min=0.0)  # Beyond W2 the last segment extends
        e1 = (pos - self.w0 - self.u1 * t1.unsqueeze(-1)).norm(dim=-1)
        e2 = (pos - self.w1 - self.u2 * t2.unsqueeze(-1)).norm(dim=-1)
        return torch.where(e2 < e1, self.l1 + t2, t1)


def _lanes(env, kind_idx, kinds, sub):
    n, dev = env.num_envs, env.device
    table = []
    for kind in kinds:
        p = rough_path(kind, sub)
        if len(p) == 2:  # Straight: split into two collinear segments
            p = [p[0], (0.5 * (p[0][0] + p[1][0]), 0.5 * (p[0][1] + p[1][1])), p[1]]
        table.append(p)
    waypoints = torch.tensor(table, device=dev)[kind_idx] + env.scene.env_origins[:, :2].unsqueeze(1)
    is_lineup = torch.tensor([k == "lineup" for k in kinds], device=dev)[kind_idx]
    sign = torch.where((torch.arange(n, device=dev) // len(kinds)) % 2 == 0, 1.0, -1.0)
    is_corner = torch.tensor([k in ("left", "right") for k in kinds], device=dev)[kind_idx]
    return _Lane(waypoints, torch.where(is_lineup, sign, 0.0), is_corner)


def _trial(env, step, speed, dt, lane):
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("base_velocity")
    n, dev = env.num_envs, env.device
    env.reset()
    origins = env.scene.env_origins
    heading = torch.atan2(lane.u1[:, 1], lane.u1[:, 0])
    pose = robot.data.default_root_pose.torch.clone()
    pose[:, :2] = origins[:, :2]
    pose[:, 2] += origins[:, 2]
    zero = torch.zeros_like(heading)
    pose[:, 3:7] = torch.stack((zero, zero, torch.sin(0.5 * heading), torch.cos(0.5 * heading)), -1)
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))

    command = torch.zeros(n, 3, device=dev)
    lim = torch.tensor(LIMITS, device=dev) * dt
    active = torch.ones(n, dtype=torch.bool, device=dev)
    outcome = torch.full((n,), 2, dtype=torch.long, device=dev)  # On the way unless something else happens
    t_finish = torch.full((n,), float("nan"), device=dev)
    window = max(int(round(STALL_WINDOW / dt)), 1)
    history = torch.zeros(window, n, device=dev)  # Progress over the last STALL_WINDOW s (ring buffer)
    stalled = torch.zeros(n, dtype=torch.bool, device=dev)
    snags, t_stall = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    for k in range(int(TRIAL / dt)):
        pos, yaw = robot.data.root_pos_w.torch[:, :2], _yaw(robot.data.root_quat_w.torch)
        s = lane.progress(pos)
        aim = lane.point(s + LOOKAHEAD) - pos
        yaw_err = _wrap(torch.atan2(aim[:, 1], aim[:, 0]) - yaw)
        lat = lane.point(s) - pos
        lat_b = -torch.sin(yaw) * lat[:, 0] + torch.cos(yaw) * lat[:, 1]  # Lane offset in the body frame
        curve = lane.is_corner & ((s - lane.l1).abs() < 1.0)
        vy = torch.where((yaw_err.abs() < 0.45) & ~curve, (0.45 * lat_b).clamp(-0.22, 0.22), 0.0).clamp(-0.28, 0.28)
        v = MIN_SPEED + (speed - MIN_SPEED) * torch.cos(yaw_err).clamp(min=0.0) ** 2
        v = torch.where(yaw_err.abs() > 0.7, MIN_SPEED, v)
        target = torch.stack((v, vy, (1.6 * yaw_err).clamp(-MAX_TURN, MAX_TURN)), -1)
        target[~active] = 0.0
        command += (target - command).clamp(-lim, lim)
        cmd.time_left[:] = 1.0e9
        cmd.vel_command_b[:] = command
        with torch.inference_mode():
            step()

        # A terminated robot was already put back on its spawn point by the env
        done = active & env.termination_manager.terminated
        outcome[done] = 1
        active &= ~done
        s = lane.progress(robot.data.root_pos_w.torch[:, :2])
        finished = active & (s > lane.total - FINISH_BEFORE_END)
        outcome[finished] = 0
        t_finish[finished] = (k + 1) * dt
        active &= ~finished
        now_stalled = active & ((k + 1) * dt > STARTUP) & (s - history[k % window] < STALL_PROGRESS)
        if k < window:
            now_stalled[:] = False
        snags += (now_stalled & ~stalled).float()
        t_stall += now_stalled.float() * dt
        stalled = now_stalled
        history[k % window] = s
    return outcome, snags, t_stall, t_finish


def _pct(k, n):
    return f"{100.0 * k / n:5.1f}%" if n else "    -"


def rough_playback(step, *, dt, args_cli, env_cfg):
    env = _find_env(step)
    cmd = env.command_manager.get_term("base_velocity")
    cmd.cfg.heading_command = False
    cmd.is_heading_env[:] = False
    cmd.is_standing_env[:] = False
    gen = env_cfg.scene.terrain.terrain_generator
    sub = next(iter(gen.sub_terrains.values()))
    sub.size = tuple(gen.size)  # rough_path reads the tile size
    kinds = sub.kinds
    kind_idx = _assign_corridors(env)
    lane = _lanes(env, kind_idx, kinds, sub)
    print(
        f"[ROUGH] {env.num_envs} robots, corridor kinds {list(kinds)}, {REPEATS} trial(s) of {TRIAL:.0f} s per speed",
        flush=True,
    )
    for speed in SPEEDS:
        runs = [_trial(env, step, speed, dt, lane) for _ in range(REPEATS)]
        outcome, snags, t_stall, t_finish = (torch.cat([r[i] for r in runs]) for i in range(4))
        idx = kind_idx.repeat(REPEATS)
        print(f"\n[ROUGH] speed {speed:.2f} m/s", flush=True)
        print(
            f"{'kind':>9}{'n':>6}{'finished':>10}{'fell/crash':>11}{'on the way':>11}{'snags/robot':>13}"
            f"{'snagged':>9}{'stalled':>10}{'t_finish':>10}",
            flush=True,
        )
        for ki, kind in enumerate(kinds):
            sel = idx == ki
            n = int(sel.sum())
            counts = [int((outcome[sel] == o).sum()) for o in range(len(OUTCOMES))]
            tf = t_finish[sel & (outcome == 0)]
            tf_s = f"{float(tf.mean()):7.2f} s" if len(tf) else "      - "
            print(
                f"{kind:>9}{n:>6}{_pct(counts[0], n):>10}{_pct(counts[1], n):>11}{_pct(counts[2], n):>11}"
                f"{float(snags[sel].mean()):>13.2f}{_pct(int((snags[sel] > 0).sum()), n):>9}"
                f"{float(t_stall[sel].mean()):>8.2f} s{tf_s:>10}",
                flush=True,
            )


play_rsl_rl.run_playback = rough_playback

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--task" not in argv:
        argv = ["--task", "G1-Rough-Test"] + argv
    if "--num_envs" not in argv:  # The shared play mode caps the scene at 50 robots
        argv += ["--num_envs", "500"]
    play_rsl_rl.run(argv)
