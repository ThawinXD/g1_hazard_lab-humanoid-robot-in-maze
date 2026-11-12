"""Ice turn test (headless): many G1 robots walk onto a maze cell of ice and turn on the spot like pilot.py.

The corridors have the hazard maze's dimensions and crash rule and one 1.85 m ice tile with the challenge's friction
(0.2), see ``src/g1_maze_lab/tasks/trench_test/``:

* ``G1-Ice-Test`` (default): a straight corridor; the robots turn by one of the ICE_TURNS angles and stand.
* ``G1-Ice-Corner-Test``: an L-shaped corridor turning left at the ice cell, like a maze corner; the robots turn 90
  deg, walk out toward the next cell and stand. Terrain rows are what that next cell holds (plain floor, loose boxes,
  a 0.28 m ramp, a spike bed), which the height scanner sees while the robot turns.

Each robot copies pilot.py's turn on the spot (``_pivot_command``) on the ice:

1. walk in along the centreline at ICE_SPEED (pure pursuit, the pilot's acceleration and turn-rate limits)
2. from ICE_BRAKE_U before the ice cell's centre (the cell border, where the pilot decides an ice corner): brake at
   ICE_BRAKE until the command is below 0.05 m/s and the body below 0.15 m/s (at most 1.5 s)
3. rotate in place: w = min(max(1.6 * error, 0.5), rate) until within 0.3 rad
4. (corner only) walk out at ICE_SPEED toward the next cell, to 0.9 m past the ice cell's centre
5. stand still for 1 s

For every rotation rate and turn angle (or next cell) it prints the share of robots that completed the turn or fell /
crashed (and in which phase), plus near misses: spun (body yaw rate above twice the rate limit while rotating) and the
mean rotation time.

Usage (from the project folder):
    PYTHONIOENCODING=utf-8 uv run python scripts/ice_turn_test.py --checkpoint logs/rsl_rl/<exp>/<run>/model_XXXX.pt
    PYTHONIOENCODING=utf-8 uv run python scripts/ice_turn_test.py --task G1-Ice-Corner-Test --checkpoint ...
Env vars:
    ICE_RATES    rotation rate limits [rad/s] (default "0.3,0.45,0.6"; pilot.py PIVOT_TURN_RATE_ICE = 0.6)
    ICE_TURNS    turn angles [deg], straight corridor only, alternating direction (default "90,180")
    ICE_SPEED    approach speed [m/s] (default 0.55, pilot.py's ice limit)
    ICE_BRAKE    braking deceleration [m/s^2] (default 0.6, PIVOT_BRAKE_ACC_ICE)
    ICE_BRAKE_U  start braking this far before the ice cell's centre [m] (default 1.0)
    ICE_REPEATS  trials per rate (default 2)
    ICE_APPROACH approaches to compare, comma-separated (default "centre"): ``centre`` walks in on the centreline
                 (+-0.1); the ``lane*`` ones come out of a spike lane like pilot.py: on the lane line 0.48 m off the
                 centreline (half the robots each side; + = inside of the left turn) at the lane speed 0.48 until the
                 bed's end (1.22 m before the ice cell's centre), then ``lane`` steers for the ice cell's centre
                 (normal tracking at <= ICE_SPEED), ``lane_straight`` keeps the lane line, ``lane_slow`` /
                 ``lane_straight_slow`` do the same at <= 0.25 m/s (pilot.py's ICE_FROM_SPIKE_SPEED), ``lane_wait``
                 stands 1 s before rotating, ``lane_centre`` walks slowly on until 0.15 m from the centreline.
                 Per lane side it prints the falls and where they happened. No wall clearance as in pilot.py's turn
                 (inside-lane robots touch the inner corner's wall end)
"""

import math
import os
import sys

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

import g1_maze_lab.tasks  # noqa: E402, F401  (registers G1-Ice-Test / G1-Ice-Corner-Test)

RATES = [float(v) for v in os.environ.get("ICE_RATES", "0.3,0.45,0.6").split(",")]
TURNS = [float(v) for v in os.environ.get("ICE_TURNS", "90,180").split(",")]
SPEED = float(os.environ.get("ICE_SPEED", 0.55))
BRAKE = float(os.environ.get("ICE_BRAKE", 0.6))
BRAKE_U = float(os.environ.get("ICE_BRAKE_U", 1.0))
REPEATS = int(os.environ.get("ICE_REPEATS", 2))
APPROACHES = os.environ.get("ICE_APPROACH", "centre").split(",")
LANE, LANE_SPEED, BED_END, SLOW = 0.48, 0.48, 1.22, 0.25  # pilot.py lane offset / speed; bed's end before the centre
LOOKAHEAD = 1.0  # pilot.py PURSUIT_LOOKAHEAD [m]
MAX_ACC, MAX_ANG_ACC, MAX_TURN = 2.0, 2.8, 1.1  # pilot.py ap.max_acc [m/s^2], ap.max_ang_acc, ap.max_turn
ICE_ACC, ICE_TURN = 0.6, 0.65  # pilot.py: acceleration and tracking yaw-rate limits on ice
EXIT_ERR = 0.3  # pilot.py PIVOT_EXIT_ERR [rad]
OUT_V = 0.9  # Corner: walk out until this far past the ice cell's centre [m]
HOLD = 1.0  # Stand still after the turn [s]
LATERAL = 0.1  # Random start offset from the centreline [m]
TRIAL = 30.0  # Time limit per trial [s]
PHASES = ("walk", "brake", "rotate", "out", "stand")
DONE = len(PHASES)


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


def _assign_rows(env):
    """Robot i -> terrain row i % rows, spread over the columns; returns the row of every robot."""
    terrain = env.scene.terrain
    rows, cols = terrain.terrain_origins.shape[:2]
    ids = torch.arange(env.num_envs, device=env.device)
    terrain.terrain_levels[:] = ids % rows
    terrain.terrain_types[:] = (ids // rows) % cols
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    return ids % rows


def _trial(env, step, rate, dt, start, turn, corner, approach="centre"):
    """One trial; ``turn`` (n,) signed turn angle per robot [rad], ``corner`` walk out north after a left turn."""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("base_velocity")
    n, dev = env.num_envs, env.device
    env.reset()
    origins = env.scene.env_origins.clone()
    ice_x = origins[:, 0] + start  # The terrain's spawn point is ``start`` before the ice cell's centre
    centre_y = origins[:, 1]

    pose = robot.data.default_root_pose.torch.clone()
    pose[:, 0] = ice_x - start
    lane_v = torch.where(torch.arange(n, device=dev) % 2 == 0, LANE, -LANE)  # Lane line per robot (lane approaches)
    if approach == "centre":
        pose[:, 1] = centre_y + (torch.rand(n, device=dev) * 2.0 - 1.0) * LATERAL
    else:
        pose[:, 1] = centre_y + lane_v
    pose[:, 2] += origins[:, 2]
    pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))

    command = torch.zeros(n, 3, device=dev)
    phase = torch.zeros(n, dtype=torch.long, device=dev)
    t_phase = torch.zeros(n, device=dev)  # Time in the current phase
    target = torch.zeros(n, device=dev)  # Heading to rotate to
    fell = torch.full((n,), -1, dtype=torch.long, device=dev)  # Phase in which the robot fell (-1: did not)
    spun = torch.zeros(n, dtype=torch.bool, device=dev)
    t_rot = torch.full((n,), float("nan"), device=dev)
    prev_yaw = _yaw(robot.data.root_quat_w.torch)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    fall_at = torch.full((n, 3), float("nan"), device=dev)  # (u, v, lane side) where a robot fell
    for _ in range(int(TRIAL / dt)):
        pos = robot.data.root_pos_w.torch
        yaw = _yaw(robot.data.root_quat_w.torch)
        u, v = pos[:, 0] - ice_x, pos[:, 1] - centre_y
        on_ice = (u.abs() < 0.925) & (v.abs() < 0.925)
        speed_b = torch.linalg.norm(robot.data.root_lin_vel_b.torch[:, :2], dim=-1)

        # Phase changes
        go = (phase == 0) & (u > -BRAKE_U)
        if approach == "lane_centre":  # Walk on (slowly, steering for the centre) until near the centreline
            go = (phase == 0) & (u > -BRAKE_U) & ((v.abs() < 0.15) | (u > -0.2))
        phase[go], t_phase[go] = 1, 0.0
        settle = 1.0 if approach == "lane_wait" else 0.0  # Stand this long after stopping, before rotating
        stopped = (command[:, 0].abs() < 0.05) & (speed_b < 0.15)
        go = (phase == 1) & ((stopped & (t_phase > settle)) | (t_phase > 1.5 + settle))
        target[go] = _wrap(yaw[go] + turn[go])
        phase[go], t_phase[go] = 2, 0.0
        err = _wrap(target - yaw)
        go = (phase == 2) & (err.abs() < EXIT_ERR)
        t_rot[go] = t_phase[go]
        phase[go], t_phase[go] = (3 if corner else 4), 0.0
        go = (phase == 3) & (v > OUT_V)
        phase[go], t_phase[go] = 4, 0.0
        go = (phase == 4) & (t_phase > HOLD)
        phase[go] = DONE

        # Commands per phase (rate limits as in pilot.py)
        new = command.clone()
        lim_acc = torch.where(on_ice, ICE_ACC, MAX_ACC) * dt
        w_lim = torch.where(on_ice, ICE_TURN, MAX_TURN)
        walk_in, walk_out = phase == 0, phase == 3
        head_in = _wrap(torch.atan2(-v, torch.full_like(v, LOOKAHEAD)) - yaw)  # Pursuit along the entry corridor
        speed_in = torch.full_like(v, SPEED)
        if approach != "centre":
            in_lane = u < -BED_END
            on_line = _wrap(torch.atan2(lane_v - v, torch.full_like(v, LOOKAHEAD)) - yaw)  # Along the lane line
            to_centre = _wrap(torch.atan2(-v, -u) - yaw)  # Steer for the ice cell's centre
            after = on_line if "straight" in approach else to_centre
            head_in = torch.where(in_lane, on_line, after)
            v_after = SLOW if ("slow" in approach or approach == "lane_centre") else SPEED
            speed_in = torch.where(in_lane, torch.full_like(v, LANE_SPEED), torch.full_like(v, v_after))
        head_out = _wrap(torch.atan2(torch.full_like(u, LOOKAHEAD), -u) - yaw)  # ... and north along the exit one
        head = torch.where(walk_out, head_out, head_in)
        walking = walk_in | walk_out
        v_des = torch.where(walk_in, speed_in, torch.full_like(speed_in, SPEED)) * torch.cos(head).clamp(min=0.0) ** 2
        new[walking, 0] = command[walking, 0] + (v_des - command[:, 0])[walking].clamp(
            -lim_acc[walking], lim_acc[walking]
        )
        w_des = torch.maximum(torch.minimum(1.6 * head, w_lim), -w_lim)
        new[walking, 2] = command[walking, 2] + (w_des - command[:, 2])[walking].clamp(
            -MAX_ANG_ACC * dt, MAX_ANG_ACC * dt
        )
        brake = phase == 1
        new[brake, 0] = command[brake, 0] - command[brake, 0].clamp(-BRAKE * dt, BRAKE * dt)
        new[brake, 2] = command[brake, 2] - command[brake, 2].clamp(-MAX_ANG_ACC * dt, MAX_ANG_ACC * dt)
        rot = phase == 2
        w_rot = torch.sign(err) * (1.6 * err.abs()).clamp(min=0.5).clamp(max=rate)
        new[rot] = torch.stack((torch.zeros_like(err), torch.zeros_like(err), w_rot), -1)[rot]
        new[phase >= 4] = 0.0
        new[:, 1] = 0.0
        new[~active] = 0.0
        command = new
        cmd.time_left[:] = 1.0e9
        cmd.vel_command_b[:] = command
        with torch.inference_mode():
            step()

        yaw_now = _yaw(robot.data.root_quat_w.torch)
        spun |= active & (phase == 2) & (_wrap(yaw_now - prev_yaw).abs() / dt > 2.0 * rate)
        prev_yaw = yaw_now
        t_phase += dt
        terminated = active & env.termination_manager.terminated
        fell[terminated] = phase[terminated].clamp(max=DONE - 1)
        fall_at[terminated] = torch.stack((u, v, lane_v), -1)[terminated]
        active &= ~terminated & (phase < DONE)
        if not bool(active.any()):
            break
    return fell, phase, spun, t_rot, fall_at, lane_v


def _pct(k, n):
    return f"{100.0 * k / n:5.1f}%" if n else "    -"


def ice_playback(step, *, dt, args_cli, env_cfg):
    env = _find_env(step)
    cmd = env.command_manager.get_term("base_velocity")
    cmd.cfg.heading_command = False
    cmd.is_heading_env[:] = False
    cmd.is_standing_env[:] = False
    name, sub = next(iter(env_cfg.scene.terrain.terrain_generator.sub_terrains.items()))
    corner = name == "corridor_ice_corner"
    row = _assign_rows(env)
    ids = torch.arange(env.num_envs, device=env.device)
    if corner:  # Groups = what follows the corner; every robot turns 90 deg left
        groups, labels = row, [f"{k:>6}" for k in sub.next_kinds]
        turn = torch.full((env.num_envs,), math.pi / 2.0, device=env.device)
    else:  # Groups = turn angles, both directions
        groups = ids % len(TURNS)
        labels = [f"{a:>4.0f} d" for a in TURNS]
        sign = torch.where((ids // len(TURNS)) % 2 == 0, 1.0, -1.0)
        turn = sign * torch.tensor(TURNS, device=env.device)[groups] * math.pi / 180.0
    print(
        f"[ICE] {env.num_envs} robots, {'ice corner' if corner else 'straight corridor'}, approach {SPEED} m/s, brake "
        f"{BRAKE} m/s^2 from {BRAKE_U} m before the ice cell's centre, {REPEATS} trial(s) per rate",
        flush=True,
    )
    for rate, approach in [(r, a) for a in APPROACHES for r in RATES]:
        rows = [_trial(env, step, rate, dt, sub.start_distance, turn, corner, approach) for _ in range(REPEATS)]
        fell, phase, spun, t_rot, fall_at, lane_v = (torch.cat([r[i] for r in rows]) for i in range(6))
        g = groups.repeat(REPEATS)
        print(f"\n[ICE] rotation rate {rate:.2f} rad/s", flush=True)
        print(
            f"{'group':>6}{'n':>6}{'done':>8}{'fell':>8}   {'in ' + ' / '.join(PHASES):<38}"
            f"{'not done':>9}{'spun':>8}{'t_rot':>9}",
            flush=True,
        )
        for gi, label in enumerate(labels):
            sel = g == gi
            n = int(sel.sum())
            f = fell[sel]
            n_fell = int((f >= 0).sum())
            by_phase = " / ".join(str(int((f == k).sum())) for k in range(len(PHASES)))
            done = int(((phase[sel] == DONE) & (f < 0)).sum())
            tr = t_rot[sel & torch.isfinite(t_rot)]
            tr_s = f"{float(tr.mean()):6.2f} s" if len(tr) else "     - "
            print(
                f"{label}{n:>6}{_pct(done, n):>8}{_pct(n_fell, n):>8}   {by_phase:<38}"
                f"{_pct(n - done - n_fell, n):>9}{_pct(int(spun[sel].sum()), n):>8}{tr_s:>9}",
                flush=True,
            )
        if approach != "centre":  # By lane side (+ = inside of the left turn)
            for side, name in ((LANE, "inside"), (-LANE, "outside")):
                sel = lane_v == side
                n, f = int(sel.sum()), fell[sel]
                fa = fall_at[sel & (fell >= 0)]
                spots = ", ".join(f"({a:+.2f},{b:+.2f})" for a, b, _ in fa[:8].tolist())
                print(
                    f"{'':6}{name:>8} lane: {_pct(int((f >= 0).sum()), n)} fell, rotate {int((f == 2).sum())}; "
                    f"at (u, v): {spots}",
                    flush=True,
                )


play_rsl_rl.run_playback = ice_playback

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--task" not in argv:
        argv = ["--task", "G1-Ice-Test"] + argv
    if "--num_envs" not in argv:  # The shared play mode caps the scene at 50 robots
        argv += ["--num_envs", "500"]
    play_rsl_rl.run(argv)
