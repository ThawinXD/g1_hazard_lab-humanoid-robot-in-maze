"""Challenge 7.5 -- run one team's submission on the hazard maze, with a timer, and score it.

A submission is a folder ``submissions/<team>/`` (upload it with ``scripts/challenge_server.py`` or copy it):
    policy.pt   walking policy checkpoint (rsl_rl ``model_*.pt``, same network as G1 Rough)   required
    pilot.py    your autopilot: ``class Pilot`` (see ``DefaultPilot`` below for the interface)    optional

Levels (--level, or CHALLENGE_LEVEL; see challenge_levels.py):
    7      7 x 7, start bottom-left, goal top-right, 300 s        (practice)
    9      9 x 9, same corners, more hazards, a slalom, 480 s     (practice)
    final  10 x 10, start and goal cells random from the seed, 600 s (the real run)
    --random-ends on 7 / 9: random start and goal there too (practise for the final)

Usage (from the project folder):
    # watch it: Newton viewer window with the timer (the instructor's screen)
    uv run python scripts/challenge.py --team team_a --level final --seed 7 --viz newton
    # score it without a window
    uv run python scripts/challenge.py --team team_a --seed 7
    # try your own checkpoint / pilot before uploading (nothing is written to the leaderboard)
    uv run python scripts/challenge.py --policy logs/rsl_rl/<exp>/<run>/model_399.pt --pilot my_pilot.py --viz newton

Rules (the defaults of this script):
    * the clock starts with the robot standing in the start cell; GOAL = within 0.6 m of the goal-cell centre
    * a fall, a wall touching the arms / torso, being stuck (< 0.25 m in 10 s) or ANY part of the robot inside a
      spike bed below the cone tips (SPIKE) sends the robot back to the start, the clock keeps running
    * after CHALLENGE_TIME seconds (default: the level's limit) the run ends: DNF, ranked by how close it got to the goal (metres
      still to walk along the route it was on, the HUD's "closest"; falling back to the start does not lose it),
      then by fewer falls
    * everyone gets the same maze: the instructor announces the level and the seed at the start of the challenge
    * the pilot only uses what the robot sees during the run (obs): no reading the real map (GENERATED / HAZARDS)
      and no files written in one run and read in the next

Settings (environment variables): CHALLENGE_TIME [s], MAZE_CAM chase | top, MAZE_REC_DIR (frames for a video),
MAZE_REC_MP4 (the run as an .mp4, used by challenge_runner.py for the replays).
"""

import argparse
import contextlib
import csv
import importlib.util
import math
import os
import sys
import time
from collections import deque

# Load imgui_bundle (the viewer's UI, used for the timer) before Isaac Lab (see maze_autopilot.py).
with contextlib.suppress(ImportError):
    import imgui_bundle  # noqa: F401

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

import g1_maze_lab.tasks  # noqa: E402, F401  (registers the G1-Maze-* tasks)
from g1_maze_lab.autopilot import MazeAutopilot  # noqa: E402
from g1_maze_lab.tasks.maze.env_cfg import CELL_SIZE, HARD_CELLS  # noqa: E402
from g1_maze_lab.terrains import GENERATED, HAZARDS, maze_to_ascii  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from challenge_levels import DEFAULT_LEVEL, level_info  # noqa: E402

TIME_LIMIT = 300.0  # set from the level (or CHALLENGE_TIME) in __main__
STUCK_TIME = 10.0  # not moving 0.25 m in this many seconds = a fall
CAMERA = os.environ.get("MAZE_CAM", "chase")
REC_DIR = os.environ.get("MAZE_REC_DIR", "")
REC_MP4 = os.environ.get("MAZE_REC_MP4", "")
REC = bool(REC_DIR or REC_MP4)
REC_EVERY = int(os.environ.get("MAZE_REC_EVERY", 2))
SNAP_AT = os.environ.get("MAZE_SNAP_AT", "")  # e.g. "5,30": save viewer frames (with the timer) at these times [s]
SNAP_DIR = os.environ.get("MAZE_SNAP_DIR", "snapshots")
HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSIONS = os.environ.get("CHALLENGE_DIR", os.path.join(HERE, "..", "submissions"))


class DefaultPilot:
    """The Lab 7.2 autopilot (flood fill + height scan). Copy this class into your ``pilot.py`` and improve it.

    Interface:
        Pilot(num_envs, num_cells, cell_size, device, start_cell, goal_cell)   (cells as (i, j) tuples;
            a pilot whose __init__ has only the first four arguments still runs, but only knows the default corners)
        pilot.act(obs) -> (num_envs, 3) tensor: v_x [m/s], v_y [m/s], w_z [rad/s] in the robot frame
        pilot.reset(env_ids)       called after a fall / crash (the robot is back in the start cell)

    ``obs`` (all torch tensors on ``device``, positions relative to the centre of cell (0, 0): cell (i, j) is
    centred at (i * cell_size, j * cell_size)):
        pos_xy (N, 2), yaw (N,), t (float, seconds since the start), dt (float)
        start_cell (2,), goal_cell (2,)        long tensors: start and goal cell (i, j)
        scan_xy (N, R, 2), scan_z (N, R)       maze height scanner: 33 x 33 points every 0.1 m around the robot
        lidar_dist (N, 72), lidar_yaw (N, 72)  chest-height lidar: range [m] and world yaw of each ray
        lin_vel_b (N, 3), ang_vel_b (N, 3)     base velocity in the robot frame
    """

    def __init__(self, num_envs, num_cells, cell_size, device, start_cell=(0, 0), goal_cell=None):
        # turn_early / min_speed / max_turn: pick the next cell 0.6 m before the centre and walk the corner as an arc
        self.ap = MazeAutopilot(
            num_envs, num_cells, cell_size, device, strategy="flood", speed=0.6,
            turn_early=0.6, min_speed=0.3, max_turn=1.0, start=start_cell, goal=goal_cell,
        )

    def act(self, obs):
        cell = self.ap.cell_of(obs["pos_xy"])
        walls = self.ap.walls_from_height_scan(obs["scan_xy"], obs["scan_z"], cell)
        return self.ap.step(obs["pos_xy"], obs["yaw"], walls, obs["dt"])

    def reset(self, env_ids):
        self.ap.reset(env_ids)


def load_pilot(path):
    if not path:
        return DefaultPilot
    spec = importlib.util.spec_from_file_location("team_pilot", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))  # the pilot may import its own helper files
    spec.loader.exec_module(mod)
    return mod.Pilot


def make_pilot(cls, n_env, n, device, start, goal):
    """Pilot(..., start_cell, goal_cell); an older pilot with four arguments gets only those."""
    import inspect

    params = inspect.signature(cls.__init__).parameters
    if "start_cell" in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return cls(n_env, n, CELL_SIZE, device, start_cell=start, goal_cell=goal)
    if start != (0, 0) or goal != (n - 1, n - 1):
        print("[CHALLENGE] WARNING: this pilot does not take start_cell / goal_cell, it will look for the goal in "
              "the far corner", flush=True)
    return cls(n_env, n, CELL_SIZE, device)


def cells_to_goal(walls, n, start=(0, 0), goal=None):
    """Steps from every cell to the goal along the real maze (BFS on the true walls).

    The way back through the start cell does not count: the maze has two routes that meet only at the start,
    so a robot on the long route is measured along the long route, not "back to the start and the short way".
    """
    goal = (n - 1, n - 1) if goal is None else tuple(goal)
    dist = {goal: 0}
    queue = deque([goal])
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))
    while queue:
        i, j = queue.popleft()
        for d, (di, dj) in enumerate(dirs):
            a, b = i + di, j + dj
            if not walls[i, j, d] and 0 <= a < n and 0 <= b < n and (a, b) not in dist and (a, b) != start:
                dist[(a, b)] = dist[(i, j)] + 1
                queue.append((a, b))
    si, sj = start
    dist[start] = 1 + min(dist[(si + di, sj + dj)] for d, (di, dj) in enumerate(dirs)
                          if not walls[si, sj, d] and (si + di, sj + dj) in dist)
    return dist


def spike_hit(points, beds, top):
    """True when any of ``points`` (B, 3; maze frame, z above the floor) is inside a spike bed below the cone tips."""
    for x0, y0, x1, y1 in beds:
        inside = (points[:, 0] > x0) & (points[:, 0] < x1) & (points[:, 1] > y0) & (points[:, 1] < y1) & (points[:, 2] < top)
        if bool(inside.any()):
            return True
    return False


def route_left_m(p, cell, to_goal, walls):
    """Metres still to walk to the goal along the shortest route, from position ``p`` in ``cell``."""
    left = to_goal.get(cell)
    if left is None:
        return None
    rel = (p[0] - cell[0] * CELL_SIZE, p[1] - cell[1] * CELL_SIZE)
    if left == 0:
        return math.hypot(*rel)
    for d, (di, dj) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
        if not walls[cell[0], cell[1], d] and to_goal.get((cell[0] + di, cell[1] + dj)) == left - 1:
            # to the middle of the edge shared with the next cell, then cell centre to cell centre
            exit_dist = math.hypot(rel[0] - 0.5 * CELL_SIZE * di, rel[1] - 0.5 * CELL_SIZE * dj)
            return (left - 0.5) * CELL_SIZE + min(exit_dist, CELL_SIZE)
    return left * CELL_SIZE


# ----------------------------------------------------------------------------- viewer: timer + camera + pilot map
HUD = {"team": "", "seed": "", "level": "", "why": "FELL / CRASH", "t": 0.0, "left": 0, "falls": 0, "goal": None, "flash": False, "limit": TIME_LIMIT,
       "dist": 0.0, "best": 0.0, "route": 0.0, "pilot": None, "pos_xy": (0.0, 0.0), "yaw": 0.0,
       "show_3d_route": os.environ.get("MAZE_SHOW_ROUTE", "1") == "1",
       "show_lidar": os.environ.get("MAZE_SHOW_LIDAR", "0") == "1",
       "show_scan": os.environ.get("MAZE_SHOW_SCAN", "0") == "1"}
_CAM = {"yaw": None, "follow": True, "c_down": False}

_HAZARD_META = {
    0: ("?", (24, 32, 44, 215)),
    1: ("FLAT", (42, 58, 78, 235)),
    2: ("ICE", (35, 130, 175, 235)),
    3: ("RMP", (38, 130, 108, 235)),
    4: ("SPK", (200, 68, 45, 235)),
    5: ("SLM", (168, 52, 180, 235)),
    6: ("TRN", (112, 55, 195, 235)),
    7: ("STP", (200, 138, 28, 235)),
}


def _hud(imgui):
    io = imgui.get_io()
    imgui.set_next_window_pos(
        imgui.ImVec2(io.display_size.x - 0.18 * io.display_size.x, 12.0), imgui.Cond_.always, imgui.ImVec2(1.0, 0.0)
    )
    imgui.set_next_window_bg_alpha(0.85)
    flags = (
        imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.always_auto_resize
        | imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.no_focus_on_appearing
        | imgui.WindowFlags_.no_nav
    )
    green, orange, red = (imgui.ImVec4(0.48, 0.79, 0.26, 1.0), imgui.ImVec4(0.95, 0.65, 0.35, 1.0),
                          imgui.ImVec4(0.95, 0.3, 0.3, 1.0))
    imgui.begin("challenge timer", None, flags)
    imgui.text_colored(green, f"CHALLENGE 7.5  |  {HUD['team']}  |  level {HUD['level']}  seed {HUD['seed']}")
    imgui.push_font(None, 2.0 * imgui.get_font_size())
    if HUD["goal"] is not None:
        imgui.text_colored(green, HUD["goal"])
    else:
        remaining = HUD["limit"] - HUD["t"]
        imgui.text_colored(red if remaining < 30 else imgui.ImVec4(1, 1, 1, 1), f"{HUD['t']:6.1f} s")
    imgui.pop_font()
    imgui.text(f"to goal {HUD['dist']:4.1f} m   closest {HUD['best']:4.1f} m")
    imgui.text(f"cells to goal {HUD['left']}   falls {HUD['falls']}   limit {HUD['limit']:.0f} s")
    if HUD["flash"]:
        imgui.text_colored(orange, f"{HUD['why']} -> back to start")
    imgui.end()
    _draw_pilot_map(imgui)


def _draw_pilot_map(imgui):
    dbg = HUD.get("pilot")
    if not dbg:
        return
    imgui.set_next_window_pos(imgui.ImVec2(12.0, 12.0), imgui.Cond_.first_use_ever)
    imgui.set_next_window_bg_alpha(0.88)
    win_flags = imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_saved_settings
    imgui.begin("Pilot Saved Map & Route", None, win_flags)

    _, HUD["show_3d_route"] = imgui.checkbox("3D Route", bool(HUD["show_3d_route"]))
    imgui.same_line()
    _, HUD["show_lidar"] = imgui.checkbox("2D LiDAR", bool(HUD["show_lidar"]))
    imgui.same_line()
    _, HUD["show_scan"] = imgui.checkbox("3D Scan", bool(HUD["show_scan"]))

    walls = dbg["walls"]
    ctype = dbg["cell_type"]
    ccost = dbg["cell_cost"]
    cdeaths = dbg["cell_deaths"]
    route = dbg["route"]
    route_set = set(tuple(c) for c in route)
    n = len(walls)
    cell_px = 36.0 if n <= 7 else (30.0 if n <= 9 else 27.0)
    pad = 8.0
    canvas_w = n * cell_px + 2 * pad
    canvas_h = n * cell_px + 2 * pad

    p0 = imgui.get_cursor_screen_pos()
    imgui.dummy(imgui.ImVec2(canvas_w, canvas_h))
    dl = imgui.get_window_draw_list()
    x0, y0 = p0.x + pad, p0.y + pad

    def cell_rect(ci, cj):
        sx = x0 + ci * cell_px
        sy = y0 + (n - 1 - cj) * cell_px
        return sx, sy, sx + cell_px, sy + cell_px

    def world_to_screen(wx, wy):
        sx = x0 + (wx / CELL_SIZE + 0.5) * cell_px
        sy = y0 + (n - 0.5 - wy / CELL_SIZE) * cell_px
        return sx, sy

    for i in range(n):
        for j in range(n):
            x1, y1, x2, y2 = cell_rect(i, j)
            w_ij = walls[i][j]
            explored = any(w != -1 for w in w_ij) or ctype[i][j] != 0
            tag, rgba = _HAZARD_META.get(ctype[i][j], _HAZARD_META[0])
            if not explored and ctype[i][j] == 0:
                rgba = (18, 24, 34, 200)
                tag = ""
            elif ctype[i][j] == 0 and explored:
                rgba = (36, 50, 68, 225)
                tag = ""
            dl.add_rect_filled(imgui.ImVec2(x1, y1), imgui.ImVec2(x2, y2), imgui.IM_COL32(*rgba))

            if (i, j) in route_set:
                dl.add_rect_filled(imgui.ImVec2(x1 + 2, y1 + 2), imgui.ImVec2(x2 - 2, y2 - 2),
                                   imgui.IM_COL32(0, 245, 212, 42))
            if (i, j) == tuple(MAZE["start"]):
                dl.add_rect(imgui.ImVec2(x1 + 2, y1 + 2), imgui.ImVec2(x2 - 2, y2 - 2),
                            imgui.IM_COL32(242, 166, 90, 255), 0.0, 0, 2.0)
                tag = "START"
            elif (i, j) == tuple(MAZE["goal"]):
                dl.add_rect(imgui.ImVec2(x1 + 2, y1 + 2), imgui.ImVec2(x2 - 2, y2 - 2),
                            imgui.IM_COL32(122, 201, 67, 255), 0.0, 0, 2.0)
                tag = "GOAL"

            if cdeaths[i][j] > 0:
                dl.add_rect(imgui.ImVec2(x1 + 1, y1 + 1), imgui.ImVec2(x2 - 1, y2 - 1),
                            imgui.IM_COL32(255, 50, 50, 255), 0.0, 0, 2.5)
                dl.add_text(imgui.ImVec2(x1 + 3, y2 - 14), imgui.IM_COL32(255, 110, 110, 255),
                            f"X{int(cdeaths[i][j])}")
            elif explored and ccost[i][j] > 1.1:
                dl.add_text(imgui.ImVec2(x1 + 3, y2 - 14), imgui.IM_COL32(210, 220, 230, 190),
                            f"{ccost[i][j]:.0f}")
            if tag:
                dl.add_text(imgui.ImVec2(x1 + 3, y1 + 2), imgui.IM_COL32(245, 245, 240, 245), tag[:4])

    col_wall = imgui.IM_COL32(245, 243, 235, 255)
    col_unk = imgui.IM_COL32(90, 105, 125, 95)
    for i in range(n):
        for j in range(n):
            x1, y1, x2, y2 = cell_rect(i, j)
            w_ij = walls[i][j]
            edges = (
                (w_ij[0], (x2, y1), (x2, y2)),
                (w_ij[1], (x1, y1), (x2, y1)),
                (w_ij[2], (x1, y1), (x1, y2)),
                (w_ij[3], (x1, y2), (x2, y2)),
            )
            for state, a, b in edges:
                if state == 1:
                    dl.add_line(imgui.ImVec2(*a), imgui.ImVec2(*b), col_wall, 3.0)
                elif state == -1:
                    dl.add_line(imgui.ImVec2(*a), imgui.ImVec2(*b), col_unk, 1.0)

    col_route = imgui.IM_COL32(0, 245, 212, 245)
    for k in range(len(route) - 1):
        ax, ay = world_to_screen(route[k][0] * CELL_SIZE, route[k][1] * CELL_SIZE)
        bx, by = world_to_screen(route[k + 1][0] * CELL_SIZE, route[k + 1][1] * CELL_SIZE)
        dl.add_line(imgui.ImVec2(ax, ay), imgui.ImVec2(bx, by), col_route, 3.2)
        dl.add_circle_filled(imgui.ImVec2(bx, by), 3.2, col_route)

    rx, ry = world_to_screen(HUD["pos_xy"][0], HUD["pos_xy"][1])
    wpx, wpy = world_to_screen(dbg["waypoint"][0], dbg["waypoint"][1])
    col_wp = imgui.IM_COL32(255, 225, 55, 255)
    dl.add_line(imgui.ImVec2(rx, ry), imgui.ImVec2(wpx, wpy), col_wp, 2.0)
    dl.add_circle_filled(imgui.ImVec2(wpx, wpy), 4.5, col_wp)

    yaw = float(HUD["yaw"])
    hx = rx + 11.0 * math.cos(yaw)
    hy = ry - 11.0 * math.sin(yaw)
    dl.add_circle_filled(imgui.ImVec2(rx, ry), 5.5, imgui.IM_COL32(122, 201, 67, 255))
    dl.add_line(imgui.ImVec2(rx, ry), imgui.ImVec2(hx, hy), imgui.IM_COL32(255, 255, 255, 255), 2.5)

    cur_c = tuple(dbg["current_cell"])
    tgt_c = tuple(dbg["target_cell"])
    cur_tag = _HAZARD_META.get(ctype[cur_c[0]][cur_c[1]], ("?",))[0]
    tgt_tag = _HAZARD_META.get(ctype[tgt_c[0]][tgt_c[1]], ("?",))[0]
    dist_goal = dbg["dist"][cur_c[0]][cur_c[1]]
    cost_str = f"{dist_goal:.1f}" if dist_goal < 1e5 else "inf"
    vx, vy, wz = dbg["cmd"]
    imgui.text(f"Cell {cur_c} [{cur_tag}] -> Target {tgt_c} [{tgt_tag}]")
    imgui.text(f"Planned route: {max(len(route) - 1, 0)} cells (cost {cost_str})")
    imgui.text(f"Cmd: vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}" + ("  [UNSTUCK]" if dbg["unstuck"] else ""))
    imgui.text_colored(imgui.ImVec4(0.7, 0.78, 0.86, 1.0), "Cyan=Route  Yellow=Waypoint  Red=Spike/Death")
    imgui.end()


def _find_env(step):
    for cell in step.__closure__ or ():
        obj = cell.cell_contents
        if hasattr(obj, "unwrapped") and hasattr(obj.unwrapped, "command_manager"):
            return obj.unwrapped
    raise RuntimeError("Could not find the environment in the play loop.")


def _find_viewer():
    from isaaclab.sim import SimulationContext

    for viz in getattr(SimulationContext.instance(), "visualizers", []):
        viewer = getattr(viz, "_viewer", None)
        if viewer is not None and hasattr(viewer, "is_key_down"):
            return viewer
    return None


def _yaw(q):
    x, y, z, w = q.unbind(-1)  # xyzw
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _set_camera(viewer, robot, origin):
    """``origin`` = world position of the centre of cell (0, 0)."""
    if CAMERA == "top":
        n = MAZE["n"]
        c = (origin[:2] + 0.5 * (n - 1) * CELL_SIZE).tolist()
        viewer.set_camera(wp.vec3(c[0], c[1] - 4.5 * n / 7, 17.0 * n / 7), -75.0, 90.0)
        return
    if CAMERA.startswith("look:"):  # look:i,j = fixed camera on cell (i, j), from above and to the west
        i, j = (int(v) for v in CAMERA[5:].split(","))
        c = (origin[:2] + torch.tensor((i, j), device=origin.device) * CELL_SIZE).tolist()
        viewer.set_camera(wp.vec3(c[0] - 1.6, c[1], 2.6), -55.0, 0.0)
        return
    c_down = viewer.is_key_down("c")  # C = free camera / follow again
    if c_down and not _CAM["c_down"]:
        _CAM["follow"] = not _CAM["follow"]
    _CAM["c_down"] = c_down
    if not _CAM["follow"]:
        return
    p = robot.data.root_pos_w.torch[0].tolist()
    yaw = float(_yaw(robot.data.root_quat_w.torch[0]))
    if _CAM["yaw"] is None:
        _CAM["yaw"] = yaw
    _CAM["yaw"] += 0.05 * math.atan2(math.sin(yaw - _CAM["yaw"]), math.cos(yaw - _CAM["yaw"]))
    cy = _CAM["yaw"]
    viewer.set_camera(wp.vec3(p[0] - 3.2 * math.cos(cy), p[1] - 3.2 * math.sin(cy), 2.6), -28.0, math.degrees(cy))


def _markers(env):
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ChallengeGoal",
        markers={
            name: sim_utils.CuboidCfg(size=(1.2, 1.2, 0.02), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=col))
            for name, col in (("goal", (0.48, 0.79, 0.26)), ("start", (0.95, 0.65, 0.35)))
        },
    )
    m = VisualizationMarkers(cfg)
    o = env.scene.env_origins[:1]  # start-cell centre
    goal = o.clone()
    goal[0, 0] += (MAZE["goal"][0] - MAZE["start"][0]) * CELL_SIZE
    goal[0, 1] += (MAZE["goal"][1] - MAZE["start"][1]) * CELL_SIZE
    m.visualize(translations=torch.cat((goal, o)) + 0.025, marker_indices=torch.tensor([0, 1], device=env.device))
    return m


class _RouteMarkers:
    """3D floor markers showing the robot's planned Dijkstra route, active local waypoint, and discovered hazards."""

    def __init__(self, env):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PilotRoute",
            markers={
                "route": sim_utils.CylinderCfg(
                    radius=0.20, height=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.96, 0.83)),
                ),
                "waypoint": sim_utils.SphereCfg(
                    radius=0.13,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.88, 0.15)),
                ),
                "hazard": sim_utils.CuboidCfg(
                    size=(0.42, 0.42, 0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.32, 0.18)),
                ),
                "death": sim_utils.CuboidCfg(
                    size=(0.55, 0.55, 0.12),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.12, 0.25)),
                ),
            },
        )
        self.markers = VisualizationMarkers(cfg)
        self.device = env.device

    def update(self, origin_w: torch.Tensor, dbg: dict | None, visible: bool = True):
        if not dbg or not visible:
            self.markers.set_visibility(False)
            return
        self.markers.set_visibility(True)
        pts, idxs = [], []
        ox, oy, oz = float(origin_w[0].item()), float(origin_w[1].item()), float(origin_w[2].item())
        for ci, cj in dbg["route"]:
            pts.append((ox + ci * CELL_SIZE, oy + cj * CELL_SIZE, oz + 0.04))
            idxs.append(0)
        wpx, wpy = dbg["waypoint"]
        pts.append((ox + wpx, oy + wpy, oz + 0.22))
        idxs.append(1)
        ctype = dbg["cell_type"]
        cdeaths = dbg["cell_deaths"]
        n = len(ctype)
        for i in range(n):
            for j in range(n):
                if cdeaths[i][j] > 0:
                    pts.append((ox + i * CELL_SIZE, oy + j * CELL_SIZE, oz + 0.08))
                    idxs.append(3)
                elif ctype[i][j] >= 3:
                    pts.append((ox + i * CELL_SIZE, oy + j * CELL_SIZE, oz + 0.05))
                    idxs.append(2)
        if pts:
            t_pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
            t_idx = torch.tensor(idxs, dtype=torch.long, device=self.device)
            self.markers.visualize(translations=t_pts, marker_indices=t_idx)


def _extract_pilot_debug(pilot, cell, n: int) -> dict | None:
    if hasattr(pilot, "get_debug_snapshot"):
        return pilot.get_debug_snapshot(0)
    ap = getattr(pilot, "ap", None)
    if ap is None or not hasattr(ap, "walls"):
        return None
    walls = ap.walls[0].detach().cpu().tolist()
    dist = ap.flood_fill(ap.walls)[0].detach().cpu().tolist()
    tgt = tuple(int(v) for v in ap.target[0].detach().cpu().tolist())
    return {
        "walls": walls,
        "cell_type": [[0] * n for _ in range(n)],
        "cell_cost": [[1.0] * n for _ in range(n)],
        "cell_deaths": [[0.0] * n for _ in range(n)],
        "safe_offset": [[[0.0, 0.0] for _ in range(n)] for _ in range(n)],
        "dist": dist,
        "route": [cell, tgt],
        "current_cell": cell,
        "target_cell": tgt,
        "waypoint": [tgt[0] * CELL_SIZE, tgt[1] * CELL_SIZE],
        "cmd": ap.cmd[0].detach().cpu().tolist() if hasattr(ap, "cmd") else [0.0, 0.0, 0.0],
        "unstuck": False,
    }


class _Recorder:
    """1280x720 frames with the timer drawn on them (a video of the run). The viewer frame is read here, drawing and
    saving run on a worker thread so the run does not slow down much."""

    def __init__(self):
        import queue
        import threading

        from PIL import ImageFont

        try:
            self.big = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
            self.small = ImageFont.truetype("DejaVuSans.ttf", 22)
        except OSError:
            self.big = self.small = ImageFont.load_default()
        self.ffmpeg = None
        if REC_MP4:  # raw frames -> ffmpeg (H.264, 25 fps = every 2nd 50 Hz step)
            import subprocess

            import imageio_ffmpeg

            self.ffmpeg = subprocess.Popen(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", "1280x720", "-framerate", str(50 // REC_EVERY), "-i", "-", "-c:v", "libx264", "-preset",
                 "veryfast", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", REC_MP4],
                stdin=subprocess.PIPE,
            )
        else:
            os.makedirs(REC_DIR, exist_ok=True)
        self.q = queue.Queue(maxsize=64)
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def add(self, viewer, index):
        import numpy as np

        img = np.asarray(viewer.get_frame(render_ui=False).numpy())
        self.q.put((index, img, dict(HUD)))

    def _work(self):
        while (item := self.q.get()) is not None:
            index, img, hud = item
            im = self._draw(img, hud)
            if self.ffmpeg is not None:
                self.ffmpeg.stdin.write(im.tobytes())
            else:
                im.save(os.path.join(REC_DIR, f"{index:05d}.jpg"), quality=90)

    def close(self):
        self.q.put(None)
        self.worker.join()
        if self.ffmpeg is not None:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()

    def _draw(self, img, hud):
        import numpy as np
        from PIL import Image, ImageDraw

        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(img[..., :3]).resize((1280, 720))
        d = ImageDraw.Draw(im, "RGBA")
        big, small = self.big, self.small
        d.rounded_rectangle((16, 16, 470, 182), 14, fill=(14, 22, 33, 215))
        d.text((32, 24), f"CHALLENGE 7.5 · {hud['team']} · L{hud['level']} · seed {hud['seed']}", font=small,
               fill=(122, 201, 67))
        d.text((32, 54), f"{hud['t']:6.1f} s", font=big, fill=(244, 243, 238))
        d.text((32, 108), f"to goal {hud['dist']:4.1f} m   closest {hud['best']:4.1f} m", font=small,
               fill=(244, 243, 238))
        d.text(
            (32, 140),
            f"cells to goal {hud['left']}   falls {hud['falls']}",
            font=small,
            fill=(242, 166, 90) if hud["falls"] else (184, 196, 208),
        )
        if hud["flash"]:
            d.rounded_rectangle((500, 16, 940, 80), 14, fill=(184, 40, 40, 220))
            d.text((522, 32), f"{hud['why']} -> back to start", font=small, fill=(255, 255, 255))
        if hud["goal"] is not None:
            d.rounded_rectangle((500, 16, 900, 84), 14, fill=(61, 122, 18, 230))
            d.text((522, 26), hud["goal"], font=big, fill=(255, 255, 255))
        return im


def _snap(viewer, t):
    import numpy as np
    from PIL import Image

    os.makedirs(SNAP_DIR, exist_ok=True)
    img = np.asarray(viewer.get_frame(render_ui=os.environ.get("MAZE_SNAP_UI", "1") == "1").numpy())
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    path = os.path.join(SNAP_DIR, f"challenge_{CAMERA}_{t:05.1f}s.png")
    Image.fromarray(img[..., :3]).save(path)
    print(f"[CHALLENGE] saved {path}", flush=True)


# ----------------------------------------------------------------------------- the run
RUN = {"team": "", "seed": 0, "level": DEFAULT_LEVEL, "pilot": "", "policy": "", "score": True}
MAZE = {"n": HARD_CELLS, "start": (0, 0), "goal": (HARD_CELLS - 1, HARD_CELLS - 1)}


def challenge_playback(step, *, dt, args_cli, env_cfg):
    env = _find_env(step)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("base_velocity")
    cmd.cfg.heading_command = False
    cmd.is_heading_env[:] = False
    cmd.is_standing_env[:] = False
    scanner, lidar = env.scene["maze_scanner"], env.scene["wall_lidar"]
    scanner.set_debug_vis(True)  # show the lidar rays in the viewer
    walls, hazards = GENERATED[-1]["walls"], HAZARDS[-1]
    n_env, n = env.num_envs, walls.shape[0]
    start, goal = tuple(hazards["start"]), tuple(hazards["goal"])
    MAZE.update(n=n, start=start, goal=goal)
    # the robot that reaches the goal stays there (the run is over)
    env.termination_manager.get_term_cfg("goal_reached").func = lambda env, **kw: torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )

    to_goal = cells_to_goal(walls, n, start, goal)
    beds = [tuple(b[3:]) for b in hazards["spike"]]
    spike_top = hazards["spike_top"] + 0.02
    print(f"[CHALLENGE] level {RUN['level']} seed {RUN['seed']}  maze {n}x{n}, start {start} goal {goal}, "
          f"shortest route {to_goal[start]} cells", flush=True)
    print("[CHALLENGE] hazards: " + ", ".join(f"{k} {len(v)}" for k, v in hazards.items()
                                              if isinstance(v, list)), flush=True)
    if os.environ.get("CHALLENGE_SHOW_MAZE") == "1":  # instructor only: print the solution maze
        print(maze_to_ascii(walls), "\n", hazards, flush=True)

    pilot = make_pilot(load_pilot(RUN["pilot"]), n_env, n, env.device, start, goal)
    viewer = _find_viewer()
    route_markers = None
    last_lidar_vis = last_scan_vis = False
    if viewer is not None:
        markers = _markers(env)  # noqa: F841
        route_markers = _RouteMarkers(env)
        viewer.register_ui_callback(_hud, position="free")
    HUD.update(team=RUN["team"] or "(test)", seed=RUN["seed"], level=RUN["level"])
    recorder = _Recorder() if viewer is not None and REC else None

    stuck_ref = (0.0, 0.0)
    snap_steps = {int(round(float(x) / dt)) for x in SNAP_AT.split(",")} if SNAP_AT else set()
    falls, best_left, finish, cell = 0, to_goal[start], None, start
    route_m = to_goal[start] * CELL_SIZE  # start-cell centre -> goal-cell centre along the shortest route
    start_t = torch.tensor(start, dtype=torch.long, device=env.device)
    goal_t = torch.tensor(goal, dtype=torch.long, device=env.device)
    why = "FELL / CRASH"
    dist_m = best_m = route_m  # metres still to walk to the goal: now / the least so far
    last_fall_t = -10.0
    wall_t0 = time.time()
    steps = int(TIME_LIMIT / dt)
    for k in range(steps):
        o = env.scene.env_origins.clone()  # -> centre of cell (0, 0): positions in the maze frame
        o[:, :2] -= start_t.float() * CELL_SIZE
        pos_xy = robot.data.root_pos_w.torch[:, :2] - o[:, :2]
        obs = {
            "pos_xy": pos_xy,
            "yaw": _yaw(robot.data.root_quat_w.torch),
            "t": k * dt,
            "dt": dt,
            "start_cell": start_t,
            "goal_cell": goal_t,
            "scan_xy": torch.nan_to_num(scanner.data.ray_hits_w.torch[..., :2], nan=0.0) - o[:, None, :2],
            "scan_z": torch.nan_to_num(scanner.data.ray_hits_w.torch[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
            - o[:, None, 2],
            "lin_vel_b": robot.data.root_lin_vel_b.torch,
            "ang_vel_b": robot.data.root_ang_vel_b.torch,
        }
        rel = lidar.data.ray_hits_w.torch[..., :2] - lidar.data.pos_w.torch[:, None, :2]
        obs["lidar_dist"] = torch.nan_to_num(torch.linalg.norm(rel, dim=-1), nan=1e6, posinf=1e6)
        obs["lidar_yaw"] = torch.atan2(rel[..., 1], rel[..., 0])

        command = torch.as_tensor(pilot.act(obs), dtype=torch.float32, device=env.device).reshape(n_env, 3)
        if finish is not None:
            command[:] = 0.0
        cmd.time_left[:] = 1.0e9
        cmd.vel_command_b[:] = command

        if viewer is not None:
            if not viewer.is_running():
                break
            _set_camera(viewer, robot, o[0])
        with torch.inference_mode():
            step()

        t = (k + 1) * dt
        prev_cell = cell
        p = (robot.data.root_pos_w.torch[0, :2] - o[0, :2]).tolist()
        cell = (min(max(round(p[0] / CELL_SIZE), 0), n - 1), min(max(round(p[1] / CELL_SIZE), 0), n - 1))
        left = to_goal.get(cell, best_left)
        best_left = min(best_left, left)
        rem = route_left_m(p, cell, to_goal, walls)
        if rem is not None and finish is None:
            dist_m = rem
            best_m = min(best_m, dist_m)
        if finish is None and math.dist(p, (goal[0] * CELL_SIZE, goal[1] * CELL_SIZE)) < 0.6:
            finish = t
            dist_m = best_m = 0.0
            print(f"[CHALLENGE] GOAL at {t:.1f} s, falls {falls}", flush=True)
        # stuck (e.g. a leg wedged in a trench): moved less than 0.25 m in STUCK_TIME s -> counts as a fall
        if k % int(STUCK_TIME / dt) == 0:
            if finish is None and k > 0 and math.dist(p, stuck_ref) < 0.25:
                with torch.inference_mode():  # same as a termination: reset the robot to the start
                    env._reset_idx(torch.arange(n_env, device=env.device))
                    env.reset_buf[:] = True
                print(f"[CHALLENGE] t={t:.1f}s stuck for {STUCK_TIME:.0f} s in cell {cell}", flush=True)
                why = "STUCK"
                p = [start[0] * CELL_SIZE, start[1] * CELL_SIZE]  # back at the start
            stuck_ref = p
        # spikes are deadly: any body of the robot inside a bed, lower than the cone tips
        if finish is None and beds and not bool(env.reset_buf[0]):
            bodies = robot.data.body_pos_w.torch[0].clone()
            bodies[:, :2] -= o[0, :2]
            bodies[:, 2] -= o[0, 2]
            if spike_hit(bodies, beds, spike_top):
                with torch.inference_mode():
                    env._reset_idx(torch.arange(n_env, device=env.device))
                    env.reset_buf[:] = True
                print(f"[CHALLENGE] t={t:.1f}s SPIKE in cell {cell}", flush=True)
                why = "SPIKE"
                stuck_ref = [start[0] * CELL_SIZE, start[1] * CELL_SIZE]
        if finish is None and bool(env.reset_buf[0]) and k + 1 < steps:  # fell / crashed / stuck: back to start
            falls += 1
            last_fall_t = t
            pilot.reset(torch.arange(n_env, device=env.device))
            print(f"[CHALLENGE] t={t:.1f}s fall/crash #{falls} in cell {prev_cell} -> back to start", flush=True)
            HUD["why"], why = why, "FELL / CRASH"
        if viewer is not None and k % 2 == 0:
            dbg = _extract_pilot_debug(pilot, cell, n)
            HUD["pilot"] = dbg
            HUD["pos_xy"] = p
            HUD["yaw"] = float(obs["yaw"][0].item())
            if route_markers is not None:
                route_markers.update(o[0], dbg, bool(HUD["show_3d_route"]))
            if bool(HUD["show_lidar"]) != last_lidar_vis:
                last_lidar_vis = bool(HUD["show_lidar"])
                lidar.set_debug_vis(last_lidar_vis)
            if bool(HUD["show_scan"]) != last_scan_vis:
                last_scan_vis = bool(HUD["show_scan"])
                scanner.set_debug_vis(last_scan_vis)
        HUD.update(
            t=finish if finish is not None else t,
            left=0 if finish is not None else left,
            falls=falls,
            goal=None if finish is None else f"GOAL  {finish:.1f} s",
            flash=t - last_fall_t < 1.5,
            dist=dist_m,
            best=best_m,
            route=route_m,
        )
        if recorder is not None and k % REC_EVERY == 0:
            recorder.add(viewer, k // REC_EVERY)
        if viewer is not None and (k + 1 in snap_steps or (SNAP_AT and finish is not None and 0 <= t - finish - 1.5 < dt)):
            _snap(viewer, t)  # MAZE_SNAP_AT also saves a frame 1.5 s after the goal
        if finish is not None:
            if viewer is None or REC:
                if REC and t < finish + 3.0:
                    continue
                break  # headless: done
        if k % int(10.0 / dt) == 0:
            print(
                f"[CHALLENGE] t={t:5.1f}s cell={cell} cells to goal {left} to goal {dist_m:.1f} m "
                f"(closest {best_m:.1f} m, shortest route {route_m:.0f} m) falls {falls}",
                flush=True,
            )

    result = "GOAL" if finish is not None else "DNF"
    print(
        f"[CHALLENGE] RESULT team={RUN['team'] or '(test)'} seed={RUN['seed']}: {result}"
        + (f" {finish:.1f} s" if finish is not None else f" (closest {best_m:.1f} m to the goal, "
           f"{best_left} cells; shortest route {route_m:.0f} m)")
        + f", falls {falls}  (wall clock {time.time() - wall_t0:.0f} s)",
        flush=True,
    )
    if RUN["score"]:
        path = os.path.join(SUBMISSIONS, "leaderboard.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["team", "seed", "result", "time_s", "cells_to_goal", "falls", "finished_at", "best_left_m",
                            "route_m", "level"])
            w.writerow(
                [RUN["team"], RUN["seed"], result, f"{finish:.1f}" if finish is not None else "", best_left if finish is None else 0,
                 falls, time.strftime("%Y-%m-%d %H:%M:%S"), f"{best_m:.1f}", f"{route_m:.0f}", RUN["level"]]
            )
        print(f"[CHALLENGE] written to {os.path.abspath(path)}", flush=True)
    if recorder is not None:
        recorder.close()
    if viewer is not None and finish is not None and not REC:
        while viewer.is_running():  # keep the window on the finished run until it is closed
            with torch.inference_mode():
                step()


play_rsl_rl.run_playback = challenge_playback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--team", default="")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("CHALLENGE_SEED", 2026)))
    parser.add_argument("--policy", default="")
    parser.add_argument("--pilot", default="")
    parser.add_argument("--level", default=os.environ.get("CHALLENGE_LEVEL", DEFAULT_LEVEL))
    parser.add_argument("--random-ends", action="store_true", help="random start / goal on level 7 or 9 too")
    ours, rest = parser.parse_known_args(sys.argv[1:])
    lv = level_info(ours.level)
    TIME_LIMIT = float(os.environ.get("CHALLENGE_TIME", lv["time"]))
    HUD["limit"] = TIME_LIMIT
    if ours.team:
        folder = os.path.join(SUBMISSIONS, ours.team)
        ours.policy = ours.policy or os.path.join(folder, "policy.pt")
        if not ours.pilot and os.path.exists(os.path.join(folder, "pilot.py")):
            ours.pilot = os.path.join(folder, "pilot.py")
    if not ours.pilot and not ours.team and os.path.exists(os.path.join(HERE, "..", "pilot.py")):
        ours.pilot = os.path.join(HERE, "..", "pilot.py")
    if not ours.policy:
        ours.policy = os.path.join(HERE, "..", "pretrained", "g1_rough", "model_pretrained.pt")
    RUN.update(team=ours.team, seed=ours.seed, level=ours.level, pilot=ours.pilot, policy=ours.policy,
               score=bool(ours.team))
    print(f"[CHALLENGE] policy {ours.policy}  pilot {ours.pilot or '(default flood autopilot)'}", flush=True)
    play_rsl_rl.run(
        ["--task", lv["task"], "--num_envs", "1", "--checkpoint", ours.policy]
        + rest
        + [f"env.scene.terrain.terrain_generator.seed={ours.seed}"]
        + (["env.scene.terrain.terrain_generator.sub_terrains.hazard_maze.random_ends=true"] if ours.random_ends else [])
    )
