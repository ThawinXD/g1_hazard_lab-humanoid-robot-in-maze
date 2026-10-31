"""Barebone Pilot for Challenge 7.5 (G1 Hazard Maze).

Run locally with the challenge script:
    uv run python scripts/challenge.py --pilot pilot.py --seed 1 --viz newton
    uv run python scripts/challenge.py --level 9 --pilot pilot.py --seed 1 --viz newton

Or with a custom fine-tuned walking policy:
    uv run python scripts/challenge.py --policy logs/rsl_rl/g1_hazard_starter/<run>/model_XXXX.pt --pilot pilot.py --seed 1 --viz newton

Observation dictionary ``obs`` passed to ``pilot.act(obs)`` (all tensors on ``self.device``,
positions relative to the centre of cell (0, 0) where cell (i, j) is at (i * cell_size, j * cell_size)):
    - pos_xy:     (N, 2)   robot base (x, y) in maze frame [m]
    - yaw:        (N,)     robot heading yaw in maze frame [rad]
    - t:          float    elapsed time since start [s]
    - dt:         float    policy step timestep [s]
    - start_cell: (2,)     long tensor (i, j) of start cell
    - goal_cell:  (2,)     long tensor (i, j) of goal cell
    - scan_xy:    (N, R, 2) height-scanner ray hit (x, y) in maze frame [m] (33x33 grid, 0.1 m spacing)
    - scan_z:     (N, R)   height-scanner ray hit height z relative to floor [m]
    - lidar_dist: (N, 72)  horizontal chest lidar range [m]
    - lidar_yaw:  (N, 72)  world yaw angle of each lidar ray [rad]
    - lin_vel_b:  (N, 3)   linear velocity in robot base frame [m/s]
    - ang_vel_b:  (N, 3)   angular velocity in robot base frame [rad/s]
"""

from __future__ import annotations

import torch

from g1_maze_lab.autopilot import DIRS, MazeAutopilot


class Pilot:
    """Barebone maze pilot with hooks for hazard perception, route planning, and velocity control."""

    def __init__(
        self,
        num_envs: int,
        num_cells: int,
        cell_size: float,
        device: str,
        start_cell: tuple[int, int] = (0, 0),
        goal_cell: tuple[int, int] | None = None,
    ):
        self.num_envs = num_envs
        self.num_cells = num_cells
        self.cell_size = cell_size
        self.device = device
        self.start_cell = tuple(int(v) for v in start_cell)
        self.goal_cell = (
            (num_cells - 1, num_cells - 1) if goal_cell is None else tuple(int(v) for v in goal_cell)
        )

        # Base flood-fill autopilot (turn_early=0.6 m walks corners smoothly as an arc)
        self.ap = MazeAutopilot(
            num_envs=num_envs,
            num_cells=num_cells,
            cell_size=cell_size,
            device=device,
            strategy="flood",
            speed=0.6,
            turn_early=0.6,
            min_speed=0.3,
            max_turn=1.0,
            start=self.start_cell,
            goal=self.goal_cell,
        )

    # ------------------------------------------------------------------
    # 1. Perception
    # ------------------------------------------------------------------
    def sense_walls(self, obs: dict, cell: torch.Tensor) -> torch.Tensor:
        """Detect blocked sides (N, 4) of the current cell from the height scan.
        Directions: 0 = +x (east), 1 = +y (north), 2 = -x (west), 3 = -y (south).
        """
        return self.ap.walls_from_height_scan(obs["scan_xy"], obs["scan_z"], cell)

    # ------------------------------------------------------------------
    # 2. Planning & Control Step
    # ------------------------------------------------------------------
    def act(self, obs: dict) -> torch.Tensor:
        """Compute velocity command (num_envs, 3) -> [v_x, v_y, w_z] in robot frame."""
        pos_xy = obs["pos_xy"]
        yaw = obs["yaw"]
        dt = obs["dt"]

        cell = self.ap.cell_of(pos_xy)
        walls_now = self.sense_walls(obs, cell)

        # TODO: Add hazard detection from obs["scan_xy"] and obs["scan_z"]
        # (e.g., trenches z < -0.05, ramps/spikes 0.10 < z < 0.40) and adjust target/velocity.
        cmd = self.ap.step(pos_xy, yaw, walls_now, dt)
        return cmd

    def reset(self, env_ids: torch.Tensor):
        """Called when the robot falls, hits a spike bed, or gets stuck and resets to start."""
        self.ap.reset(env_ids)
        # Note: if you want the robot to remember walls/hazards discovered before falling,
        # you can preserve self.ap.walls across resets instead of clearing them.
        _ = DIRS
