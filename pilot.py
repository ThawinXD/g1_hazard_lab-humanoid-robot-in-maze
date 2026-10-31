"""Hazard-Aware Pilot for Challenge 7.5 (G1 Hazard Maze) -- Option B.

Combines four layers on top of ``MazeAutopilot``:
1. **Multi-Cell Hazard Perception**: classifies walls, trenches (gaps), steep vs. low ramps,
   single-lane spike beds, slalom spike beds, rough boxes, and ice from ``scan_xy`` / ``scan_z`` + ``lidar``.
2. **Persistent Map & Failure Attribution**: preserves discovered walls and hazard costs across resets
   (falls / spike hits / stuck resets) and adds a +50 cost penalty to any cell where a reset occurred.
3. **Hazard-Weighted Dijkstra Planner**: propagates true traversal costs to the goal so the robot
   automatically pivots from the Short Route to the Long Route when hard hazards block the Short Route
   (or after a single reset).
4. **Safe-Lane Waypoint, Slalom Crossover & Anti-Stuck Controller**: shifts waypoints into the +-0.48 m
   free lane around spike beds, executes a gap-centered lane change in slalom cells, adapts speed and
   lateral strafing (``v_y``) to each terrain type, and triggers a reverse/turn reflex if stuck for 3.5 s.
"""

from __future__ import annotations  # Enable postponed evaluation of type annotations

import math  # Standard math functions (cos, sin, atan2) for heading and frame transforms
import os  # Read environment variables (PILOT_MODE, PILOT_SPEED) for runtime tuning

import torch  # PyTorch for GPU/CPU tensor operations

from g1_maze_lab.autopilot import DIRS, MazeAutopilot  # Base maze solver and cardinal direction offsets

# Integer IDs representing the detected floor category of each maze cell
CELL_UNKNOWN = 0  # Cell has not been scanned yet
CELL_PLAIN = 1  # Flat, normal-friction corridor floor
CELL_ICE = 2  # Slippery ice tile (1 cm raised step, low friction)
CELL_ROUGH = 3  # Cell containing loose 2-8 cm boxes on the floor
CELL_LOW_RAMP = 4  # Low trapezoid ramp (0.10-0.18 m high, found on the Long Route)
CELL_SPIKE_LANE = 5  # Single spike bed on one side of the corridor (leaves a ~1 m free lane)
CELL_SPIKE_SLALOM = 6  # Two staggered spike beds on opposite sides with a 0.6 m crossover gap
CELL_TRENCH = 7  # Deep trench/gap across the corridor (0.15-0.22 m wide, 1.0 m deep)
CELL_STEEP_RAMP = 8  # Steep trapezoid ramp (0.24-0.30 m high, found on the Short Route)

# Traversal cost table for "auto" mode: penalizes Short Route traps so the planner pivots to the Long Route
COSTS_AUTO = {
    CELL_UNKNOWN: 1.05,  # Slight penalty for unexplored cells compared to known plain floor
    CELL_PLAIN: 1.0,  # Baseline cost of walking through 1 plain cell
    CELL_ICE: 1.2,  # Small penalty for slippery ice tiles
    CELL_ROUGH: 1.35,  # Moderate penalty for loose boxes (also helps detect Short Route rough fill)
    CELL_LOW_RAMP: 1.45,  # Small penalty for easily climbable Long Route ramps
    CELL_SPIKE_LANE: 1.75,  # Moderate cost for single-lane spikes (safe once shifted into free lane)
    CELL_SPIKE_SLALOM: 16.0,  # High cost so the robot avoids Short Route slaloms if Long Route is open
    CELL_TRENCH: 18.0,  # High cost so the robot avoids Short Route trenches if Long Route is open
    CELL_STEEP_RAMP: 22.0,  # High cost so the robot avoids Short Route steep ramps if Long Route is open
}

# Traversal cost table for "short" mode: attempts the Short Route first, falls back to Long Route on reset
COSTS_SHORT = {
    CELL_UNKNOWN: 1.05,  # Unexplored cell cost
    CELL_PLAIN: 1.0,  # Plain cell cost
    CELL_ICE: 1.15,  # Ice cell cost
    CELL_ROUGH: 1.2,  # Rough box cell cost
    CELL_LOW_RAMP: 1.3,  # Low ramp cell cost
    CELL_SPIKE_LANE: 1.5,  # Single-lane spike cost
    CELL_SPIKE_SLALOM: 3.5,  # Low enough to attempt slalom on Short Route (< 6-cell detour)
    CELL_TRENCH: 2.5,  # Low enough to attempt trench on Short Route
    CELL_STEEP_RAMP: 3.0,  # Low enough to attempt steep ramp on Short Route
}


class Pilot:
    """Hazard-aware maze autopilot for Challenge 7.5."""

    def __init__(
        self,
        num_envs: int,  # Number of parallel robot environments (1 in challenge evaluation)
        num_cells: int,  # Maze grid dimension N (7, 9, or 10)
        cell_size: float,  # Width of each square cell in meters (2.0 m)
        device: str,  # PyTorch device ("cuda:0" or "cpu")
        start_cell: tuple[int, int] = (0, 0),  # (i, j) indices of the start cell
        goal_cell: tuple[int, int] | None = None,  # (i, j) indices of the goal cell
    ):
        self.num_envs = num_envs  # Store number of environments
        self.num_cells = num_cells  # Store grid size N
        self.cell_size = cell_size  # Store cell size (2.0 m)
        self.device = device  # Store compute device string
        self.start_cell = tuple(int(v) for v in start_cell)  # Normalize start cell to (int, int) tuple
        self.goal_cell = (  # Default goal to top-right corner (N-1, N-1) if not provided
            (num_cells - 1, num_cells - 1) if goal_cell is None else tuple(int(v) for v in goal_cell)
        )

        mode = os.environ.get("PILOT_MODE", "auto").lower()  # Read route preference mode from env var
        self.cost_table = COSTS_SHORT if mode == "short" else COSTS_AUTO  # Pick cost table for mode
        self.fast_speed = float(os.environ.get("PILOT_SPEED", 0.85))  # Cruise speed on clear straight cells [m/s]
        self.safe_lane_offset = 0.48  # Lateral offset [m] placing robot in the middle of the ~0.98 m free lane

        # Instantiate the underlying MazeAutopilot for basic grid indexing and wall-scan helpers
        self.ap = MazeAutopilot(
            num_envs=num_envs,  # Number of parallel robots
            num_cells=num_cells,  # Grid size N
            cell_size=cell_size,  # Cell width [m]
            device=device,  # Torch device
            strategy="flood",  # Flood-fill strategy (overridden by weighted_flood_fill below)
            speed=self.fast_speed,  # Default cruising forward speed [m/s]
            turn_early=0.50,  # Default distance before cell center to start turning [m]
            min_speed=0.25,  # Minimum forward speed during sharp turns [m/s]
            max_turn=1.1,  # Maximum yaw rate command [rad/s]
            start=self.start_cell,  # Start cell tuple
            goal=self.goal_cell,  # Goal cell tuple
        )
        self.ap.max_acc = 1.2  # Linear velocity rate limit [m/s^2]
        self.ap.max_ang_acc = 2.8  # Angular velocity rate limit [rad/s^2]

        # Persistent hazard type map (N_env, N, N) -- survives across falls/resets
        self.cell_type = torch.zeros(
            (num_envs, num_cells, num_cells), dtype=torch.long, device=device
        )
        # Persistent traversal cost map (N_env, N, N) -- initialized to unexplored cost (1.05)
        self.cell_cost = torch.full(
            (num_envs, num_cells, num_cells),
            self.cost_table[CELL_UNKNOWN],
            dtype=torch.float32,
            device=device,
        )
        # Count of falls/resets attributed to each cell (N_env, N, N)
        self.cell_deaths = torch.zeros(
            (num_envs, num_cells, num_cells), dtype=torch.float32, device=device
        )
        # Corridor orientation axis for each cell: 0 = along X, 1 = along Y, -1 = unknown/corner
        self.cell_axis = torch.full(
            (num_envs, num_cells, num_cells), -1, dtype=torch.long, device=device
        )
        # Safe lateral offset v_safe [m] for each cell's [u < 0 half, u > 0 half]
        self.safe_offset = torch.zeros(
            (num_envs, num_cells, num_cells, 2), dtype=torch.float32, device=device
        )

        # Mark start and goal cells as known plain floor (never contain hazards)
        si, sj = self.start_cell  # Unpack start cell indices
        gi, gj = self.goal_cell  # Unpack goal cell indices
        self.cell_type[:, si, sj] = CELL_PLAIN  # Start cell is plain floor
        self.cell_type[:, gi, gj] = CELL_PLAIN  # Goal cell is plain floor
        self.cell_cost[:, si, sj] = 1.0  # Start cell has unit cost
        self.cell_cost[:, gi, gj] = 1.0  # Goal cell has unit cost

        # State buffers for death attribution and the 3.5-second anti-stuck reflex
        start_t = torch.tensor(self.start_cell, dtype=torch.long, device=device)  # Start cell tensor (2,)
        self.last_cell = start_t.repeat(num_envs, 1)  # Last cell occupied by each robot (N, 2)
        self.last_target = start_t.repeat(num_envs, 1)  # Last target cell of each robot (N, 2)
        self.pos_anchor = self.ap.centre_of(self.last_cell)  # Reference XY position for stuck detection (N, 2)
        self.anchor_time = torch.zeros(num_envs, dtype=torch.float32, device=device)  # Time when anchor was updated
        self.unstuck_until = torch.full((num_envs,), -1.0, dtype=torch.float32, device=device)  # End time of unstuck maneuver
        self.unstuck_dir = torch.ones(num_envs, dtype=torch.float32, device=device)  # Alternating +1/-1 turn direction
        self.last_wp = self.ap.centre_of(self.last_cell)  # Latest continuous sub-waypoint XY for visualization (N, 2)
        self.sim_time = 0.0  # Latest elapsed simulation time [s]

    # ------------------------------------------------------------------
    # 1. Perception: Walls + Multi-Cell Hazard Classification
    # ------------------------------------------------------------------
    def _corridor_axis(self, env_id: int, ci: int, cj: int, fallback_axis: int = 0) -> int:
        """Determine corridor axis (0 = along x, 1 = along y) from known walls or travel direction."""
        w = self.ap.walls[env_id, ci, cj]  # Wall state (4,) of cell (ci, cj): 0=E, 1=N, 2=W, 3=S
        # If North (1) or South (3) has a wall and East (0) & West (2) are open -> corridor runs along X (axis 0)
        if (w[1] == 1 or w[3] == 1) and w[0] != 1 and w[2] != 1:
            return 0  # Corridor axis is X
        # If East (0) or West (2) has a wall and North (1) & South (3) are open -> corridor runs along Y (axis 1)
        if (w[0] == 1 or w[2] == 1) and w[1] != 1 and w[3] != 1:
            return 1  # Corridor axis is Y
        known = int(self.cell_axis[env_id, ci, cj].item())  # Check previously stored axis for this cell
        return known if known >= 0 else fallback_axis  # Use stored axis if valid, otherwise travel axis

    def _classify_cell_from_scan(
        self,
        env_id: int,  # Index of the environment
        ci: int,  # Cell X index
        cj: int,  # Cell Y index
        pts_xy: torch.Tensor,  # Height-scanner ray hit XY positions in maze frame (R, 2)
        pts_z: torch.Tensor,  # Height-scanner ray hit Z heights relative to floor (R,)
        fallback_axis: int,  # Fallback corridor axis if walls of (ci, cj) are not fully known yet
    ):
        """Classify hazard type and safe lateral offset for cell (ci, cj) from 3D height-scan hits."""
        if (ci, cj) == self.start_cell or (ci, cj) == self.goal_cell:  # Start and goal never have hazards
            return  # Skip classification for start and goal cells
        cx = ci * self.cell_size  # World X coordinate of cell (ci, cj) center [m]
        cy = cj * self.cell_size  # World Y coordinate of cell (ci, cj) center [m]
        dx = pts_xy[:, 0] - cx  # X offset of each scan ray from cell center [m]
        dy = pts_xy[:, 1] - cy  # Y offset of each scan ray from cell center [m]

        # Select only rays landing inside the inner cell floor (excluding walls at +-1.0 m)
        in_cell = (torch.abs(dx) < 0.80) & (torch.abs(dy) < 0.80)
        n_total = int(in_cell.sum().item())  # Count how many rays landed inside this cell
        if n_total < 12:  # Require at least 12 rays in the cell before making a classification
            return  # Cell is too far outside the 3.2x3.2m scanner window

        axis = self._corridor_axis(env_id, ci, cj, fallback_axis)  # Get corridor axis (0=X, 1=Y)
        self.cell_axis[env_id, ci, cj] = axis  # Cache corridor axis for waypoint generation
        u = dx[in_cell] if axis == 0 else dy[in_cell]  # Longitudinal coordinate along corridor [-0.8, +0.8]
        v = dy[in_cell] if axis == 0 else dx[in_cell]  # Lateral coordinate across corridor [-0.8, +0.8]
        z = pts_z[in_cell]  # Height Z of rays inside this cell

        # 1. Trench / gap check: pit depth is -1.0 m, so >= 3 rays below -0.15 m indicates a trench
        if int((z < -0.15).sum().item()) >= 3:
            self._set_cell_hazard(env_id, ci, cj, CELL_TRENCH)  # Mark cell as TRENCH
            return  # Done classifying this cell

        # 2. Elevated obstacle check (0.11 m < z < 0.42 m): could be a Ramp, Spike Lane, or Slalom
        elev = (z > 0.11) & (z < 0.42)  # Mask of rays hitting elevated features below wall height (1.0 m)
        n_elev = int(elev.sum().item())  # Number of elevated ray hits
        if n_elev >= 3:  # At least 3 elevated hits -> analyze spatial distribution across (u, v)
            u_e = u[elev]  # Longitudinal coordinates of elevated hits
            v_e = v[elev]  # Lateral coordinates of elevated hits
            z_e = z[elev]  # Heights of elevated hits

            # Split cell into longitudinal halves (u_neg / u_pos) and lateral halves (v_neg / v_pos)
            u_neg = u_e < -0.12  # First longitudinal half (u in [-0.80, -0.12])
            u_pos = u_e > 0.12  # Second longitudinal half (u in [+0.12, +0.80])
            v_neg = v_e < -0.08  # Right/Negative lateral half (v in [-0.80, -0.08])
            v_pos = v_e > 0.08  # Left/Positive lateral half (v in [+0.08, +0.80])

            q_nn = int((u_neg & v_neg).sum().item())  # Elevated hits in (u < 0, v < 0) quadrant
            q_np = int((u_neg & v_pos).sum().item())  # Elevated hits in (u < 0, v > 0) quadrant
            q_pn = int((u_pos & v_neg).sum().item())  # Elevated hits in (u > 0, v < 0) quadrant
            q_pp = int((u_pos & v_pos).sum().item())  # Elevated hits in (u > 0, v > 0) quadrant
            tot_vn = int(v_neg.sum().item())  # Total elevated hits on negative lateral side
            tot_vp = int(v_pos.sum().item())  # Total elevated hits on positive lateral side

            # Check center longitudinal band (|u| < 0.22 m): ramps have flat top here, slaloms have a 0.6 m gap
            mid_band = torch.abs(u_e) < 0.22  # Rays near longitudinal center of cell
            mid_vn = int((mid_band & v_neg).sum().item())  # Center-band elevated hits on v < 0
            mid_vp = int((mid_band & v_pos).sum().item())  # Center-band elevated hits on v > 0

            # Slalom test: u < 0 has spikes on one side only, u > 0 has spikes on opposite side, center gap is clear
            neg_has_one_side = (q_nn >= 2 and q_np <= 1) or (q_np >= 2 and q_nn <= 1)  # Asymmetric in u < 0
            pos_has_one_side = (q_pn >= 2 and q_pp <= 1) or (q_pp >= 2 and q_pn <= 1)  # Asymmetric in u > 0
            opposite_sides = (q_nn > q_np) != (q_pn > q_pp)  # Active side flips between u < 0 and u > 0

            if neg_has_one_side and pos_has_one_side and opposite_sides and (mid_vn + mid_vp) <= 2:
                s_neg = -1.0 if q_nn > q_np else 1.0  # Spike side in u < 0 half (-1 or +1)
                s_pos = -1.0 if q_pn > q_pp else 1.0  # Spike side in u > 0 half (-1 or +1)
                self.safe_offset[env_id, ci, cj, 0] = -s_neg * self.safe_lane_offset  # Safe lane for u < 0
                self.safe_offset[env_id, ci, cj, 1] = -s_pos * self.safe_lane_offset  # Safe lane for u > 0
                self._set_cell_hazard(env_id, ci, cj, CELL_SPIKE_SLALOM)  # Mark cell as SLALOM
                return  # Done classifying this cell

            # Ramp test: both v < 0 and v > 0 are elevated simultaneously in the same longitudinal slice
            same_slice_sym = (q_nn >= 2 and q_np >= 2) or (q_pn >= 2 and q_pp >= 2) or (mid_vn >= 1 and mid_vp >= 1)
            if same_slice_sym and min(tot_vn, tot_vp) >= 2:
                # Do not overwrite an already confirmed slalom cell
                if int(self.cell_type[env_id, ci, cj].item()) != CELL_SPIKE_SLALOM:
                    # Use 90th percentile height to distinguish Steep Ramp (0.24-0.30 m) from Low Ramp (0.10-0.18 m)
                    z_max = float(torch.quantile(z_e, 0.90).item()) if len(z_e) >= 4 else float(z_e.max().item())
                    ramp_kind = CELL_STEEP_RAMP if z_max > 0.205 else CELL_LOW_RAMP  # Threshold at 0.205 m
                    self.safe_offset[env_id, ci, cj] = 0.0  # Ramps span full width -> walk down centerline (0.0)
                    self._set_cell_hazard(env_id, ci, cj, ramp_kind)  # Record STEEP_RAMP or LOW_RAMP
                return  # Done classifying this cell

            # Single-Lane Spike test: elevated hits sit on only one lateral side of the corridor
            if tot_vn >= 2 or tot_vp >= 2:
                prev_type = int(self.cell_type[env_id, ci, cj].item())  # Check existing classification
                if prev_type == CELL_SPIKE_SLALOM:  # Keep slalom if already upgraded
                    return
                spike_side = -1.0 if tot_vn > tot_vp else 1.0  # Side occupied by the spike cones (-1 or +1)
                safe_v = -spike_side * self.safe_lane_offset  # Free lane is on the opposite side (+-0.48 m)
                # Check if we previously saw spikes on the opposite side in the other longitudinal half (slalom!)
                old_u0 = float(self.safe_offset[env_id, ci, cj, 0].item())  # Previous u < 0 safe offset
                old_u1 = float(self.safe_offset[env_id, ci, cj, 1].item())  # Previous u > 0 safe offset
                if (q_nn + q_np) >= 2 and (q_pn + q_pp) == 0 and old_u1 != 0.0 and old_u1 * safe_v < 0:
                    self.safe_offset[env_id, ci, cj, 0] = safe_v  # Record u < 0 offset
                    self._set_cell_hazard(env_id, ci, cj, CELL_SPIKE_SLALOM)  # Upgrade to SLALOM
                    return
                if (q_pn + q_pp) >= 2 and (q_nn + q_np) == 0 and old_u0 != 0.0 and old_u0 * safe_v < 0:
                    self.safe_offset[env_id, ci, cj, 1] = safe_v  # Record u > 0 offset
                    self._set_cell_hazard(env_id, ci, cj, CELL_SPIKE_SLALOM)  # Upgrade to SLALOM
                    return

                self.safe_offset[env_id, ci, cj, 0] = safe_v  # Same safe lane offset for u < 0
                self.safe_offset[env_id, ci, cj, 1] = safe_v  # Same safe lane offset for u > 0
                self._set_cell_hazard(env_id, ci, cj, CELL_SPIKE_LANE)  # Mark as single-lane spike cell
                return  # Done classifying this cell

        # 3. Rough floor (2-8 cm loose boxes) vs Ice (flat 1 cm plate) vs Plain floor
        if int(self.cell_type[env_id, ci, cj].item()) in (
            CELL_TRENCH,
            CELL_STEEP_RAMP,
            CELL_LOW_RAMP,
            CELL_SPIKE_LANE,
            CELL_SPIKE_SLALOM,
        ):
            return  # Do not downgrade a major hazard if a partial scan only sees flat floor

        rough_hits = int(((z > 0.025) & (z < 0.095)).sum().item())  # Count hits in the 2.5-9.5 cm box height range
        if rough_hits >= 4:  # Multiple low bumps -> loose boxes
            self._set_cell_hazard(env_id, ci, cj, CELL_ROUGH)  # Mark cell as ROUGH
        elif n_total >= 25:  # Enough rays to judge flat floor height accurately
            z_med = float(z.median().item())  # Median floor height in this cell
            z_std = float(z.std().item())  # Standard deviation of floor height
            if 0.006 <= z_med <= 0.016 and z_std < 0.008:  # Flat surface ~1 cm above z = 0 -> ice tile
                self._set_cell_hazard(env_id, ci, cj, CELL_ICE)  # Mark cell as ICE
            elif int(self.cell_type[env_id, ci, cj].item()) == CELL_UNKNOWN:  # Flat at z = 0 -> plain floor
                self._set_cell_hazard(env_id, ci, cj, CELL_PLAIN)  # Mark cell as PLAIN

    def _set_cell_hazard(self, env_id: int, ci: int, cj: int, hazard_type: int):
        """Update cell classification and recompute its traversal cost (base cost + death penalty)."""
        self.cell_type[env_id, ci, cj] = hazard_type  # Store integer hazard category
        base_cost = self.cost_table[hazard_type]  # Look up base traversal cost for this hazard type
        self.cell_cost[env_id, ci, cj] = base_cost + 50.0 * self.cell_deaths[env_id, ci, cj]  # Add reset penalty

    def _scan_walls_with_coverage(
        self,
        scan_xy: torch.Tensor,
        scan_z: torch.Tensor,
        cell: torch.Tensor,
        lidar_xy: torch.Tensor | None = None,
        lidar_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect walls and sensor coverage for each of the 4 sides of ``cell``.

        Returns:
            walls_hit: (N, 4) bool -- True when >= 3 tall height-scan points (or >= 2 2D LiDAR hits)
                       land inside side d's wall bounding box.
            covered:   (N, 4) bool -- True when >= 8 height-scan rays landed inside side d's bounding
                       box, proving the wall is inside the 3.2x3.2 m scanner window (not behind out of range).
        """
        centre = self.ap.centre_of(cell).unsqueeze(1)  # (N, 1, 2) world XY center of cell
        rel = scan_xy - centre  # (N, R, 2) ray offsets from cell center
        tall = scan_z > 0.50  # (N, R) points above 0.50 m (maze walls are 1.0 m tall; all floor hazards <= 0.35 m)
        n_envs = scan_xy.shape[0]
        walls_hit = torch.zeros((n_envs, 4), dtype=torch.bool, device=self.device)
        covered = torch.zeros((n_envs, 4), dtype=torch.bool, device=self.device)
        half = 0.5 * self.cell_size  # 1.0 m distance from cell center to each wall

        for d, (dx, dy) in enumerate(DIRS):  # 0=East, 1=North, 2=West, 3=South
            along = rel[..., 0] * dx + rel[..., 1] * dy  # Distance along normal toward side d [m]
            across = rel[..., 0] * dy - rel[..., 1] * dx  # Distance along wall tangent [m]
            in_box = (torch.abs(along - half) < 0.15) & (torch.abs(across) < 0.60)  # 0.30 m x 1.20 m wall box
            covered[:, d] = in_box.sum(dim=1) >= 8  # At least 8 rays must land in the box to confirm visibility
            walls_hit[:, d] = (in_box & tall).sum(dim=1) >= 3  # >= 3 tall hits confirms a physical wall

            # Fuse 2D horizontal chest LiDAR hit points (which fly at z ~ 0.75 m over all hazards and only hit walls)
            if lidar_xy is not None and lidar_valid is not None:
                l_rel = lidar_xy - centre  # (N, 72, 2)
                l_along = l_rel[..., 0] * dx + l_rel[..., 1] * dy
                l_across = l_rel[..., 0] * dy - l_rel[..., 1] * dx
                l_in_box = lidar_valid & (torch.abs(l_along - half) < 0.18) & (torch.abs(l_across) < 0.60)
                walls_hit[:, d] |= l_in_box.sum(dim=1) >= 2  # >= 2 LiDAR hits on the boundary confirms a wall

        return walls_hit, covered

    def _update_cell_walls(
        self, env_id: int, ci: int, cj: int, walls_hit: torch.Tensor, covered: torch.Tensor
    ):
        """Latch confirmed walls (1) permanently and only mark open (0) when inside sensor coverage."""
        for d, (di, dj) in enumerate(DIRS):  # 0=East, 1=North, 2=West, 3=South
            ni, nj = ci + di, cj + dj  # Neighbor cell across side d
            in_grid = 0 <= ni < self.num_cells and 0 <= nj < self.num_cells
            opp_d = (d + 2) % 4  # Opposite wall index in neighbor cell

            if bool(walls_hit[d].item()):
                # Confirmed wall: latch permanently as 1 (never overwritten when robot walks out of range)
                self.ap.walls[env_id, ci, cj, d] = 1
                if in_grid:
                    self.ap.walls[env_id, ni, nj, opp_d] = 1
            elif bool(covered[d].item()) and int(self.ap.walls[env_id, ci, cj, d].item()) != 1:
                # Confirmed open with full sensor coverage (and never previously latched as a wall)
                self.ap.walls[env_id, ci, cj, d] = 0
                if in_grid and int(self.ap.walls[env_id, ni, nj, opp_d].item()) != 1:
                    self.ap.walls[env_id, ni, nj, opp_d] = 0

    def sense_environment(self, obs: dict, cell: torch.Tensor) -> torch.Tensor:
        """Sense walls of current + adjacent cells and classify floor hazards within the 3.2x3.2 m scan."""
        scan_xy = obs["scan_xy"]  # Ray hit XY coordinates in maze frame (N, 1089, 2)
        scan_z = obs["scan_z"]  # Ray hit Z coordinates relative to floor (N, 1089)

        # Reconstruct 2D chest LiDAR hit XY coordinates in maze frame (up to 5.5 m range)
        lidar_xy = None
        lidar_valid = None
        if "lidar_dist" in obs and "lidar_yaw" in obs:
            l_dist = obs["lidar_dist"]  # (N, 72)
            l_yaw = obs["lidar_yaw"]  # (N, 72)
            lidar_valid = (l_dist > 0.15) & (l_dist < 5.5)  # Valid wall hits within 5.5 m
            lidar_xy = obs["pos_xy"].unsqueeze(1) + l_dist.unsqueeze(-1) * torch.stack(
                [torch.cos(l_yaw), torch.sin(l_yaw)], dim=-1
            )  # (N, 72, 2)

        # Detect walls and sensor coverage for the current cell
        walls_hit, covered = self._scan_walls_with_coverage(scan_xy, scan_z, cell, lidar_xy, lidar_valid)

        # Also fuse angular 2D LiDAR cone check when reasonably near the cell center (< 0.60 m, includes safe lanes)
        if "lidar_dist" in obs and "lidar_yaw" in obs:
            offset_xy = obs["pos_xy"] - self.ap.centre_of(cell)  # Robot offset from current cell center (N, 2)
            near = torch.linalg.norm(offset_xy, dim=1) < 0.60  # Includes +-0.48 m safe lanes
            if near.any():
                lidar_walls = self.ap.walls_from_lidar(obs["lidar_dist"], obs["lidar_yaw"], offset_xy)
                walls_hit = torch.where(near.unsqueeze(1), walls_hit | lidar_walls, walls_hit)

        # Update persistent wall map for current cell using coverage-gated latching (never forgets passed walls!)
        for b in range(self.num_envs):
            ci, cj = int(cell[b, 0].item()), int(cell[b, 1].item())
            self._update_cell_walls(b, ci, cj, walls_hit[b], covered[b])

        # Classify floor hazards and sense walls in all neighbor cells within scanner/LiDAR range
        for b in range(self.num_envs):  # Loop over each environment
            ci, cj = int(cell[b, 0].item()), int(cell[b, 1].item())  # Current cell (ci, cj)
            h_dir = int(self.ap.heading[b].item())  # Current heading direction index (0..3)
            fallback_axis = 0 if DIRS[h_dir][0] != 0 else 1  # Travel axis (0=X, 1=Y)
            self._classify_cell_from_scan(b, ci, cj, scan_xy[b], scan_z[b], fallback_axis)  # Classify current cell

            for d, (di, dj) in enumerate(DIRS):  # Check all 4 neighbors around (ci, cj)
                ni, nj = ci + di, cj + dj  # Neighbor cell indices
                if not (0 <= ni < self.num_cells and 0 <= nj < self.num_cells):  # Must lie inside the maze grid
                    continue
                nb_cell = torch.tensor([[ni, nj]], dtype=torch.long, device=self.device)  # Neighbor tensor (1, 2)
                nb_centre = self.ap.centre_of(nb_cell)  # World XY center of neighbor cell
                dist_to_nb = float(torch.linalg.norm(obs["pos_xy"][b : b + 1] - nb_centre).item())

                # Sense neighbor walls via top-down height scan + 2D LiDAR hits (up to 2.6 m away)
                if dist_to_nb < 2.60:
                    l_xy_b = lidar_xy[b : b + 1] if lidar_xy is not None else None
                    l_val_b = lidar_valid[b : b + 1] if lidar_valid is not None else None
                    nb_hit, nb_cov = self._scan_walls_with_coverage(
                        scan_xy[b : b + 1], scan_z[b : b + 1], nb_cell, l_xy_b, l_val_b
                    )
                    self._update_cell_walls(b, ni, nj, nb_hit[0], nb_cov[0])

                # Only classify floor hazards inside open neighbors (do not classify across a blocked wall)
                if int(self.ap.walls[b, ci, cj, d].item()) == 1:
                    continue
                nb_axis = 0 if di != 0 else 1  # Entering neighbor along X (0) or Y (1)
                self._classify_cell_from_scan(b, ni, nj, scan_xy[b], scan_z[b], nb_axis)  # Classify neighbor

        # Return the persistent remembered walls (N, 4) of the current cell (so passed walls stay 1!)
        env_arange = torch.arange(self.num_envs, device=self.device)
        return self.ap.walls[env_arange, cell[:, 0], cell[:, 1]] == 1

    # ------------------------------------------------------------------
    # 2. Global Route Planning: Hazard-Weighted Dijkstra
    # ------------------------------------------------------------------
    def weighted_flood_fill(self, walls: torch.Tensor) -> torch.Tensor:
        """Compute minimum cumulative traversal cost from every cell to goal_cell via Bellman-Ford."""
        n = self.num_cells  # Grid dimension N
        big = 1e6  # Infinity cost for unreachable/blocked transitions
        dist = torch.full((walls.shape[0], n, n), big, dtype=torch.float32, device=self.device)  # Init cost-to-goal
        dist[:, self.goal_cell[0], self.goal_cell[1]] = 0.0  # Goal cell has 0 remaining cost
        blocked = walls == 1  # Only confirmed walls (1) block edges; unknown (-1) and open (0) are traversable
        costs = self.cell_cost  # Per-cell traversal cost tensor (N_env, n, n)

        for _ in range(n * n):  # Relax edges up to N^2 times until convergence
            new = dist.clone()  # Copy current cost-to-goal estimates
            # Transition East (+x, d=0): from (i, j) into (i+1, j) pays cost of (i+1, j)
            new[:, :-1, :] = torch.minimum(
                new[:, :-1, :],
                torch.where(blocked[:, :-1, :, 0], big, dist[:, 1:, :] + costs[:, 1:, :]),
            )
            # Transition West (-x, d=2): from (i, j) into (i-1, j) pays cost of (i-1, j)
            new[:, 1:, :] = torch.minimum(
                new[:, 1:, :],
                torch.where(blocked[:, 1:, :, 2], big, dist[:, :-1, :] + costs[:, :-1, :]),
            )
            # Transition North (+y, d=1): from (i, j) into (i, j+1) pays cost of (i, j+1)
            new[:, :, :-1] = torch.minimum(
                new[:, :, :-1],
                torch.where(blocked[:, :, :-1, 1], big, dist[:, :, 1:] + costs[:, :, 1:]),
            )
            # Transition South (-y, d=3): from (i, j) into (i, j-1) pays cost of (i, j-1)
            new[:, :, 1:] = torch.minimum(
                new[:, :, 1:],
                torch.where(blocked[:, :, 1:, 3], big, dist[:, :, :-1] + costs[:, :, :-1]),
            )
            if torch.allclose(new, dist):  # Stop early once fixed point is reached
                break
            dist = new  # Update cost-to-goal tensor
        return dist  # Return (N_env, n, n) cumulative cost-to-goal

    def choose_direction(
        self, cell: torch.Tensor, open_sides: torch.Tensor, heading: torch.Tensor
    ) -> torch.Tensor:
        """Pick the open neighbor direction with minimum weighted cost-to-goal (preferring straight on ties)."""
        dist = self.weighted_flood_fill(self.ap.walls)  # Compute cost-to-goal map for all environments
        idx = torch.arange(len(heading), device=self.device)  # Batch indices for subset of environments deciding
        best = torch.full_like(heading, -1)  # Best direction found so far (-1 = none)
        best_cost = torch.full(heading.shape, 1e9, dtype=torch.float32, device=self.device)  # Best total cost

        for t in (0, 1, 3, 2):  # Evaluate straight (0), left (1), right (3), back (2) in tie-break order
            d = (heading + t) % 4  # World cardinal direction (0..3)
            nb = (cell + self.ap._dirs[d]).clamp(0, self.num_cells - 1)  # Neighbor cell coordinates (B, 2)
            turn_pen = 0.08 if t in (1, 3) else (0.25 if t == 2 else 0.0)  # Small penalty for turning/reversing
            step_cost = self.cell_cost[idx, nb[:, 0], nb[:, 1]]  # Cost to step into neighbor cell
            cost = dist[idx, nb[:, 0], nb[:, 1]] + step_cost + turn_pen  # Total cost through this neighbor
            ok = open_sides[idx, d] & (cost < best_cost - 1e-4)  # Must be open and strictly cheaper
            best = torch.where(ok, d, best)  # Update best direction
            best_cost = torch.where(ok, cost, best_cost)  # Update lowest cost

        return torch.where(best >= 0, best, (heading + 2) % 4)  # Turn back if all sides appear blocked

    # ------------------------------------------------------------------
    # 3. Local Trajectory & Waypoint Control (Safe Lane & Slalom)
    # ------------------------------------------------------------------
    def _compute_waypoint_and_speed(
        self, env_id: int, pos: torch.Tensor, yaw: float, cell: torch.Tensor, scan_xy: torch.Tensor, scan_z: torch.Tensor
    ) -> tuple[torch.Tensor, float, float]:
        """Return (waypoint_xy, desired_max_speed, lateral_vy_bias) for robot ``env_id``."""
        ci, cj = int(cell[0].item()), int(cell[1].item())  # Current cell coordinates
        ti, tj = int(self.ap.target[env_id, 0].item()), int(self.ap.target[env_id, 1].item())  # Target cell coords
        ctype = int(self.cell_type[env_id, ci, cj].item())  # Hazard type of current cell
        ttype = int(self.cell_type[env_id, ti, tj].item())  # Hazard type of target cell

        wp = self.ap.centre_of(self.ap.target[env_id]).clone()  # Default waypoint is center of target cell
        speed = self.fast_speed  # Default max speed is fast cruise speed (0.85 m/s)
        vy_bias = 0.0  # Default lateral velocity bias is 0.0 m/s

        # Determine travel direction vector and sign (+1 or -1) along the corridor
        h_dir = int(self.ap.heading[env_id].item())  # Current travel direction (0..3)
        di, dj = DIRS[h_dir]  # Step delta (di, dj)
        travel_axis = 0 if di != 0 else 1  # 0 = moving along X, 1 = moving along Y
        travel_sign = float(di if travel_axis == 0 else dj)  # +1.0 for +x/+y, -1.0 for -x/-y

        # Case 1: Robot is currently inside a Single-Lane Spike or Slalom Spike cell
        if ctype in (CELL_SPIKE_LANE, CELL_SPIKE_SLALOM):
            axis = int(self.cell_axis[env_id, ci, cj].item())  # Corridor axis of current spike cell
            if axis < 0:  # Fallback to travel axis if corridor walls weren't fully seen
                axis = travel_axis
            cx, cy = ci * self.cell_size, cj * self.cell_size  # World center of current spike cell
            u_world = float(pos[0].item() - cx) if axis == 0 else float(pos[1].item() - cy)  # Longitudinal pos
            v_world = float(pos[1].item() - cy) if axis == 0 else float(pos[0].item() - cx)  # Lateral pos
            u_rel = u_world * travel_sign  # Signed progress along travel direction (-1.0 entry -> +1.0 exit)

            if ctype == CELL_SPIKE_LANE:
                v_safe = float(self.safe_offset[env_id, ci, cj, 0].item())  # Safe lateral lane offset (+-0.48 m)
                # Keep tracking the safe lane until u_rel > 0.78 m (spike bed ends at u_rel = +0.70 m)
                if u_rel < 0.78:
                    u_aim = max(u_world + travel_sign * 0.65, -0.85) if travel_sign < 0 else min(u_world + 0.65, 0.85)
                    if (ci, cj) != (ti, tj) and axis == travel_axis:  # Already targeting next cell ahead
                        u_aim = travel_sign * 0.90  # Aim at exit border while holding v_safe
                    wp[0] = (cx + u_aim) if axis == 0 else (cx + v_safe)  # Set waypoint X
                    wp[1] = (cy + v_safe) if axis == 0 else (cy + u_aim)  # Set waypoint Y
                    speed = 0.48  # Moderate speed for precise safe-lane tracking
            else:
                # Slalom cell: two opposite-side beds separated by a 0.60 m gap at u_world in [-0.30, +0.30]
                v_neg = float(self.safe_offset[env_id, ci, cj, 0].item())  # Safe lane in u_world < 0 half
                v_pos = float(self.safe_offset[env_id, ci, cj, 1].item())  # Safe lane in u_world > 0 half
                v_entry = v_neg if travel_sign > 0 else v_pos  # Safe lane for entry half
                v_exit = v_pos if travel_sign > 0 else v_neg  # Safe lane for exit half

                if u_rel < -0.22:  # Stage 1: before the central gap -> stay in entry safe lane
                    u_aim = -0.10 * travel_sign  # Aim just inside the central gap
                    v_aim = v_entry  # Hold entry safe lane offset
                    speed = 0.42  # Cautious speed approaching crossover gap
                elif abs(v_world - v_exit) > 0.18 and u_rel < 0.25:  # Stage 2: inside the 0.6 m crossover gap
                    u_aim = 0.05 * travel_sign  # Stay centered in the gap longitudinally
                    v_aim = v_exit  # Shift laterally across to the exit safe lane
                    speed = 0.35  # Slow down during lateral lane change
                else:  # Stage 3: aligned with exit safe lane -> walk straight out past second bed (+0.90 m)
                    u_aim = 0.95 * travel_sign  # Aim past exit border of slalom cell
                    v_aim = v_exit  # Hold exit safe lane offset
                    speed = 0.45  # Steady exit speed

                wp[0] = (cx + u_aim) if axis == 0 else (cx + v_aim)  # Set slalom waypoint X
                wp[1] = (cy + v_aim) if axis == 0 else (cy + u_aim)  # Set slalom waypoint Y

        # Case 2: Target cell ahead is a Spike Lane or Slalom -> pre-align into its entry safe lane early!
        if ttype in (CELL_SPIKE_LANE, CELL_SPIKE_SLALOM) and (ci, cj) != (ti, tj):
            t_axis = int(self.cell_axis[env_id, ti, tj].item())  # Corridor axis of upcoming spike cell
            if t_axis < 0:  # Fallback to travel axis
                t_axis = travel_axis
            if t_axis == travel_axis:  # Moving straight into the spike cell
                v_entry = (  # Pick the safe lane offset of the entry half of the upcoming cell
                    float(self.safe_offset[env_id, ti, tj, 0].item())
                    if travel_sign > 0
                    else float(self.safe_offset[env_id, ti, tj, 1].item())
                )
                tcx, tcy = ti * self.cell_size, tj * self.cell_size  # Center of target spike cell
                u_entry = -0.75 * travel_sign  # Entry edge of the spike bed
                wp[0] = (tcx + u_entry) if t_axis == 0 else (tcx + v_entry)  # Pre-align waypoint X
                wp[1] = (tcy + v_entry) if t_axis == 0 else (tcy + u_entry)  # Pre-align waypoint Y
                speed = min(speed, 0.50)  # Moderate approach speed for accurate pre-alignment

        # Case 3: Speed adaptation for Trench, Ramp, Ice, and Rough cells
        if ctype in (CELL_TRENCH, CELL_STEEP_RAMP) or ttype in (CELL_TRENCH, CELL_STEEP_RAMP):
            speed = min(speed, 0.70)  # Steady momentum to step across trenches and climb steep ramps
        elif ctype == CELL_ICE or ttype == CELL_ICE:
            speed = min(speed, 0.55)  # Reduced speed on low-friction ice tiles
        elif ctype in (CELL_ROUGH, CELL_LOW_RAMP) or ttype in (CELL_ROUGH, CELL_LOW_RAMP):
            speed = min(speed, 0.68)  # Balanced speed over loose boxes and low ramps

        # Local point-cloud repulsion shield: push robot away from any spike cones close to the body
        if ctype in (CELL_SPIKE_LANE, CELL_SPIKE_SLALOM) or ttype in (CELL_SPIKE_LANE, CELL_SPIKE_SLALOM):
            rel_xy = scan_xy - pos.unsqueeze(0)  # Scan points relative to robot base (R, 2)
            c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)  # Cosine and sine of robot yaw
            rx = rel_xy[:, 0] * c_yaw + rel_xy[:, 1] * s_yaw  # Forward distance in robot frame [m]
            ry = -rel_xy[:, 0] * s_yaw + rel_xy[:, 1] * c_yaw  # Lateral distance in robot frame (+left, -right) [m]
            # Check for cone-height points (0.14-0.38 m) in the immediate front corridor of the robot
            cone_mask = (scan_z > 0.14) & (scan_z < 0.38) & (rx > -0.15) & (rx < 0.65) & (torch.abs(ry) < 0.38)
            if int(cone_mask.sum().item()) >= 2:  # Cones detected dangerously close
                mean_ry = float(ry[cone_mask].mean().item())  # Average lateral position of nearby cones
                vy_bias = -0.22 if mean_ry > 0.0 else 0.22  # Sidestep away from the cones
                speed = min(speed, 0.38)  # Slow forward speed while sidestepping away

        return wp, speed, vy_bias  # Return target waypoint (2,), speed limit, and lateral velocity bias

    # ------------------------------------------------------------------
    # 4. Main Step & Anti-Stuck Reflex
    # ------------------------------------------------------------------
    def act(self, obs: dict) -> torch.Tensor:
        """Compute velocity command (num_envs, 3) -> [v_x, v_y, w_z] in robot frame."""
        pos_xy = obs["pos_xy"]  # Current robot XY positions in maze frame (N, 2)
        yaw = obs["yaw"]  # Current robot yaw headings in maze frame (N,)
        dt = float(obs["dt"])  # Policy step timestep (0.02 s)
        self.sim_time = float(obs.get("t", self.sim_time + dt))  # Current elapsed simulation time [s]

        cell = self.ap.cell_of(pos_xy)  # Current grid cell (i, j) of each robot (N, 2)
        walls_now = self.sense_environment(obs, cell)  # Update wall map and classify hazards in view
        self.last_cell = cell.clone()  # Record current cell for failure attribution if a reset occurs

        # Use a tight turn radius (0.22 m) near spikes/trenches so the robot never cuts corners into hazards
        cur_type = self.cell_type[torch.arange(self.num_envs, device=self.device), cell[:, 0], cell[:, 1]]
        tgt_type = self.cell_type[
            torch.arange(self.num_envs, device=self.device), self.ap.target[:, 0], self.ap.target[:, 1]
        ]
        cautious_cell = (cur_type >= CELL_SPIKE_LANE) | (tgt_type >= CELL_SPIKE_LANE)  # True near major hazards
        turn_radius = torch.where(  # 0.22 m in hazard cells, 0.52 m in plain/easy cells for smooth arc turns
            cautious_cell,
            torch.full((self.num_envs,), 0.22, device=self.device),
            torch.full((self.num_envs,), 0.52, device=self.device),
        )

        # Check if robot has arrived at its current target cell (accounting for +-0.48 m safe-lane offset in spike cells)
        rel_centre = pos_xy - self.ap.centre_of(cell)  # (N, 2) offset from cell center
        dir_vec = self.ap._dirs[self.ap.heading].to(pos_xy.dtype)  # (N, 2) unit vector along travel heading
        u_along = (rel_centre * dir_vec).sum(dim=1)  # Signed longitudinal position in cell [-1.0, +1.0]
        in_spike_cell = (cur_type == CELL_SPIKE_LANE) | (cur_type == CELL_SPIKE_SLALOM)
        near_spike_mid = in_spike_cell & (torch.abs(u_along) < 0.30)  # Passing longitudinal center in safe lane
        near_centre = (torch.linalg.norm(rel_centre, dim=1) < turn_radius) | near_spike_mid
        arrived = near_centre & (cell == self.ap.target).all(dim=1)  # Arrived at target cell decision point

        # Re-evaluate direction if a newly sensed wall blocks our heading OR a hard hazard is spotted ahead
        still_there = (cell == self.ap.origin).all(dim=1) & ~self.ap.need_decision
        surprise = still_there & walls_now.gather(1, self.ap.heading.unsqueeze(1)).squeeze(1)
        hazard_surprise = still_there & (tgt_type >= CELL_SPIKE_SLALOM)  # Spotted slalom/trench/steep ramp ahead

        decide = self.ap.need_decision | arrived | surprise | hazard_surprise  # Envs needing a planner update
        if decide.any():
            idx = torch.nonzero(decide).squeeze(1)  # Indices of deciding environments
            open_sides = self.ap.walls[idx, cell[idx, 0], cell[idx, 1]] == 0  # Open walls of current cell
            d = self.choose_direction(cell[idx], open_sides, self.ap.heading[idx])  # Run Weighted Dijkstra
            moved = ~self.ap.need_decision[idx] & ~surprise[idx] & ~hazard_surprise[idx]  # True if advanced a cell
            self.ap.cells_walked[idx] += moved.long()  # Increment cells walked counter
            self.ap.heading[idx] = d  # Update chosen travel direction
            self.ap.target[idx] = (cell[idx] + self.ap._dirs[d]).clamp(0, self.num_cells - 1)  # Next target cell
            self.ap.origin[idx] = cell[idx]  # Record decision origin cell
            self.ap.need_decision[idx] = False  # Clear decision flag
            self.last_target[idx] = self.ap.target[idx]  # Save target cell for reset attribution

        # Anti-stuck reflex: check if robot has progressed at least 0.28 m within the last 3.5 seconds
        moved_dist = torch.linalg.norm(pos_xy - self.pos_anchor, dim=1)  # Distance from anchor position
        progressed = moved_dist > 0.28  # Robot made normal forward progress
        if progressed.any():
            self.pos_anchor[progressed] = pos_xy[progressed]  # Reset anchor position to current XY
            self.anchor_time[progressed] = self.sim_time  # Reset anchor timestamp

        # Trigger a 1.2-second unstuck maneuver if stuck for > 3.5 s (well before the 10.0 s challenge penalty)
        stuck_now = ((self.sim_time - self.anchor_time) > 3.5) & (self.sim_time > self.unstuck_until)
        if stuck_now.any():
            for b in torch.nonzero(stuck_now).squeeze(1).tolist():  # Handle each stuck robot
                self.unstuck_until[b] = self.sim_time + 1.2  # Back up and turn for 1.2 seconds
                self.unstuck_dir[b] = -self.unstuck_dir[b]  # Flip turn/strafe direction from last attempt
                self.pos_anchor[b] = pos_xy[b]  # Reset anchor position
                self.anchor_time[b] = self.sim_time  # Reset anchor time
                ti, tj = int(self.ap.target[b, 0].item()), int(self.ap.target[b, 1].item())  # Blocking target cell
                if (ti, tj) != self.start_cell and (ti, tj) != self.goal_cell:
                    self.cell_cost[b, ti, tj] += 15.0  # Penalize blocking cell so planner considers alternatives
                self.ap.need_decision[b] = True  # Replan immediately after unstuck maneuver

        # Compute body-frame velocity commands [v_x, v_y, w_z] for each robot
        out_cmd = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        for b in range(self.num_envs):
            if self.sim_time < float(self.unstuck_until[b].item()):  # Currently executing unstuck reflex
                s = float(self.unstuck_dir[b].item())  # Turn/strafe sign (+1 or -1)
                out_cmd[b, 0] = -0.25  # Walk backward at -0.25 m/s to free wedged feet
                out_cmd[b, 1] = 0.20 * s  # Strafe sideways at 0.20 m/s
                out_cmd[b, 2] = 0.75 * s  # Yaw at 0.75 rad/s
                self.ap.cmd[b] = out_cmd[b]  # Sync internal rate-limiter state
                continue

            # Compute desired waypoint, hazard-adjusted speed, and cone-repulsion lateral bias
            wp, max_spd, vy_bias = self._compute_waypoint_and_speed(
                b, pos_xy[b], float(yaw[b].item()), cell[b], obs["scan_xy"][b], obs["scan_z"][b]
            )
            self.last_wp[b] = wp  # Store active local waypoint XY for viewer HUD & 3D markers
            to_wp = wp - pos_xy[b]  # Vector from robot to waypoint in maze frame (2,)
            yaw_err = math.atan2(float(to_wp[1].item()), float(to_wp[0].item())) - float(yaw[b].item())
            yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))  # Wrap heading error to [-pi, pi]

            # Project waypoint offset into robot lateral axis for smooth holonomic lane tracking
            c_yaw, s_yaw = math.cos(float(yaw[b].item())), math.sin(float(yaw[b].item()))
            lat_err_b = -float(to_wp[0].item()) * s_yaw + float(to_wp[1].item()) * c_yaw

            cos_err = max(math.cos(yaw_err), 0.0)  # Scale forward speed by cos^2(yaw_err)
            v_des = self.ap.min_speed + (max_spd - self.ap.min_speed) * (cos_err**2)
            if abs(yaw_err) > 0.7:  # Slow to min_speed when a large heading change is required
                v_des = self.ap.min_speed

            # Limit yaw rate on slippery ice cells to prevent spinning out
            max_w = 0.65 if int(self.cell_type[b, cell[b, 0], cell[b, 1]].item()) == CELL_ICE else self.ap.max_turn
            w_des = max(min(1.6 * yaw_err, max_w), -max_w)  # Proportional yaw rate controller

            # Apply gentle lateral strafing (v_y) when roughly aligned with the corridor
            vy_des = vy_bias  # Start with spike-repulsion bias
            if abs(yaw_err) < 0.45:  # Add proportional lateral lane-tracking velocity
                vy_des += max(min(0.45 * lat_err_b, 0.22), -0.22)
            vy_des = max(min(vy_des, 0.28), -0.28)  # Clamp total lateral velocity command

            # Apply acceleration rate limits for smooth locomotion commands
            acc = 0.6 if int(self.cell_type[b, cell[b, 0], cell[b, 1]].item()) == CELL_ICE else self.ap.max_acc
            ang_acc = self.ap.max_ang_acc  # Angular acceleration limit
            self.ap.cmd[b, 0] += max(min(v_des - float(self.ap.cmd[b, 0].item()), acc * dt), -acc * dt)
            self.ap.cmd[b, 1] += max(min(vy_des - float(self.ap.cmd[b, 1].item()), acc * dt), -acc * dt)
            self.ap.cmd[b, 2] += max(min(w_des - float(self.ap.cmd[b, 2].item()), ang_acc * dt), -ang_acc * dt)
            out_cmd[b] = self.ap.cmd[b]  # Copy smoothed command to output tensor

        return out_cmd  # Return (N, 3) velocity commands [v_x, v_y, w_z]

    def reset(self, env_ids: torch.Tensor):
        """Called when the robot falls, hits a spike bed, or gets stuck and resets to start.

        Preserves the discovered wall and hazard maps and penalizes the cell where the failure occurred
        so the planner automatically switches routes on the next attempt.
        """
        for b in env_ids.tolist():  # Attribute failure to the cell where the robot just reset
            ci, cj = int(self.last_cell[b, 0].item()), int(self.last_cell[b, 1].item())  # Last occupied cell
            ti, tj = int(self.last_target[b, 0].item()), int(self.last_target[b, 1].item())  # Last target cell
            if (ci, cj) != self.start_cell and (ci, cj) != self.goal_cell:  # Penalize last occupied cell
                self.cell_deaths[b, ci, cj] += 1.0  # Increment death counter
                self.cell_cost[b, ci, cj] += 50.0  # Add +50 traversal cost penalty
            elif (ti, tj) != self.start_cell and (ti, tj) != self.goal_cell:  # Or penalize target cell
                self.cell_deaths[b, ti, tj] += 1.0  # Increment death counter
                self.cell_cost[b, ti, tj] += 50.0  # Add +50 traversal cost penalty

        # Reset robot navigation state to start_cell WITHOUT erasing self.ap.walls or self.cell_cost
        start = torch.tensor(self.start_cell, dtype=torch.long, device=self.device)  # Start cell tensor (2,)
        self.ap.target[env_ids] = start  # Reset current target to start cell
        self.ap.origin[env_ids] = start  # Reset decision origin to start cell
        self.ap.heading[env_ids] = 0  # Reset initial heading to +X (East)
        self.ap.need_decision[env_ids] = True  # Force direction decision on first step after respawn
        self.ap.cells_walked[env_ids] = 0  # Reset cells walked counter for this attempt
        self.ap.cmd[env_ids] = 0.0  # Zero velocity commands
        self.last_cell[env_ids] = start  # Reset last cell tracker
        self.last_target[env_ids] = start  # Reset last target tracker
        self.pos_anchor[env_ids] = self.ap.centre_of(start)  # Reset anti-stuck position anchor
        self.last_wp[env_ids] = self.ap.centre_of(start)  # Reset visualized waypoint to start cell center
        self.anchor_time[env_ids] = self.sim_time  # Reset anti-stuck timer
        self.unstuck_until[env_ids] = -1.0  # Clear any active unstuck reflex

    # ------------------------------------------------------------------
    # 5. Telemetry & Route Visualization Helpers
    # ------------------------------------------------------------------
    def trace_planned_route(
        self, env_id: int = 0, start_cell: tuple[int, int] | None = None
    ) -> list[tuple[int, int]]:
        """Trace the sequence of cells [(i, j), ...] along the current Weighted Dijkstra route to goal_cell."""
        dist = self.weighted_flood_fill(self.ap.walls)[env_id]  # Cost-to-goal matrix (N, N) for env_id
        costs = self.cell_cost[env_id]  # Per-cell traversal cost matrix (N, N) for env_id
        walls = self.ap.walls[env_id]  # Known/unknown wall matrix (N, N, 4) for env_id
        cur = start_cell or (
            int(self.last_cell[env_id, 0].item()),
            int(self.last_cell[env_id, 1].item()),
        )  # Starting cell tuple
        route = [cur]  # Ordered list of cells along the planned route
        visited = {cur}  # Prevent cycles if all directions are equally blocked
        for _ in range(self.num_cells * self.num_cells):  # Step at most N^2 times
            if cur == self.goal_cell:  # Reached goal cell
                break
            ci, cj = cur  # Current cell coordinates in trace
            best_nb = None  # Best neighbor cell found
            best_val = 1e9  # Lowest total cost (dist + step_cost)
            for d, (di, dj) in enumerate(DIRS):  # Check East (0), North (1), West (2), South (3)
                if int(walls[ci, cj, d].item()) == 1:  # Confirmed wall blocks this edge
                    continue
                ni, nj = ci + di, cj + dj  # Neighbor coordinates
                if 0 <= ni < self.num_cells and 0 <= nj < self.num_cells and (ni, nj) not in visited:
                    val = float(dist[ni, nj].item() + costs[ni, nj].item())  # Total cost via neighbor
                    if val < best_val:  # Found cheaper next cell
                        best_val = val  # Update lowest cost
                        best_nb = (ni, nj)  # Update chosen neighbor
            if best_nb is None or best_val >= 1e5:  # No reachable path found
                break
            route.append(best_nb)  # Append next cell to route
            visited.add(best_nb)  # Mark cell visited in trace
            cur = best_nb  # Advance trace cursor
        return route  # Return ordered list of cells from current cell to goal

    def get_debug_snapshot(self, env_id: int = 0) -> dict:
        """Return a snapshot dict of the robot's saved map, hazard classifications, costs, and planned route."""
        dist = self.weighted_flood_fill(self.ap.walls)[env_id]  # Current cost-to-goal map (N, N)
        cur = (int(self.last_cell[env_id, 0].item()), int(self.last_cell[env_id, 1].item()))  # Current cell
        tgt = (int(self.ap.target[env_id, 0].item()), int(self.ap.target[env_id, 1].item()))  # Target cell
        return {
            "walls": self.ap.walls[env_id].detach().cpu().tolist(),  # (N, N, 4): 1=wall, 0=open, -1=unknown
            "cell_type": self.cell_type[env_id].detach().cpu().tolist(),  # (N, N): hazard type enum
            "cell_cost": self.cell_cost[env_id].detach().cpu().tolist(),  # (N, N): traversal cost
            "cell_deaths": self.cell_deaths[env_id].detach().cpu().tolist(),  # (N, N): reset count per cell
            "safe_offset": self.safe_offset[env_id].detach().cpu().tolist(),  # (N, N, 2): safe lane offsets [m]
            "dist": dist.detach().cpu().tolist(),  # (N, N): cumulative Dijkstra cost-to-goal
            "route": self.trace_planned_route(env_id, cur),  # Planned route cells [(i, j), ...]
            "current_cell": cur,  # Current cell (ci, cj)
            "target_cell": tgt,  # Immediate target cell (ti, tj)
            "waypoint": self.last_wp[env_id].detach().cpu().tolist(),  # Active local waypoint [wp_x, wp_y]
            "cmd": self.ap.cmd[env_id].detach().cpu().tolist(),  # Current command [v_x, v_y, w_z]
            "unstuck": self.sim_time < float(self.unstuck_until[env_id].item()),  # True during unstuck reflex
        }
