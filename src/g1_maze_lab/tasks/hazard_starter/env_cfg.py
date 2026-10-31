# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Challenge 7.5 Option B custom training tasks (``G1-Hazard-Starter`` and ``G1-Hazard-Pro``).

Custom sub-terrains replicate the exact geometry of ``g1_maze_lab.terrains.hazard_maze`` while keeping
a 20% flat/gentle-rough proportion so the trained policy excels on BOTH the Short Route and the Long Route:
    * ``trenches`` (25%) -> two concentric square trench rings (0.06 -> 0.24 m wide, 1.0 m deep) around spawn
    * ``ramps``    (30%) -> exact maze trapezoid prisms (0.75 m up, 0.50 m flat, 0.75 m down; 0.08 -> 0.31 m high)
    * ``boxes``    (25%) -> loose maze boxes (0.15-0.40 m wide, 0.02-0.09 m high) + 1 cm ice-like step plates
    * ``rough``    (20%) -> flat / gentle 1-4 cm rough ground for high-speed sprinting and 90-deg turns (Long Route)
    * ``friction``       -> randomized foot friction 0.2 (ice) to 1.0 (normal floor)
"""

from __future__ import annotations  # Enable postponed evaluation of type annotations

import numpy as np  # NumPy for mesh coordinate arrays and random number generation
import torch  # PyTorch for tensor operations in termination checks
import trimesh  # Trimesh for constructing 3D terrain meshes (boxes and convex-hull ramps)

import isaaclab.terrains as terrain_gen  # Built-in Isaac Lab heightfield/mesh terrain generators
from isaaclab.assets import Articulation  # Robot articulation class for reading root and joint states
from isaaclab.envs import ManagerBasedRLEnv  # Base manager-based RL environment class
from isaaclab.managers import CurriculumTermCfg as CurrTerm  # Configuration wrapper for curriculum terms
from isaaclab.managers import SceneEntityCfg  # Reference to a scene entity such as "robot"
from isaaclab.managers import TerminationTermCfg as DoneTerm  # Configuration wrapper for termination terms
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg  # Base classes for custom sub-terrains
from isaaclab.utils import configclass  # Decorator for Isaac Lab configuration dataclasses

from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1RoughPPORunnerCfg  # G1 Rough PPO config
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg  # G1 Rough base environment config

from ..curriculum import mdp as curr_mdp  # Week 7 curriculum terms (RewardWeightRamp, AdaptivePush, CommandRange)

STEPS_PER_IT = 24  # Number of environment steps per PPO iteration (num_steps_per_env = 24)


def physics_blowup(env: ManagerBasedRLEnv, max_speed: float = 20.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reset any robot whose physics state turns NaN/Inf or flies off (> max_speed [m/s]) before obs computation."""
    asset: Articulation = env.scene[asset_cfg.name]  # Get the robot articulation from the scene
    root = torch.cat((asset.data.root_pos_w.torch, asset.data.root_lin_vel_w.torch), dim=1)  # Root pos + lin vel (N, 6)
    joints = torch.cat((asset.data.joint_pos.torch, asset.data.joint_vel.torch), dim=1)  # Joint pos + vel (N, 2*J)
    bad = ~torch.isfinite(root).all(dim=1) | ~torch.isfinite(joints).all(dim=1)  # True if any state value is NaN/Inf
    # Also trigger reset if linear speed exceeds max_speed (20 m/s indicates solver instability)
    return bad | (torch.linalg.norm(torch.nan_to_num(asset.data.root_lin_vel_w.torch, nan=0.0), dim=1) > max_speed)


# ---------------------------------------------------------------------------
# Option B Custom Sub-Terrains (matching hazard_maze.py geometry)
# ---------------------------------------------------------------------------
def _box_mesh(cx: float, cy: float, cz: float, sx: float, sy: float, sz: float) -> trimesh.Trimesh:
    """Create an axis-aligned box mesh of size (sx, sy, sz) centered at (cx, cy, cz)."""
    return trimesh.creation.box((sx, sy, sz), trimesh.transformations.translation_matrix((cx, cy, cz)))


def _ring_slabs(cx: float, cy: float, r_in: float, r_out: float, depth: float) -> list[trimesh.Trimesh]:
    """Build four floor slabs forming a hollow square ring [r_in, r_out] around (cx, cy) with top at z = 0."""
    w = r_out - r_in  # Radial width of the ring band [m]
    mid = 0.5 * (r_in + r_out)  # Distance from center to the middle of the ring band [m]
    fz = -0.5 * depth  # Z center of each slab so its top surface sits at z = 0.0
    return [
        _box_mesh(cx, cy - mid, fz, 2.0 * r_out, w, depth),  # South slab spanning full outer width
        _box_mesh(cx, cy + mid, fz, 2.0 * r_out, w, depth),  # North slab spanning full outer width
        _box_mesh(cx - mid, cy, fz, w, 2.0 * r_in, depth),  # West slab fitting between North and South slabs
        _box_mesh(cx + mid, cy, fz, w, 2.0 * r_in, depth),  # East slab fitting between North and South slabs
    ]


def hazard_trench_terrain(difficulty: float, cfg: HazardTrenchTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate two concentric square trenches around the spawn pad so robots cross multiple trenches in any direction."""
    sx, sy = cfg.size  # Sub-terrain tile dimensions (8.0 m x 8.0 m)
    cx, cy = 0.5 * sx, 0.5 * sy  # Tile center (4.0, 4.0) where the robot spawns
    depth = cfg.pit_depth  # Depth of the trench pits (1.0 m, matching hazard_maze.py)
    # Interpolate trench width linearly with curriculum difficulty (0.06 m at diff=0 -> 0.24 m at diff=1)
    w = cfg.gap_width_range[0] + float(difficulty) * (cfg.gap_width_range[1] - cfg.gap_width_range[0])

    r0 = cfg.inner_platform_half_width  # Half-width of center spawn platform (1.1 m -> 2.2 m x 2.2 m pad)
    r1 = r0 + w  # Outer edge of Trench Ring #1
    r2 = r1 + cfg.corridor_ring_width  # Outer edge of middle flat ring (1.35 m wide recovery zone)
    r3 = r2 + w  # Outer edge of Trench Ring #2
    r4 = 0.5 * min(sx, sy)  # Outer boundary of the sub-terrain tile (4.0 m)

    meshes = [
        _box_mesh(cx, cy, -depth - 0.5, sx, sy, 1.0),  # Solid pit bottom across entire tile at z = -depth
        _box_mesh(cx, cy, -0.5 * depth, 2.0 * r0, 2.0 * r0, depth),  # Center spawn platform with top at z = 0
    ]
    meshes.extend(_ring_slabs(cx, cy, r1, r2, depth))  # Add middle floor ring between Trench #1 and Trench #2
    if r4 > r3 + 0.1:  # Add outer floor ring beyond Trench #2 if room remains inside the tile
        meshes.extend(_ring_slabs(cx, cy, r3, r4, depth))
    return meshes, np.array([cx, cy, 0.0])  # Return mesh list and spawn origin at (cx, cy, 0)


def hazard_trapezoid_ramp_terrain(
    difficulty: float, cfg: HazardTrapezoidRampTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate exact trapezoid prism ramps from hazard_maze.py (0.75 m up, 0.50 m flat, 0.75 m down) on all 4 sides."""
    sx, sy = cfg.size  # Sub-terrain tile dimensions (8.0 m x 8.0 m)
    cx, cy = 0.5 * sx, 0.5 * sy  # Tile center (4.0, 4.0)
    c = cfg.cell_size  # Total ramp length along travel direction (2.0 m, matching 1 maze cell)
    L = cfg.ramp_length  # Length of each sloped side (0.75 m -> leaves 0.50 m flat top in the middle)
    # Interpolate ramp height with difficulty (0.08 m -> 0.30 m: covers both Long Route and Short Route ramps)
    rh = cfg.ramp_height_range[0] + float(difficulty) * (cfg.ramp_height_range[1] - cfg.ramp_height_range[0])

    meshes = [_box_mesh(cx, cy, -0.5, sx, sy, 1.0)]  # Base flat ground slab with top at z = 0.0
    # 2D trapezoid cross-section (along_coord, z): base corners at z=-0.01, flat top corners at z=rh
    prof = [(-0.5 * c, -0.01), (0.5 * c, -0.01), (0.5 * c - L, rh), (-0.5 * c + L, rh)]
    r_mid = cfg.inner_half_width + 0.5 * c  # Radial distance from tile center to ramp centerline (2.0 m)
    span = cfg.inner_half_width + c  # Lateral half-length (3.0 m) so the 4 ramps form a closed square ring

    # Create East (+1) and West (-1) trapezoid ramps oriented along the X axis
    for sign in (-1.0, 1.0):
        rcx = cx + sign * r_mid  # X center of East/West ramp
        pts = [(rcx + u, cy + v, z) for u, z in prof for v in (-span, span)]  # 8 vertices of the trapezoid prism
        meshes.append(trimesh.convex.convex_hull(np.array(pts)))  # Build convex hull mesh (same as hazard_maze.py)

    # Create North (+1) and South (-1) trapezoid ramps oriented along the Y axis
    for sign in (-1.0, 1.0):
        rcy = cy + sign * r_mid  # Y center of North/South ramp
        pts = [(cx + v, rcy + u, z) for u, z in prof for v in (-span, span)]  # 8 vertices of the trapezoid prism
        meshes.append(trimesh.convex.convex_hull(np.array(pts)))  # Build convex hull mesh

    return meshes, np.array([cx, cy, 0.0])  # Return mesh list and spawn origin


def hazard_rough_boxes_terrain(
    difficulty: float, cfg: HazardRoughBoxesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate loose maze boxes (2-9 cm high, 15-40 cm wide) + 1 cm ice-like step plates matching hazard_maze.py."""
    sx, sy = cfg.size  # Sub-terrain tile dimensions (8.0 m x 8.0 m)
    cx, cy = 0.5 * sx, 0.5 * sy  # Tile center (4.0, 4.0)
    c = cfg.cell_size  # Virtual cell size (2.0 m)
    rng = np.random.default_rng(int(round(difficulty * 1e6)) + 2026)  # Deterministic RNG seeded by difficulty

    meshes = [_box_mesh(cx, cy, -0.5, sx, sy, 1.0)]  # Base ground slab with top at z = 0.0
    # Scale maximum box height with difficulty (0.02 m -> 0.09 m)
    max_h = cfg.height_range[0] + float(difficulty) * (cfg.height_range[1] - cfg.height_range[0])
    max_h = max(max_h, cfg.height_range[0] + 0.01)  # Ensure valid uniform sampling range

    nx, ny = int(sx // c), int(sy // c)  # Number of 2x2 m cells across the tile (4 x 4 = 16 cells)
    for i in range(nx):  # Loop over cell columns
        for j in range(ny):  # Loop over cell rows
            cell_cx = (i + 0.5) * c  # X center of cell (i, j)
            cell_cy = (j + 0.5) * c  # Y center of cell (i, j)
            # Place a 1 cm raised plate on alternating cells to mimic the 1 cm step edge of maze ice tiles
            if (i + j) % 2 == 1:
                meshes.append(_box_mesh(cell_cx, cell_cy, 0.0, c - 0.15, c - 0.15, 0.02))
            for _ in range(cfg.boxes_per_cell):  # Spawn random loose boxes inside cell (i, j)
                bx, by = rng.uniform(0.15, 0.40, size=2)  # Random box horizontal dimensions (15-40 cm)
                bh = float(rng.uniform(cfg.height_range[0], max_h))  # Random box height (2-9 cm)
                px = float(rng.uniform(-0.42 * c, 0.42 * c))  # Random X offset within cell
                py = float(rng.uniform(-0.42 * c, 0.42 * c))  # Random Y offset within cell
                wx, wy = cell_cx + px, cell_cy + py  # World XY center of the box
                # Keep a 1.3 m x 1.3 m area at the tile center clear so robot feet don't spawn inside a box
                if abs(wx - cx) < 0.65 and abs(wy - cy) < 0.65:
                    continue
                meshes.append(_box_mesh(wx, wy, 0.5 * bh, float(bx), float(by), bh))  # Add box mesh on floor

    return meshes, np.array([cx, cy, 0.0])  # Return mesh list and spawn origin


@configclass
class HazardTrenchTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the concentric multi-trench training sub-terrain."""

    function = hazard_trench_terrain  # Terrain builder function
    gap_width_range: tuple[float, float] = (0.06, 0.24)  # Trench width [m] from difficulty 0.0 to 1.0
    pit_depth: float = 1.0  # Trench depth [m]
    inner_platform_half_width: float = 1.1  # Half-width of center spawn platform [m]
    corridor_ring_width: float = 1.35  # Width of flat floor ring between inner and outer trenches [m]


@configclass
class HazardTrapezoidRampTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the maze-matched trapezoid prism ramp sub-terrain."""

    function = hazard_trapezoid_ramp_terrain  # Terrain builder function
    ramp_height_range: tuple[float, float] = (0.08, 0.30)  # Ramp height [m] from difficulty 0.0 to 1.0
    ramp_length: float = 0.75  # Slope length [m] on each side of the 0.50 m flat top
    cell_size: float = 2.0  # Total ramp footprint length [m]
    inner_half_width: float = 1.0  # Half-width of flat spawn area before ramp starts [m]


@configclass
class HazardRoughBoxesTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the loose-boxes + 1 cm step-plate sub-terrain."""

    function = hazard_rough_boxes_terrain  # Terrain builder function
    height_range: tuple[float, float] = (0.02, 0.09)  # Box height range [m] from difficulty 0.0 to 1.0
    boxes_per_cell: int = 10  # Number of random boxes per 2x2 m cell (matches hazard_maze.py)
    cell_size: float = 2.0  # Cell width [m]


# Terrain generator combining all 4 sub-terrains across 10 difficulty rows x 12 columns
OPTION_B_TERRAIN = TerrainGeneratorCfg(
    size=(8.0, 8.0),  # 8 m x 8 m per sub-terrain tile
    border_width=10.0,  # 10 m flat border around the terrain grid
    num_rows=10,  # 10 difficulty levels (0 = easy / Long Route -> 9 = hard / Short Route)
    num_cols=12,  # 12 columns of sub-terrains
    horizontal_scale=0.1,  # Heightfield horizontal resolution [m]
    vertical_scale=0.005,  # Heightfield vertical resolution [m]
    slope_threshold=0.75,  # Slope threshold for heightfield meshing
    use_cache=False,  # Rebuild fresh terrain meshes
    curriculum=True,  # Enable terrain difficulty curriculum across rows
    sub_terrains={
        # 25% concentric trenches (0.06 -> 0.24 m wide)
        "trenches": HazardTrenchTerrainCfg(proportion=0.25, gap_width_range=(0.06, 0.24)),
        # 30% exact maze trapezoid ramps (0.08 -> 0.30 m tall)
        "ramps": HazardTrapezoidRampTerrainCfg(proportion=0.30, ramp_height_range=(0.08, 0.30)),
        # 25% maze loose boxes (2-9 cm tall) + 1 cm step plates
        "boxes": HazardRoughBoxesTerrainCfg(proportion=0.25, height_range=(0.02, 0.09), boxes_per_cell=10),
        # 20% gentle 1-5 cm rough ground so the robot stays fast and stable on flat corridors & 90-deg turns
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20, noise_range=(0.01, 0.05), noise_step=0.01, border_width=0.25
        ),
    },
)


@configclass
class G1HazardStarterEnvCfg(G1RoughEnvCfg):
    """Option B custom training environment for Challenge 7.5 (excels on both Short Route and Long Route)."""

    def __post_init__(self):
        super().__post_init__()  # Initialize base G1RoughEnvCfg (observations and network shapes stay untouched)
        self.scene.num_envs = 1024  # Default number of parallel training environments
        self.scene.terrain.terrain_generator = OPTION_B_TERRAIN  # Replace default terrain with Option B hazard terrain
        self.scene.terrain.max_init_terrain_level = 3  # Spawn initial robots across rows 0..3

        # Randomize robot foot friction between 0.2 (maze ice) and 1.0 (normal maze floor)
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.0)

        # Use direct [v_x, v_y, w_z] velocity commands (matching challenge.py playback)
        self.commands.base_velocity.heading_command = False  # Disable heading controller during training
        self.commands.base_velocity.rel_standing_envs = 0.02  # 2% standing environments
        self.commands.base_velocity.ranges.lin_vel_x = (-0.25, 0.85)  # Train reverse (-0.25) and forward (0.85 -> 1.35)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.35, 0.35)  # Train lateral strafing for dodging spike beds
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)  # Train responsive yaw turns for twisty corridors

        # Terrain curriculum: the default terrain_levels_vel needs straight walking (> 4 m from spawn), but random
        # yaw-rate commands make the robots walk in circles, so levels sank to row ~0.9 (ramps <= 0.13 m).
        # Promote on survival + good tracking + leaving the spawn platform instead.
        self.curriculum.terrain_levels = CurrTerm(
            func=curr_mdp.terrain_levels_skill,  # Skill-based terrain level curriculum
            params={
                "reward_term": "track_lin_vel_xy_exp",  # Velocity tracking reward to judge walking quality
                "fall_terms": ("base_contact", "physics_blowup"),  # Terminations that count as a fall
                "up_threshold": 0.75,  # Move up when tracking >= 75% of max (and no fall, left platform)
                "down_threshold": 0.5,  # Move down when tracking < 50% of max (stuck at a ramp or trench)
                "leave_distance": 2.0,  # Must end > 2.0 m from spawn (past the ramp crest / inner trench)
            },
        )

        # Curriculum 1: Ramp feet_air_time reward weight (0.25 -> 0.65) so feet clear trenches & boxes cleanly
        self.curriculum.feet_air_time_ramp = CurrTerm(
            func=curr_mdp.RewardWeightRampCurriculum,  # Linear reward weight ramp curriculum
            params={
                "term_name": "feet_air_time",  # Target reward term
                "start_weight": 0.25,  # Initial weight from G1 Rough
                "end_weight": 0.65,  # Moderate high-stepping weight (preserves fast flat cornering)
                "start_step": 20 * STEPS_PER_IT,  # Begin ramping at iteration 20
                "end_step": 220 * STEPS_PER_IT,  # Finish ramping at iteration 220
            },
        )
        # Curriculum 2: Ramp external velocity pushes (0.0 -> 0.8 m/s) to train slip recovery on ice
        self.curriculum.push_strength = CurrTerm(
            func=curr_mdp.AdaptivePushCurriculum,  # Adaptive push curriculum based on fall rate
            params={
                "event_name": "push_robot",  # Target push event
                "fall_term": "base_contact",  # Termination term counting falls
                "start_vel": 0.0,  # Start with no pushes
                "max_vel": 0.8,  # Cap push velocity at 0.8 m/s
                "step": 0.05,  # Increase/decrease push speed by 0.05 m/s
                "raise_below": 0.05,  # Increase push when fall rate < 5%
                "lower_above": 0.15,  # Decrease push when fall rate > 15%
                "update_every": 20 * STEPS_PER_IT,  # Evaluate every 20 iterations
            },
        )
        # Curriculum 3: Widen max forward speed command (0.85 -> 1.35 m/s) as tracking improves
        self.curriculum.command_vel_x = CurrTerm(
            func=curr_mdp.CommandRangeCurriculum,  # Adaptive command-range curriculum
            params={
                "command_name": "base_velocity",  # Target command term
                "reward_term": "track_lin_vel_xy_exp",  # Velocity tracking reward to monitor
                "start_max": 0.85,  # Initial max forward speed [m/s]
                "final_max": 1.35,  # Final max forward speed [m/s]
                "step": 0.05,  # Increment by 0.05 m/s per update
                "threshold": 0.78,  # Widen speed range when tracking reward >= 78% of max weight
                "update_every": 20 * STEPS_PER_IT,  # Check every 20 iterations
            },
        )

        # Reset individual robots whose physics state turns NaN/Inf on steep ramp or trench edges
        self.terminations.physics_blowup = DoneTerm(func=physics_blowup)


@configclass
class G1HazardProEnvCfg(G1HazardStarterEnvCfg):
    """Alias for G1HazardStarterEnvCfg registered under the G1-Hazard-Pro task name."""

    pass


@configclass
class G1HazardStarterPPORunnerCfg(G1RoughPPORunnerCfg):
    """PPO runner config for G1-Hazard-Starter (keeps G1 Rough actor/critic network architecture)."""

    def __post_init__(self):
        super().__post_init__()  # Inherit G1 Rough network dimensions
        self.experiment_name = "g1_hazard_starter"  # Log directory name under logs/rsl_rl/


@configclass
class G1HazardProPPORunnerCfg(G1RoughPPORunnerCfg):
    """PPO runner config for G1-Hazard-Pro (keeps G1 Rough actor/critic network architecture)."""

    def __post_init__(self):
        super().__post_init__()  # Inherit G1 Rough network dimensions
        self.experiment_name = "g1_hazard_pro"  # Log directory name under logs/rsl_rl/
