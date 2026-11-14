"""Hazard test tasks: many G1 robots, each in a walled corridor (maze dimensions) with one hazard.

``G1-Trench-Test`` (a trench), ``G1-Ramp-Test`` (the maze's ramp), ``G1-Ice-Test`` (one cell of ice),
``G1-Ice-Corner-Test`` (an L-shaped corridor turning at an ice cell), ``G1-Rough-Test`` (corridors of rough cells),
``G1-Corner-Test`` (rough L corners between flat cells) and ``G1-Strafe-Test`` (open floor without walls). Same base
config, observations and crash rule as the challenge's maze tasks (``tasks/maze/env_cfg.py``): a G1 Rough policy,
commands written directly, and torso / pelvis / elbows / palms touching anything (floor or wall) ends the episode.
``scripts/trench_test.py`` and ``scripts/ice_turn_test.py`` place the robots and drive them.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

from ...terrains import HazardTerrainGeneratorCfg, HazardTerrainImporterCfg
from ..hazard_starter.env_cfg import physics_blowup
from ..maze import mdp as maze_mdp
from .terrain import (
    ICE_CORNER_NEXT,
    RAMP_HEIGHTS,
    ROUGH_KINDS,
    TRENCH_WIDTHS,
    CorridorIceCornerTerrainCfg,
    CorridorIceTerrainCfg,
    CorridorRampTerrainCfg,
    CorridorTrenchTerrainCfg,
    OpenFloorTerrainCfg,
    RoughCorridorTerrainCfg,
)

TILE_SIZE = (8.0, 3.0)  # Corridor tile: 8 m long (trench in the middle), 3 m wide (walls at +-1.0 m)
CORNER_KINDS = ("left", "right")  # G1-Corner-Test rows
NUM_COLS = 20  # Corridors per trench width


@configclass
class G1TrenchTestEnvCfg(G1RoughEnvCfg):
    """Rows = trench widths (``TRENCH_WIDTHS``), columns = copies; robots start ``start_distance`` before the trench."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 500
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=TILE_SIZE,
            border_width=2.0,
            border_height=0.0,
            num_rows=len(TRENCH_WIDTHS),
            num_cols=NUM_COLS,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,  # Row r gets difficulty in [r, r+1) / rows -> trench width TRENCH_WIDTHS[r]
            sub_terrains={"corridor_trench": CorridorTrenchTerrainCfg(proportion=1.0, widths=TRENCH_WIDTHS)},
        )
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None  # Robots stay on their row (the script also assigns them)
        # commands: the test script writes v_x / v_y / w_z directly (as challenge.py does)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.debug_vis = False
        # events: start on the spawn point facing +x, no pushes (the script sets heading / lateral offsets)
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
        }
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        # terminations: the challenge's crash rule, plus a reset if the physics blows up
        self.episode_length_s = 1.0e6
        self.terminations.body_contact = DoneTerm(
            func=maze_mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["torso_link", "pelvis", ".*_elbow_.*", ".*_palm_link"]
                ),
                "threshold": 1.0,
            },
        )
        self.terminations.base_contact = None
        self.terminations.physics_blowup = DoneTerm(func=physics_blowup)

    def play_mode(self):
        # keep the corridor grid (the shared play mode turns the terrain into 5 x 5 random tiles)
        gen = self.scene.terrain.terrain_generator
        rows, cols, curriculum = gen.num_rows, gen.num_cols, gen.curriculum
        super().play_mode()
        gen.num_rows, gen.num_cols, gen.curriculum = rows, cols, curriculum
        self.scene.terrain.max_init_terrain_level = None


@configclass
class G1RampTestEnvCfg(G1TrenchTestEnvCfg):
    """Same corridors with the maze's ramp instead of a trench: rows = ramp heights (``RAMP_HEIGHTS``)."""

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.num_rows = len(RAMP_HEIGHTS)
        gen.sub_terrains = {"corridor_ramp": CorridorRampTerrainCfg(proportion=1.0, heights=RAMP_HEIGHTS)}


@configclass
class G1StrafeTestEnvCfg(G1TrenchTestEnvCfg):
    """Open flat floor without walls, for sideways stepping and straight-walking drift (``scripts/strafe_test.py``)."""

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.size = (8.0, 8.0)
        gen.num_rows = 1
        gen.num_cols = 10
        gen.sub_terrains = {"open_floor": OpenFloorTerrainCfg(proportion=1.0)}


@configclass
class G1RoughTestEnvCfg(G1TrenchTestEnvCfg):
    """Walled corridors of maze rough cells; rows = corridor kinds (``ROUGH_KINDS``) (``scripts/rough_test.py``)."""

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.size = (8.0, 8.0)
        gen.num_rows = len(ROUGH_KINDS)
        gen.num_cols = 25
        gen.seed = 1  # Same boxes in every run, so checkpoints are compared on the same floor
        gen.sub_terrains = {"rough_corridor": RoughCorridorTerrainCfg(proportion=1.0)}


@configclass
class G1CornerTestEnvCfg(G1RoughTestEnvCfg):
    """Rough L corners only (rows: left / right turn); the cells before and after the corner cell are flat, as spike
    cells are (``scripts/corner_lineup_test.py``)."""

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.num_rows = len(CORNER_KINDS)
        gen.sub_terrains = {
            "rough_corridor": RoughCorridorTerrainCfg(
                proportion=1.0, kinds=CORNER_KINDS, bare_cells=("before", "after")
            )
        }


@configclass
class G1IceTestEnvCfg(G1TrenchTestEnvCfg):
    """Same corridors with one maze cell of ice (friction 0.2, the challenge's) instead of a trench; rows are copies."""

    def __post_init__(self):
        super().__post_init__()
        old = self.scene.terrain
        gen = old.terrain_generator
        self.scene.terrain = HazardTerrainImporterCfg(
            prim_path=old.prim_path,
            terrain_type="generator",
            terrain_generator=HazardTerrainGeneratorCfg(
                size=gen.size,
                border_width=gen.border_width,
                border_height=gen.border_height,
                num_rows=gen.num_rows,
                num_cols=gen.num_cols,
                horizontal_scale=gen.horizontal_scale,
                vertical_scale=gen.vertical_scale,
                slope_threshold=gen.slope_threshold,
                use_cache=False,
                curriculum=True,
                sub_terrains={"corridor_ice": CorridorIceTerrainCfg(proportion=1.0)},
            ),
            max_init_terrain_level=None,
            collision_group=old.collision_group,
            physics_material=old.physics_material,
            visual_material=old.visual_material,
            debug_vis=False,
        )


@configclass
class G1IceCornerTestEnvCfg(G1IceTestEnvCfg):
    """L-shaped corridors with an ice corner cell; rows = what follows the corner (``ICE_CORNER_NEXT``)."""

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.size = (8.0, 8.0)
        gen.num_rows = len(ICE_CORNER_NEXT)
        gen.num_cols = 25
        gen.sub_terrains = {"corridor_ice_corner": CorridorIceCornerTerrainCfg(proportion=1.0)}
