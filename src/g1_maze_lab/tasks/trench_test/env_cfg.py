"""``G1-Trench-Test``: many G1 robots, each in a walled corridor (maze dimensions) with one trench across it.

Same base config, observations and crash rule as the challenge's maze tasks (``tasks/maze/env_cfg.py``): a G1 Rough
policy, commands written directly, and torso / pelvis / elbows / palms touching anything (floor or wall) ends the
episode. ``scripts/trench_test.py`` places the robots and drives them toward the trench.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

from ..hazard_starter.env_cfg import physics_blowup
from ..maze import mdp as maze_mdp
from .terrain import TRENCH_WIDTHS, CorridorTrenchTerrainCfg

TILE_SIZE = (8.0, 3.0)  # Corridor tile: 8 m long (trench in the middle), 3 m wide (walls at +-1.0 m)
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
