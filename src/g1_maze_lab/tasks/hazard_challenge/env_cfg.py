"""``G1-Hazard-Challenge``: ``G1-Hazard-Pro`` with training closer to what the challenge asks of the policy.

Same robot, observations, network, PPO settings, curricula (terrain_levels_skill, feet air time, pushes, forward
speed) and terminations as ``G1-Hazard-Pro``. The changes:

1. Real ice: the box tile's plates are maze ice cells (friction 0.2 with ``mjc:priority``), see ``terrain.py``. The
   foot-friction randomization is set to 1.0: with Newton's max-friction rule it never had any effect.
2. Commands like ``pilot.py``'s: kinds of command (slow straight, slow turning, fast, turn on the spot, sideways,
   stand, random) held 2-6 s and ramped with the pilot's acceleration limits, see ``mdp/commands.py``.
3. A sideways speed reward with a tight tolerance (``track_lin_vel_y_exp``, std 0.2). ``track_lin_vel_xy_exp`` is
   unchanged because the terrain curriculum promotes robots by it.
4. Harder trenches (0.10-0.26 m) and ramps (0.12-0.32 m), see ``terrain.py``.
"""

import copy

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ...terrains import HazardTerrainImporterCfg
from ..hazard_starter.env_cfg import G1HazardProEnvCfg, G1HazardProPPORunnerCfg
from . import mdp
from .terrain import CHALLENGE_TERRAIN


@configclass
class G1HazardChallengeEnvCfg(G1HazardProEnvCfg):
    """Challenge-like training: real ice, pilot-like commands, sideways tracking, harder hazards (module docstring)."""

    def __post_init__(self):
        super().__post_init__()  # G1-Hazard-Pro: terrain, curricula, commands, terminations

        # 1. Terrain with maze ice tiles (the hazard importer adds them as a separate low-friction mesh)
        old = self.scene.terrain
        self.scene.terrain = HazardTerrainImporterCfg(
            prim_path=old.prim_path,
            terrain_type="generator",
            terrain_generator=copy.deepcopy(CHALLENGE_TERRAIN),  # play_mode() edits the grid in place
            max_init_terrain_level=old.max_init_terrain_level,  # Spawn across rows 0..3 (G1-Hazard-Starter)
            collision_group=old.collision_group,
            physics_material=old.physics_material,  # Normal floor: friction 1.0
            visual_material=old.visual_material,
            debug_vis=False,
        )
        # Feet friction: the larger of the two shapes' friction is used, so on the 1.0 floor any value below 1.0 did
        # nothing; ice gets its own 0.2 through mjc:priority whatever the feet have
        self.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)

        # 2. Pilot-like commands (the forward-speed curriculum keeps widening ranges.lin_vel_x[1] up to 1.35 m/s)
        old_cmd = self.commands.base_velocity
        self.commands.base_velocity = mdp.PilotLikeVelocityCommandCfg(
            asset_name=old_cmd.asset_name,
            ranges=old_cmd.ranges,  # Used by the "random" kind: v_x -0.25..max, v_y +-0.35, w_z +-1.2
            rel_standing_envs=0.0,  # Standing is the "stand" kind
            debug_vis=old_cmd.debug_vis,
        )

        # 3. Sideways speed tracking with a tight tolerance
        self.rewards.track_lin_vel_y_exp = RewTerm(
            func=mdp.track_lin_vel_y_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.2},
        )


@configclass
class G1HazardChallengePPORunnerCfg(G1HazardProPPORunnerCfg):
    """PPO runner config for G1-Hazard-Challenge (G1 Rough actor/critic shapes, so policy.pt stays uploadable)."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_hazard_challenge"  # Log directory name under logs/rsl_rl/
