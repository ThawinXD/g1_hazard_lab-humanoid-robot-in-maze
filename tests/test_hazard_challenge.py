"""Unit tests for the G1-Hazard-Challenge training task (terrain, commands, reward, config), no simulator required."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

from g1_maze_lab.tasks.hazard_challenge.mdp.commands import (
    FAST,
    KINDS,
    RANDOM,
    ROTATE,
    SIDEWAYS,
    SLOW,
    SLOW_TURN,
    STAND,
    PilotLikeVelocityCommand,
    PilotLikeVelocityCommandCfg,
    ramp_toward,
    sample_targets,
)
from g1_maze_lab.tasks.hazard_challenge.mdp.rewards import track_lin_vel_y_exp
from g1_maze_lab.tasks.hazard_challenge.terrain import (
    CHALLENGE_TERRAIN,
    HazardIceBoxesTerrainCfg,
    hazard_ice_boxes_terrain,
)


def _cmd_cfg(vx_max: float = 1.35) -> PilotLikeVelocityCommandCfg:
    ranges = UniformVelocityCommandCfg.Ranges(lin_vel_x=(-0.25, vx_max), lin_vel_y=(-0.35, 0.35), ang_vel_z=(-1.2, 1.2))
    return PilotLikeVelocityCommandCfg(asset_name="robot", ranges=ranges)


def test_ice_boxes_tile_puts_maze_ice_on_every_other_cell():
    """8 of the 16 cells are 1.85 m ice tiles (relative to the centre spawn point) for the hazard importer; the other
    8 hold loose boxes, none of them on an ice cell or in the spawn area."""
    from g1_maze_lab.terrains.hazard_maze import _PENDING_ICE

    _PENDING_ICE.append((9.0, 9.0, 1.0, 1.0))  # Left over from another terrain: must be replaced
    cfg = HazardIceBoxesTerrainCfg(size=(8.0, 8.0))
    meshes, origin = hazard_ice_boxes_terrain(0.6, cfg)
    assert np.allclose(origin, [4.0, 4.0, 0.0])
    ice = sorted(_PENDING_ICE)
    expected = sorted(
        ((i + 0.5) * 2 - 4.0, (j + 0.5) * 2 - 4.0, 1.85, 1.85) for i in range(4) for j in range(4) if (i + j) % 2
    )
    assert ice == expected, f"ice tiles {ice}"
    boxes = meshes[1:]
    assert len(boxes) > 50, f"only {len(boxes)} boxes"
    max_h = cfg.height_range[0] + 0.6 * (cfg.height_range[1] - cfg.height_range[0])
    for m in boxes:
        x, y, _ = m.bounds.mean(0)
        assert (int(x // 2) + int(y // 2)) % 2 == 0, f"box centred on an ice cell at ({x:.2f}, {y:.2f})"
        assert not (abs(x - 4.0) < 0.65 and abs(y - 4.0) < 0.65), "box in the spawn area"
        assert m.bounds[1][2] <= max_h + 1e-9
    _PENDING_ICE.clear()


def test_challenge_terrain_is_hazard_pro_with_harder_hazards():
    """Same grid and proportions as G1-Hazard-Pro, the hazard generator (collects the ice), wider trenches, taller
    ramps."""
    from g1_maze_lab.tasks.hazard_starter.env_cfg import OPTION_B_TERRAIN
    from g1_maze_lab.terrains import HazardTerrainGeneratorCfg

    assert isinstance(CHALLENGE_TERRAIN, HazardTerrainGeneratorCfg)
    for field in ("size", "num_rows", "num_cols", "border_width", "curriculum"):
        assert getattr(CHALLENGE_TERRAIN, field) == getattr(OPTION_B_TERRAIN, field), field
    subs = CHALLENGE_TERRAIN.sub_terrains
    assert list(subs) == ["trenches", "ramps", "ice_boxes", "rough"]
    assert math.isclose(sum(s.proportion for s in subs.values()), 1.0)
    assert subs["trenches"].gap_width_range == (0.10, 0.26)
    assert subs["ramps"].ramp_height_range == (0.12, 0.32)
    # At the curriculum's average row (6.4) robots meet about the maze's short-route sizes
    d = (6.4 + 0.5) / 10
    assert 0.20 < 0.10 + d * 0.16 < 0.22 and 0.25 < 0.12 + d * 0.20 < 0.27


def test_each_command_kind_stays_in_its_range():
    cfg = _cmd_cfg()
    n = 4000
    out = {k: sample_targets(torch.full((n,), k), cfg) for k in range(len(KINDS))}
    vx, vy, wz = out[SLOW].unbind(-1)
    assert vx.min() >= 0.15 and vx.max() <= 0.75 and vy.abs().max() <= 0.1 and wz.abs().max() <= 0.3
    vx, vy, wz = out[SLOW_TURN].unbind(-1)
    assert vx.min() >= 0.15 and vx.max() <= 0.75 and wz.abs().min() >= 0.3 and wz.abs().max() <= 1.1
    assert (wz > 0).any() and (wz < 0).any()
    vx, vy, wz = out[FAST].unbind(-1)
    assert vx.min() >= 0.75 and vx.max() <= 1.35 and vx.max() > 1.3 and wz.abs().max() <= 0.8
    vx, vy, wz = out[ROTATE].unbind(-1)
    assert vx.abs().max() == 0 and vy.abs().max() == 0 and wz.abs().min() >= 0.4 and wz.abs().max() <= 1.0
    assert (wz > 0).any() and (wz < 0).any()
    vx, vy, wz = out[SIDEWAYS].unbind(-1)
    assert vx.min() >= -0.15 and vx.max() <= 0.05 and vy.abs().min() >= 0.25 and vy.abs().max() <= 0.5
    assert (vy > 0).any() and (vy < 0).any() and wz.abs().max() <= 0.1
    assert out[STAND].abs().max() == 0
    vx, vy, wz = out[RANDOM].unbind(-1)
    assert vx.min() >= -0.25 and vx.max() <= 1.35 and vy.abs().max() <= 0.35 and wz.abs().max() <= 1.2


def test_fast_kind_follows_the_forward_speed_curriculum():
    """The curriculum widens ranges.lin_vel_x[1] from 0.85: fast commands never exceed it."""
    vx = sample_targets(torch.full((2000,), FAST), _cmd_cfg(vx_max=0.85))[:, 0]
    assert vx.min() >= 0.75 and vx.max() <= 0.85


def test_mixed_kinds_get_their_own_values():
    kinds = torch.tensor([SLOW, ROTATE, STAND, SIDEWAYS] * 50)
    out = sample_targets(kinds, _cmd_cfg())
    assert (out[kinds == ROTATE][:, :2] == 0).all() and (out[kinds == STAND] == 0).all()
    assert (out[kinds == SLOW][:, 0] >= 0.15).all() and (out[kinds == SIDEWAYS][:, 1].abs() >= 0.25).all()


def test_weights_cover_every_kind_and_favour_slow_walking():
    cfg = _cmd_cfg()
    assert len(cfg.kind_weights) == len(KINDS) and math.isclose(sum(cfg.kind_weights), 1.0)
    w = dict(zip(KINDS, cfg.kind_weights, strict=True))
    assert w["slow"] + w["slow_turn"] >= 0.5 and w["rotate"] >= 0.1 and w["sideways"] >= 0.1
    assert cfg.resampling_time_range == (2.0, 6.0) and cfg.class_type is PilotLikeVelocityCommand


def test_ramp_toward_uses_the_pilot_acceleration_limits():
    acc = torch.tensor(_cmd_cfg().acc_limits)
    cmd = torch.zeros(2, 3)
    target = torch.tensor([[1.0, -0.3, 1.0], [0.01, 0.0, -0.02]])
    out = ramp_toward(cmd, target, acc, 0.02)
    assert torch.allclose(out[0], torch.tensor([0.04, -0.04, 0.056]))
    assert torch.allclose(out[1], target[1])  # Within one step: reached
    for _ in range(100):
        cmd = ramp_toward(cmd, target, acc, 0.02)
    assert torch.allclose(cmd, target)


def _fake_env(yaw: float, vel_w: tuple[float, float], cmd_vy: float):
    q = torch.tensor([[0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]])  # xyzw
    data = SimpleNamespace(
        root_quat_w=SimpleNamespace(torch=q),
        root_lin_vel_w=SimpleNamespace(torch=torch.tensor([[vel_w[0], vel_w[1], 0.0]])),
    )
    command = torch.tensor([[0.0, cmd_vy, 0.0]])
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )


def test_sideways_reward_uses_the_robot_frame_and_a_tight_tolerance():
    # Facing +y and moving +y in the world is forward, not sideways: a 0.35 m/s strafe command is missed completely
    r = track_lin_vel_y_exp(_fake_env(0.5 * math.pi, (0.0, 0.35), 0.35), std=0.2, command_name="base_velocity")
    assert math.isclose(float(r), math.exp(-(0.35**2) / 0.04), rel_tol=1e-4)
    # Facing +y, the robot's left is -x: moving -x at 0.35 m/s is exactly the commanded strafe
    r = track_lin_vel_y_exp(_fake_env(0.5 * math.pi, (-0.35, 0.0), 0.35), std=0.2, command_name="base_velocity")
    assert math.isclose(float(r), 1.0, rel_tol=1e-4)
    # Strafing at 0.18 m/s when asked for 0.35 (model_8497) costs about half of the term
    r = track_lin_vel_y_exp(_fake_env(0.0, (0.0, 0.18), 0.35), std=0.2, command_name="base_velocity")
    assert 0.4 < float(r) < 0.55


def test_challenge_env_cfg():
    """Hazard importer with the challenge terrain (a copy), real-ice comment true to the config, pilot-like commands,
    the sideways reward, and everything the curricula read left as in G1-Hazard-Pro."""
    import gymnasium as gym

    import g1_maze_lab.tasks  # noqa: F401
    from g1_maze_lab.tasks.hazard_challenge.env_cfg import G1HazardChallengeEnvCfg, G1HazardChallengePPORunnerCfg
    from g1_maze_lab.tasks.hazard_starter.env_cfg import G1HazardProEnvCfg
    from g1_maze_lab.terrains import HazardTerrainImporterCfg

    cfg, pro = G1HazardChallengeEnvCfg(), G1HazardProEnvCfg()
    assert isinstance(cfg.scene.terrain, HazardTerrainImporterCfg) and cfg.scene.terrain.ice_friction == 0.2
    assert cfg.scene.terrain.terrain_generator is not CHALLENGE_TERRAIN
    assert list(cfg.scene.terrain.terrain_generator.sub_terrains) == list(CHALLENGE_TERRAIN.sub_terrains)
    assert cfg.scene.terrain.max_init_terrain_level == pro.scene.terrain.max_init_terrain_level
    assert cfg.events.physics_material.params["static_friction_range"] == (1.0, 1.0)
    assert isinstance(cfg.commands.base_velocity, PilotLikeVelocityCommandCfg)
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (-0.25, 0.85)  # Start of the forward-speed curriculum
    assert cfg.rewards.track_lin_vel_y_exp.params["std"] == 0.2 and cfg.rewards.track_lin_vel_y_exp.weight == 0.5
    assert cfg.rewards.track_lin_vel_xy_exp.params == pro.rewards.track_lin_vel_xy_exp.params
    assert cfg.rewards.track_lin_vel_xy_exp.weight == pro.rewards.track_lin_vel_xy_exp.weight
    assert cfg.curriculum.terrain_levels.params == pro.curriculum.terrain_levels.params
    assert cfg.curriculum.command_vel_x.params == pro.curriculum.command_vel_x.params
    assert G1HazardChallengePPORunnerCfg().experiment_name == "g1_hazard_challenge"
    assert "G1-Hazard-Challenge" in gym.registry
