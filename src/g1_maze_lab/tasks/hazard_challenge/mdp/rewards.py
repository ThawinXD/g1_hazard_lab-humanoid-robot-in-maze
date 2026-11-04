"""Reward terms for ``G1-Hazard-Challenge``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_y_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Sideways speed tracking in the gravity-aligned robot frame: ``exp(-(v_y,cmd - v_y)^2 / std^2)``.

    G1 Rough's ``track_lin_vel_xy_exp`` (std 0.5) forgives sideways errors: missing a 0.35 m/s strafe by 0.17 m/s
    costs 11% of it and the left drift (~0.07 m/s) about 2%, so ``model_8497`` strafes at about 0.18 m/s. With a
    tighter ``std`` (0.2) the same miss costs about half of this term and the drift about 10%.
    """
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3])
    error = env.command_manager.get_command(command_name)[:, 1] - vel_yaw[:, 1]
    return torch.exp(-torch.square(error) / std**2)
