"""Velocity commands shaped like ``pilot.py``'s, for training (``G1-Hazard-Challenge``).

Isaac Lab's ``UniformVelocityCommand`` draws ``(v_x, v_y, w_z)`` independently and holds them for 10 s. Measured over
16 benchmark runs, the pilot sends something else: mostly slow straight walking (41% of the time; 9% of training
commands), turns on the spot on ice, pure sideways steps in the slalom, and it changes its command every step under
acceleration limits. This command term first picks a *kind* of command per robot (``KINDS``, ``kind_weights``), then
its values, holds it for ``resampling_time_range`` and ramps the command toward it with the pilot's acceleration
limits, starting from zero at every episode start.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass, index_fill_

KINDS = ("slow", "slow_turn", "fast", "rotate", "sideways", "stand", "random")
SLOW, SLOW_TURN, FAST, ROTATE, SIDEWAYS, STAND, RANDOM = range(len(KINDS))


def _uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo


def _sign(n: int, device) -> torch.Tensor:
    return torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)


def sample_targets(kinds: torch.Tensor, cfg: PilotLikeVelocityCommandCfg) -> torch.Tensor:
    """Command targets ``(n, 3)`` = ``[v_x, v_y, w_z]`` for robots with the given kinds (indices into ``KINDS``)."""
    n, dev, r = len(kinds), kinds.device, cfg.ranges
    vx_max = max(float(r.lin_vel_x[1]), cfg.fast_min)  # The forward-speed curriculum widens lin_vel_x[1]
    small_vy = _uniform(-cfg.small_vy, cfg.small_vy, n, dev)  # The pilot's lane-keeping corrections
    straight_wz = _uniform(-cfg.straight_wz, cfg.straight_wz, n, dev)
    per_kind = {
        SLOW: (_uniform(*cfg.slow_speed, n, dev), small_vy, straight_wz),
        SLOW_TURN: (_uniform(*cfg.slow_speed, n, dev), small_vy, _sign(n, dev) * _uniform(*cfg.turn_wz, n, dev)),
        FAST: (_uniform(cfg.fast_min, vx_max, n, dev), small_vy, _uniform(-cfg.fast_wz, cfg.fast_wz, n, dev)),
        ROTATE: (
            torch.zeros(n, device=dev),
            torch.zeros(n, device=dev),
            _sign(n, dev) * _uniform(*cfg.rotate_wz, n, dev),
        ),
        SIDEWAYS: (
            _uniform(*cfg.sideways_vx, n, dev),
            _sign(n, dev) * _uniform(*cfg.sideways_vy, n, dev),
            _uniform(-cfg.sideways_wz, cfg.sideways_wz, n, dev),
        ),
        STAND: (torch.zeros(n, device=dev),) * 3,
        RANDOM: (_uniform(*r.lin_vel_x, n, dev), _uniform(*r.lin_vel_y, n, dev), _uniform(*r.ang_vel_z, n, dev)),
    }
    out = torch.zeros(n, 3, device=dev)
    for kind, values in per_kind.items():
        sel = kinds == kind
        out[sel] = torch.stack(values, dim=-1)[sel]
    return out


def ramp_toward(command: torch.Tensor, target: torch.Tensor, acc: torch.Tensor, dt: float) -> torch.Tensor:
    """One step of the pilot's rate limit: move ``command`` toward ``target`` by at most ``acc * dt`` per axis."""
    step = acc * dt
    return command + (target - command).clamp(-step, step)


class PilotLikeVelocityCommand(UniformVelocityCommand):
    """``UniformVelocityCommand`` with pilot-like command kinds and rate limits (see the module docstring)."""

    cfg: PilotLikeVelocityCommandCfg

    def __init__(self, cfg: PilotLikeVelocityCommandCfg, env):
        super().__init__(cfg, env)
        if len(cfg.kind_weights) != len(KINDS):
            raise ValueError(f"kind_weights needs one weight per kind {KINDS}, got {cfg.kind_weights}")
        self._weights = torch.tensor(cfg.kind_weights, dtype=torch.float, device=self.device)
        self._acc = torch.tensor(cfg.acc_limits, dtype=torch.float, device=self.device)
        self.kind = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # Current kind per robot
        self.target_b = torch.zeros(self.num_envs, 3, device=self.device)  # Command the ramp is heading for

    def __str__(self) -> str:
        weights = ", ".join(f"{k} {w:.2f}" for k, w in zip(KINDS, self.cfg.kind_weights, strict=True))
        return (
            f"PilotLikeVelocityCommand:\n\tKinds: {weights}\n\tResampling time range: {self.cfg.resampling_time_range}"
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        extras = super().reset(env_ids)
        index_fill_(self.vel_command_b, env_ids, 0.0)  # Every episode starts from standing, as the pilot does
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        kinds = torch.multinomial(self._weights, n, replacement=True)
        self.kind[env_ids] = kinds
        self.target_b[env_ids] = sample_targets(kinds, self.cfg)
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = kinds == STAND

    def _update_command(self):
        self.vel_command_b[:] = ramp_toward(self.vel_command_b, self.target_b, self._acc, self._env.step_dt)


@configclass
class PilotLikeVelocityCommandCfg(UniformVelocityCommandCfg):
    """Pilot-like velocity commands (:class:`PilotLikeVelocityCommand`). ``ranges`` is used by the ``random`` kind and
    ``ranges.lin_vel_x[1]`` caps the ``fast`` kind (the forward-speed curriculum widens it)."""

    class_type: type = PilotLikeVelocityCommand
    heading_command: bool = False
    resampling_time_range: tuple[float, float] = (2.0, 6.0)  # The pilot changes its command often [s]

    kind_weights: tuple[float, ...] = (0.30, 0.20, 0.15, 0.10, 0.10, 0.05, 0.10)
    """Share of each kind in ``KINDS`` order: slow, slow_turn, fast, rotate, sideways, stand, random."""
    acc_limits: tuple[float, float, float] = (2.0, 2.0, 2.8)
    """Rate limits for ``v_x``, ``v_y`` [m/s^2] and ``w_z`` [rad/s^2] (pilot.py ``ap.max_acc`` / ``ap.max_ang_acc``)."""

    slow_speed: tuple[float, float] = (0.15, 0.75)  # slow / slow_turn forward speed [m/s] (hazard speed limits)
    fast_min: float = 0.75  # fast: forward speed from here to ranges.lin_vel_x[1] [m/s]
    small_vy: float = 0.10  # Walking kinds: lane-keeping v_y within +-this [m/s]
    straight_wz: float = 0.3  # slow: |w_z| below this [rad/s]
    turn_wz: tuple[float, float] = (0.3, 1.1)  # slow_turn: |w_z| [rad/s] (pilot.py max_turn 1.1)
    fast_wz: float = 0.8  # fast: |w_z| below this [rad/s]
    rotate_wz: tuple[float, float] = (0.4, 1.0)  # rotate (on the spot): |w_z| [rad/s]
    sideways_vx: tuple[float, float] = (-0.15, 0.05)  # sideways: v_x [m/s] (the slalom holds slightly backward)
    sideways_vy: tuple[float, float] = (0.25, 0.5)  # sideways: |v_y| [m/s]
    sideways_wz: float = 0.1  # sideways: |w_z| below this [rad/s]
