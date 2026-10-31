"""terrain_levels_skill: robots that walk well on turning commands must move up (no simulator, mocked env)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from g1_maze_lab.tasks.curriculum.mdp.curriculums import terrain_levels_skill

EPISODE_S = 20.0
WEIGHT = 1.0


class _Terrain:
    max_terrain_level = 10

    def __init__(self, levels):
        self.terrain_levels = torch.tensor(levels)

    def update_env_origins(self, env_ids, move_up, move_down):
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0, self.max_terrain_level - 1)


class _Scene:
    def __init__(self, end_xy, levels):
        n = len(end_xy)
        pos = torch.cat((torch.tensor(end_xy, dtype=torch.float32), torch.zeros(n, 1)), dim=1)
        self._robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=pos)))
        self.terrain = _Terrain(levels)
        self.env_origins = torch.zeros(n, 3)

    def __getitem__(self, name):
        return self._robot


def _run(track, fell, timed_out, end_xy, levels):
    """One robot per entry: mean tracking fraction, fell?, timed out?, final XY relative to its spawn."""
    terms = {"base_contact": torch.tensor(fell), "physics_blowup": torch.zeros(len(track), dtype=torch.bool)}
    env = SimpleNamespace(
        common_step_counter=24 * 100,
        max_episode_length_s=EPISODE_S,
        scene=_Scene(end_xy, levels),
        reward_manager=SimpleNamespace(
            _episode_sums={"track_lin_vel_xy_exp": torch.tensor(track) * WEIGHT * EPISODE_S},
            get_term_cfg=lambda name: SimpleNamespace(weight=WEIGHT),
        ),
        termination_manager=SimpleNamespace(
            active_terms=list(terms), get_term=lambda name: terms[name], time_outs=torch.tensor(timed_out)
        ),
    )
    terrain_levels_skill(env, torch.arange(len(track)), fall_terms=("base_contact", "physics_blowup"))
    return env.scene.terrain.terrain_levels.tolist()


def test_circling_robot_that_walks_well_moves_up():
    """Ends only 2.5 m from spawn (walked a circle) but tracked well and never fell: the old rule demoted it."""
    assert _run([0.85], [False], [True], [[2.5, 0.0]], [3]) == [4]


def test_robot_that_falls_moves_down():
    assert _run([0.85], [True], [False], [[3.0, 0.0]], [3]) == [2]


def test_robot_stuck_at_a_ramp_moves_down():
    """Tracked badly (stood at the foot of a ramp) without falling."""
    assert _run([0.3], [False], [True], [[1.2, 0.0]], [3]) == [2]


def test_robot_that_never_left_the_platform_stays():
    """Good tracking of a slow / tight-turn command, but it never reached the hazard ring: no promotion, no demotion."""
    assert _run([0.9], [False], [True], [[0.8, 0.5]], [3]) == [3]
