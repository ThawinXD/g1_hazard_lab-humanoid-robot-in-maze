"""Trench crossing test environment (``scripts/trench_test.py``): walled corridors with one trench each."""

import gymnasium as gym

gym.register(
    id="G1-Trench-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1TrenchTestEnvCfg",
        "default_agent": "rsl_rl",
        # Same G1 Rough network as the challenge's maze tasks
        "rsl_rl_cfg_entry_point": "g1_maze_lab.tasks.maze.agents.rsl_rl_ppo_cfg:G1MazePPORunnerCfg",
    },
)
