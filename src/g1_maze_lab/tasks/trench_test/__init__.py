"""Trench / ramp / ice test environments (``scripts/trench_test.py``, ``scripts/ice_turn_test.py``): walled
corridors with one hazard each."""

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

gym.register(
    id="G1-Ramp-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1RampTestEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": "g1_maze_lab.tasks.maze.agents.rsl_rl_ppo_cfg:G1MazePPORunnerCfg",
    },
)

gym.register(
    id="G1-Ice-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1IceTestEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": "g1_maze_lab.tasks.maze.agents.rsl_rl_ppo_cfg:G1MazePPORunnerCfg",
    },
)

gym.register(
    id="G1-Ice-Corner-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1IceCornerTestEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": "g1_maze_lab.tasks.maze.agents.rsl_rl_ppo_cfg:G1MazePPORunnerCfg",
    },
)
