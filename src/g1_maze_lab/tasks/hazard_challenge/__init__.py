"""Challenge-like training task: G1-Hazard-Challenge (real ice, pilot-like commands, harder hazards)."""

import gymnasium as gym

gym.register(
    id="G1-Hazard-Challenge",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1HazardChallengeEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:G1HazardChallengePPORunnerCfg",
    },
)
