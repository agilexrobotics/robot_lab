# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Open-Drawer-Piper",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.PiperCabinetEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Open-Drawer-Piper-Play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.PiperCabinetEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
