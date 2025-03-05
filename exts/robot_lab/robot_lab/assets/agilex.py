"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`UNITREE_A1_CFG`: Unitree A1 robot with DC motor model for the legs
* :obj:`G1_CFG`: G1 humanoid robot

Reference: https://github.com/unitreerobotics/unitree_ros
"""

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
import os
##
# Configuration
##

AGILEX_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Agilex/Piper/piper-1.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            fix_root_link=True
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 1.57,
            "joint3": -1.57,
            "joint4": 0.0,
            "joint5": 1.2,
            "joint6": 0.0,
            "joint7": 0.04,
            "joint8": -0.04,
        },
    ),
    actuators={
        "piper_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "piper_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["joint7", "joint8"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        # "ee": ImplicitActuatorCfg(
        #     joint_names_expr=["joint_ee"]
        # )

    },
    soft_joint_pos_limit_factor=1.0,
    # spawn=sim_utils.UsdFileCfg(
    #     usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Agilex/Piper/Piper.usd",
    #     activate_contact_sensors=True,
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=False,
    #         retain_accelerations=False,
    #         linear_damping=0.0,
    #         angular_damping=0.0,
    #         max_linear_velocity=1000.0,
    #         max_angular_velocity=1000.0,
    #         max_depenetration_velocity=1.0,
    #     ),
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #     ),
    # ),
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.42),
    #     joint_pos={
    #         ".*L_hip_joint": 0.1,
    #         ".*R_hip_joint": -0.1,
    #         "F[L,R]_thigh_joint": 0.8,
    #         "R[L,R]_thigh_joint": 1.0,
    #         ".*_calf_joint": -1.5,
    #     },
    #     joint_vel={".*": 0.0},
    # ),
    # soft_joint_pos_limit_factor=0.9,
    # actuators={
    #     "base_legs": DCMotorCfg(
    #         joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    #         effort_limit=33.5,
    #         saturation_effort=33.5,
    #         velocity_limit=21.0,
    #         stiffness=20.0,
    #         damping=0.5,
    #         friction=0.0,
    #     ),
    # },
)
"""Configuration of Unitree A1 using DC motor.

Note: Specifications taken from: https://www.trossenrobotics.com/a1-quadruped#specifications
"""
