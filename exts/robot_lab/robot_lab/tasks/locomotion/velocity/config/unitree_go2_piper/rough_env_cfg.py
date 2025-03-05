from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from robot_lab.tasks.locomotion.velocity.mdp.rewards import position_command_error, position_command_error_tanh, orientation_command_error
from omni.isaac.lab.utils import configclass
import robot_lab.tasks.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
import math
##
# Pre-defined configs
##
# use cloud assets
# from omni.isaac.lab_assets.unitree import UNITREE_GO2_PIPER_CFG  # isort: skip
# use local assets
from robot_lab.assets.unitree import UNITREE_GO2_PIPER_CFG  # isort: skip


@configclass
class UnitreeGo2PiperRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    _run_disable_zero_weight_rewards = True

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # switch robot to unitree-go2-piper
        self.scene.robot = UNITREE_GO2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.AMP = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100, 100)}

        # ------------------------------Events------------------------------
        self.events.reset_amp = None
        self.events.reset_base_amp = None
        self.events.reset_robot_joints_amp = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = ["trunk"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["trunk"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_parameters = None

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = ["trunk"]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = ["trunk"]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = -0.0002
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_l1", 0, [""])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0

        # Action penalties
        self.rewards.applied_torque_limits.weight = 0
        self.rewards.applied_torque_limits.params["asset_cfg"].body_names = ["trunk"]
        self.rewards.action_rate_l2.weight = -0.01
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_thigh"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [".*_foot"]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # Others
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.foot_contact.weight = 0
        self.rewards.foot_contact.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.base_height_rough_l2.weight = 0
        self.rewards.base_height_rough_l2.params["target_height"] = 0.35
        self.rewards.base_height_rough_l2.params["asset_cfg"].body_names = ["trunk"]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [".*_foot"]
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still_when_zero_command.weight = -0.5

        # If the weight of rewards is 0, set rewards to None
        if self._run_disable_zero_weight_rewards:
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["trunk"]

        # ------------------------------Commands------------------------------

        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*L_hip_joint", ".*R_hip_joint", "F[L,R]_thigh_joint",
                                             "R[L,R]_thigh_joint", ".*_calf_joint"],
            scale=0.5, use_default_offset=True, clip=None
        )
        self.actions.arm_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["arm_joint0[0-5]"],
            scale=0.5, use_default_offset=True, clip=None
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["arm_joint06", "arm_joint07"],
            open_command_expr={"arm_joint06": 0.04, "arm_joint07": -0.04},
            close_command_expr={"arm_joint06": 0.0, "arm_joint07": 0.0},
        )

        # self.scene.arm_base = UNITREE_GO2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/arm_link00")

        # self.commands.ee_pose = mdp.UniformPoseCommandCfg(
        #     asset_name="robot",
        #     body_name="link_ee",
        #     resampling_time_range=(4.0, 4.0),
        #     debug_vis=True,
        #     ranges=mdp.UniformPoseCommandCfg.Ranges(
        #         pos_x=(0.2, 0.5),
        #         pos_y=(-0.5, 0.5),
        #         pos_z=(0.2, 0.5),
        #         roll=(-math.pi, math.pi), #(-math.pi, math.pi),
        #         pitch=(-math.pi/2, math.pi/2),  # depends on end-effector axis
        #         yaw=(-math.pi/2, math.pi/2),
        #     ),
        # )
        # self.curriculum.action_rate = CurrTerm(
        #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 1000}
        # )
        # self.curriculum.joint_vel = CurrTerm(
        #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 1000}
        # )
        # self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
        # self.rewards.joint_vel = RewTerm(
        #     func=mdp.joint_vel_l2,
        #     weight=-1e-4,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )
        # self.rewards.end_effector_position_tracking = RewTerm(
        #     func=mdp.position_command_error,
        #     weight=-0.2,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names="link_ee"), "command_name": "ee_pose"},
        # )
        # self.rewards.end_effector_position_tracking_fine_grained = RewTerm(
        #     func=mdp.position_command_error_tanh,
        #     weight=0.1,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names="link_ee"), "std": 0.1, "command_name": "ee_pose"},
        # )
        # self.rewards.end_effector_orientation_tracking = RewTerm(
        #     func=mdp.orientation_command_error,
        #     weight=-0.1,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names="link_ee"), "command_name": "ee_pose"},
        # )
        # self.observations.policy.pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
