"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

import robot_lab.tasks  # noqa: F401

from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    print("agent_cfg:", agent_cfg)
    print("args_cli:", args_cli)
    # agent_cfg: UnitreeH1FlatPPORunnerCfg(seed=42, device='cuda:0', num_steps_per_env=24, max_iterations=1000,
    #                                      empirical_normalization=False,
    #                                      policy=RslRlPpoActorCriticCfg(class_name='ActorCritic', init_noise_std=1.0,
    #                                                                    actor_hidden_dims=[128, 128, 128],
    #                                                                    critic_hidden_dims=[128, 128, 128],
    #                                                                    activation='elu'),
    #                                      algorithm=RslRlPpoAlgorithmCfg(class_name='PPO', value_loss_coef=1.0,
    #                                                                     use_clipped_value_loss=True, clip_param=0.2,
    #                                                                     entropy_coef=0.01, num_learning_epochs=5,
    #                                                                     num_mini_batches=4, learning_rate=0.001,
    #                                                                     schedule='adaptive', gamma=0.99, lam=0.95,
    #                                                                     desired_kl=0.01, max_grad_norm=1.0),
    #                                      save_interval=100, experiment_name='h1_flat', run_name='',
    #                                      logger='tensorboard', neptune_project='isaaclab', wandb_project='isaaclab',
    #                                      resume=False, load_run='.*', load_checkpoint='model_.*.pt')
    # args_cli: Namespace(video=False, video_length=200, video_interval=2000, num_envs=32,
    #                     task='RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0', seed=None, max_iterations=None,
    #                     experiment_name=None, run_name=None, resume=None, load_run=None, checkpoint=None, logger=None,
    #                     log_project_name=None, device='cuda:0', cpu=False, verbose=False, headless=False, hide_ui=False,
    #                     physics_gpu=0, active_gpu=0)
    # agent_cfg.max_iterations: 1000

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    print("agent_cfg.max_iterations:", agent_cfg.max_iterations)
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print("env_cfg:", env_cfg)
#     env_cfg: UnitreeH1FlatEnvCfg(
#         viewer=ViewerCfg(eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0), cam_prim_path='/OmniverseKit_Persp',
#                          resolution=(1280, 720), origin_type='world', env_index=0, asset_name=None),
#         sim=SimulationCfg(physics_prim_path='/physicsScene', device='cuda:0', dt=0.005, render_interval=4,
#                           gravity=(0.0, 0.0, -9.81), enable_scene_query_support=False, use_fabric=True,
#                           disable_contact_processing=True, physx=PhysxCfg(solver_type=1, min_position_iteration_count=1,
#                                                                           max_position_iteration_count=255,
#                                                                           min_velocity_iteration_count=0,
#                                                                           max_velocity_iteration_count=255,
#                                                                           enable_ccd=False, enable_stabilization=True,
#                                                                           enable_enhanced_determinism=False,
#                                                                           bounce_threshold_velocity=0.5,
#                                                                           friction_offset_threshold=0.04,
#                                                                           friction_correlation_distance=0.025,
#                                                                           gpu_max_rigid_contact_count=8388608,
#                                                                           gpu_max_rigid_patch_count=163840,
#                                                                           gpu_found_lost_pairs_capacity=2097152,
#                                                                           gpu_found_lost_aggregate_pairs_capacity=33554432,
#                                                                           gpu_total_aggregate_pairs_capacity=2097152,
#                                                                           gpu_collision_stack_size=67108864,
#                                                                           gpu_heap_capacity=67108864,
#                                                                           gpu_temp_buffer_capacity=16777216,
#                                                                           gpu_max_num_partitions=8,
#                                                                           gpu_max_soft_body_contacts=1048576,
#                                                                           gpu_max_particle_contacts=1048576),
#                           physics_material=RigidBodyMaterialCfg(func= < function
#     spawn_rigid_body_material
#     at
#     0x7f5e8af193f0 >, static_friction = 1.0, dynamic_friction = 1.0, restitution = 0.0, improve_patch_friction = True, friction_combine_mode = 'multiply', restitution_combine_mode = 'multiply', compliant_contact_stiffness = 0.0, compliant_contact_damping = 0.0)), ui_window_class_type = <
#
#     class 'omni.isaac.lab.envs.ui.manager_based_rl_env_window.ManagerBasedRLEnvWindow'>, seed=None, decimation=4, scene=MySceneCfg(num_envs=32, env_spacing=2.5, lazy_sensor_update=True, replicate_physics=True, robot=ArticulationCfg(class_type= <
#
#
# class 'omni.isaac.lab.assets.articulation.articulation.Articulation'>, prim_path='{ENV_REGEX_NS}/Robot', spawn=UsdFileCfg(func= < function spawn_from_usd at 0x7f5e8af18e50 >, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=None, kinematic_enabled=None, disable_gravity=False, linear_damping=0.0, angular_damping=0.0, max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0, max_contact_impulse=None, enable_gyroscopic_forces=None, retain_accelerations=False, solver_position_iteration_count=None, solver_velocity_iteration_count=None, sleep_threshold=None, stabilization_threshold=None), collision_props=None, activate_contact_sensors=True, scale=None, articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=None, enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4, sleep_threshold=None, stabilization_threshold=None, fix_root_link=None), fixed_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=None, usd_path='http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/IsaacLab/Robots/Unitree/H1/h1_minimal.usd', variants=None), init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.05), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0), joint_pos={'.*_hip_yaw': 0.0, '.*_hip_roll': 0.0
#
# , '.*_hip_pitch': -0.28, '.*_knee': 0.79, '.*_ankle': -0.52, 'torso': 0.0, '.*_shoulder_pitch': 0.28, '.*_shoulder_roll': 0.0, '.*_shoulder_yaw': 0.0, '.*_elbow': 0.52}, joint_vel = {
#     '.*': 0.0}), collision_group = 0, debug_vis = False, soft_joint_pos_limit_factor = 0.9, actuators = {
#     'legs': ImplicitActuatorCfg(class_type= <
#
#
# class 'omni.isaac.lab.actuators.actuator_pd.ImplicitActuator'>, joint_names_expr=['.*_hip_yaw', '.*_hip_roll', '.*_hip_pitch', '.*_knee', 'torso'], effort_limit=300, velocity_limit=100.0, stiffness={'.*_hip_yaw': 150.0, '.*_hip_roll': 150.0
#
# , '.*_hip_pitch': 200.0, '.*_knee': 200.0, 'torso': 200.0}, damping = {'.*_hip_yaw': 5.0, '.*_hip_roll': 5.0,
#                                                                        '.*_hip_pitch': 5.0, '.*_knee': 5.0,
#                                                                        'torso': 5.0}, armature = None, friction = None), 'feet': ImplicitActuatorCfg(
#     class_type= <
#
#
# class 'omni.isaac.lab.actuators.actuator_pd.ImplicitActuator'>, joint_names_expr=['.*_ankle'], effort_limit=100, velocity_limit=100.0, stiffness={'.*_ankle': 20.0
#
# }, damping = {'.*_ankle': 4.0}, armature = None, friction = None), 'arms': ImplicitActuatorCfg(class_type= <
#
#
# class 'omni.isaac.lab.actuators.actuator_pd.ImplicitActuator'>, joint_names_expr=['.*_shoulder_pitch', '.*_shoulder_roll', '.*_shoulder_yaw', '.*_elbow'], effort_limit=300, velocity_limit=100.0, stiffness={'.*_shoulder_pitch': 40.0, '.*_shoulder_roll': 40.0
#
# , '.*_shoulder_yaw': 40.0, '.*_elbow': 40.0}, damping = {'.*_shoulder_pitch': 10.0, '.*_shoulder_roll': 10.0,
#                                                          '.*_shoulder_yaw': 10.0,
#                                                          '.*_elbow': 10.0}, armature = None, friction = None)}), terrain = TerrainImporterCfg(
#     class_type= <
#
#
# class 'omni.isaac.lab.terrains.terrain_importer.TerrainImporter'>, collision_group=-1, prim_path='/World/ground', num_envs= < dataclasses._MISSING_TYPE object at 0x7f5e82bbc610 >, terrain_type='plane', terrain_generator=None, usd_path=None, env_spacing=None, visual_material=MdlFileCfg(func= < function spawn_from_mdl_file at 0x7f5e8af1a9e0 >, mdl_path='{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl', project_uvw=True, albedo_brightness=None, texture_scale=None), physics_material=RigidBodyMaterialCfg(func= < function spawn_rigid_body_material at 0x7f5e8af193f0 >, static_friction=1.0, dynamic_friction=1.0, restitution=0.0, improve_patch_friction=True, friction_combine_mode='multiply', restitution_combine_mode='multiply', compliant_contact_stiffness=0.0, compliant_contact_damping=0.0), max_init_terrain_level=5, debug_vis=False), height_scanner=None, contact_forces=ContactSensorCfg(class_type= < class 'omni.isaac.lab.sensors.contact_sensor.contact_sensor.ContactSensor' >, prim_path='{ENV_REGEX_NS}/Robot/.*', update_period=0.005, history_length=3, debug_vis=False, track_pose=False, track_air_time=True, force_threshold=1.0, filter_prim_paths_expr=[], visualizer_cfg=VisualizationMarkersCfg(prim_path='/Visuals/ContactSensor', markers={'contact': SphereCfg(
#     func= < function
#
#
# spawn_sphere
# at
# 0x7f5e8838ca60 >, visible = True, semantic_tags = None, copy_from_source = True, mass_props = None, rigid_props = None, collision_props = None, activate_contact_sensors = False, visual_material_path = 'material', visual_material = PreviewSurfaceCfg(
#     func= < function
# spawn_preview_surface
# at
# 0x7f5e8af1a830 >, diffuse_color = (1.0, 0.0, 0.0), emissive_color = (0.0, 0.0,
#                                                                      0.0), roughness = 0.5, metallic = 0.0, opacity = 1.0), physics_material_path = 'material', physics_material = None, radius = 0.02), 'no_contact': SphereCfg(
#     func= < function
# spawn_sphere
# at
# 0x7f5e8838ca60 >, visible = False, semantic_tags = None, copy_from_source = True, mass_props = None, rigid_props = None, collision_props = None, activate_contact_sensors = False, visual_material_path = 'material', visual_material = PreviewSurfaceCfg(
#     func= < function
# spawn_preview_surface
# at
# 0x7f5e8af1a830 >, diffuse_color = (0.0, 1.0, 0.0), emissive_color = (0.0, 0.0,
#                                                                      0.0), roughness = 0.5, metallic = 0.0, opacity = 1.0), physics_material_path = 'material', physics_material = None, radius = 0.02)})), light = AssetBaseCfg(
#     class_type= < dataclasses._MISSING_TYPE
# object
# at
# 0x7f5e82bbe230 >, prim_path = '/World/light', spawn = DistantLightCfg(func= < function
# spawn_light
# at
# 0x7f5e8af3a3b0 >, visible = True, semantic_tags = None, copy_from_source = True, prim_type = 'DistantLight', color = (
# 0.75, 0.75,
# 0.75), enable_color_temperature = False, color_temperature = 6500.0, normalize = False, exposure = 0.0, intensity = 3000.0, angle = 0.53), init_state = AssetBaseCfg.InitialStateCfg(
#     pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)), collision_group = 0, debug_vis = False), sky_light = AssetBaseCfg(
#     class_type= < dataclasses._MISSING_TYPE
# object
# at
# 0x7f5e82bbe440 >, prim_path = '/World/skyLight', spawn = DomeLightCfg(func= < function
# spawn_light
# at
# 0x7f5e8af3a3b0 >, visible = True, semantic_tags = None, copy_from_source = True, prim_type = 'DomeLight', color = (
# 0.13, 0.13,
# 0.13), enable_color_temperature = False, color_temperature = 6500.0, normalize = False, exposure = 0.0, intensity = 1000.0, texture_file = None, texture_format = 'automatic', visible_in_primary_ray = True), init_state = AssetBaseCfg.InitialStateCfg(
#     pos=(0.0, 0.0, 0.0),
#     rot=(1.0, 0.0, 0.0, 0.0)), collision_group = 0, debug_vis = False)), observations = ObservationsCfg(
#     policy=ObservationsCfg.PolicyCfg(concatenate_terms=True, enable_corruption=True,
#                                      base_lin_vel=ObservationTermCfg(func= < function
# base_lin_vel
# at
# 0x7f5e85fe8430 >, params = {}, modifiers = None, noise = UniformNoiseCfg(func= < function
# uniform_noise
# at
# 0x7f5e883ff0a0 >, operation = 'add', n_min = -0.1, n_max = 0.1), clip = (
# -100.0, 100.0), scale = 1.0), base_ang_vel = ObservationTermCfg(func= < function
# base_ang_vel
# at
# 0x7f5e85fe84c0 >, params = {}, modifiers = None, noise = UniformNoiseCfg(func= < function
# uniform_noise
# at
# 0x7f5e883ff0a0 >, operation = 'add', n_min = -0.2, n_max = 0.2), clip = (
# -100.0, 100.0), scale = 1.0), projected_gravity = ObservationTermCfg(func= < function
# projected_gravity
# at
# 0x7f5e85fe8550 >, params = {}, modifiers = None, noise = UniformNoiseCfg(func= < function
# uniform_noise
# at
# 0x7f5e883ff0a0 >, operation = 'add', n_min = -0.05, n_max = 0.05), clip = (
# -100.0, 100.0), scale = 1.0), velocity_commands = ObservationTermCfg(func= < function
# generated_commands
# at
# 0x7f5e85fe8ca0 >, params = {'command_name': 'base_velocity'}, modifiers = None, noise = None, clip = (
# -100.0, 100.0), scale = 1.0), joint_pos = ObservationTermCfg(func= < function
# joint_pos_rel
# at
# 0x7f5e85fe88b0 >, params = {}, modifiers = None, noise = UniformNoiseCfg(func= < function
# uniform_noise
# at
# 0x7f5e883ff0a0 >, operation = 'add', n_min = -0.01, n_max = 0.01), clip = (
# -100.0, 100.0), scale = 1.0), joint_vel = ObservationTermCfg(func= < function
# joint_vel_rel
# at
# 0x7f5e85fe8a60 >, params = {}, modifiers = None, noise = UniformNoiseCfg(func= < function
# uniform_noise
# at
# 0x7f5e883ff0a0 >, operation = 'add', n_min = -1.5, n_max = 1.5), clip = (
# -100.0, 100.0), scale = 1.0), actions = ObservationTermCfg(func= < function
# last_action
# at
# 0x7f5e85fe8c10 >, params = {}, modifiers = None, noise = None, clip = (
# -100.0, 100.0), scale = 1.0), height_scan = None), AMP = None), actions = ActionsCfg(
#     joint_pos=JointPositionActionCfg(class_type= <
#
#
# class 'robot_lab.tasks.locomotion.velocity.mdp.actions.joint_actions.JointPositionAction'>, asset_name='robot', debug_vis=False, joint_names=['.*'], scale=0.5, offset=0.0, preserve_order=False, clip=None, use_default_offset=True)), events=EventCfg(physics_material=EventTermCfg(func= < function randomize_rigid_body_material at 0x7f5e85fd3910 >, params={'asset_cfg': SceneEntityCfg(
#     name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None,
#     fixed_tendon_ids=slice(None, None, None), body_names='.*', body_ids=slice(None, None, None),
#     preserve_order=False), 'static_friction_range': (0.8, 0.8)
#
# , 'dynamic_friction_range': (0.6, 0.6), 'restitution_range': (0.0,
#                                                               0.0), 'num_buckets': 64}, mode = 'startup', interval_range_s = None, is_global_time = False, min_step_count_between_reset = 0), add_base_mass = None, base_external_force_torque = EventTermCfg(
#     func= < function
# apply_external_force_torque
# at
# 0x7f5e85fd3c70 >, params = {
#     'asset_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None),
#                                 fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None),
#                                 body_names=['.*torso_link'], body_ids=slice(None, None, None), preserve_order=False),
#     'force_range': (5.0, 5.0), 'torque_range': (-5.0,
#                                                 5.0)}, mode = 'reset', interval_range_s = None, is_global_time = False, min_step_count_between_reset = 0), reset_base = EventTermCfg(
#     func= < function
# reset_root_state_uniform
# at
# 0x7f5e85fd3d90 >, params = {'pose_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'yaw': (-3.14, 3.14)},
#                             'velocity_range': {'x': (0.0, 0.0), 'y': (0.0, 0.0), 'z': (0.0, 0.0), 'roll': (0.0, 0.0),
#                                                'pitch': (0.0, 0.0), 'yaw': (0.0,
#                                                                             0.0)}}, mode = 'reset', interval_range_s = None, is_global_time = False, min_step_count_between_reset = 0), reset_robot_joints = EventTermCfg(
#     func= < function
# reset_joints_by_scale
# at
# 0x7f5e85fd3f40 >, params = {'position_range': (1.0, 1.0), 'velocity_range': (0.0,
#                                                                              0.0)}, mode = 'reset', interval_range_s = None, is_global_time = False, min_step_count_between_reset = 0), reset_base_amp = None, reset_robot_joints_amp = None, randomize_actuator_gains = None, randomize_joint_parameters = None, push_robot = None), is_finite_horizon = False, episode_length_s = 20.0, rewards = RewardsCfg(
#     is_terminated=RewardTermCfg(func= < function
# is_terminated
# at
# 0x7f5e85fe9090 >, params = {}, weight = -200), lin_vel_z_l2 = None, ang_vel_xy_l2 = RewardTermCfg(func= < function
# ang_vel_xy_l2
# at
# 0x7f5e85fe92d0 >, params = {}, weight = -0.05), flat_orientation_l2 = RewardTermCfg(func= < function
# flat_orientation_l2
# at
# 0x7f5e85fe9360 >, params = {}, weight = -1.0), base_height_l2 = None, body_lin_acc_l2 = None, joint_torques_l2 = None, joint_vel_l2 = None, joint_acc_l2 = RewardTermCfg(
#     func= < function
# joint_acc_l2
# at
# 0x7f5e85fe96c0 >, params = {}, weight = -1.25e-07), joint_pos_limits = RewardTermCfg(func= < function
# joint_pos_limits
# at
# 0x7f5e85fe97e0 >, params = {
#     'asset_cfg': SceneEntityCfg(name='robot', joint_names=['.*_ankle'], joint_ids=slice(None, None, None),
#                                 fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=None,
#                                 body_ids=slice(None, None, None),
#                                 preserve_order=False)}, weight = -1.0), joint_vel_limits = None, applied_torque_limits = None, action_rate_l2 = RewardTermCfg(
#     func= < function
# action_rate_l2
# at
# 0x7f5e85fe9990 >, params = {}, weight = -0.005), undesired_contacts = None, contact_forces = None, track_lin_vel_xy_exp = RewardTermCfg(
#     func= < function
# track_lin_vel_xy_yaw_frame_exp
# at
# 0x7f5e84380550 >, params = {'command_name': 'base_velocity',
#                             'std': 0.5}, weight = 1.0), track_ang_vel_z_exp = RewardTermCfg(func= < function
# track_ang_vel_z_world_exp
# at
# 0x7f5e843805e0 >, params = {'command_name': 'base_velocity', 'std': 0.5}, weight = 1.0), feet_air_time = RewardTermCfg(
#     func= < function
# feet_air_time_positive_biped
# at
# 0x7f5e84380430 >, params = {'command_name': 'base_velocity',
#                             'sensor_cfg': SceneEntityCfg(name='contact_forces', joint_names=None,
#                                                          joint_ids=slice(None, None, None), fixed_tendon_names=None,
#                                                          fixed_tendon_ids=slice(None, None, None),
#                                                          body_names=['.*ankle_link'], body_ids=slice(None, None, None),
#                                                          preserve_order=False),
#                             'threshold': 0.6}, weight = 1.0), foot_contact = None, base_height_rough_l2 = None, feet_slide = RewardTermCfg(
#     func= < function
# feet_slide
# at
# 0x7f5e843804c0 >, params = {
#     'sensor_cfg': SceneEntityCfg(name='contact_forces', joint_names=None, joint_ids=slice(None, None, None),
#                                  fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None),
#                                  body_names=['.*ankle_link'], body_ids=slice(None, None, None), preserve_order=False),
#     'asset_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None),
#                                 fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None),
#                                 body_names=['.*ankle_link'], body_ids=slice(None, None, None),
#                                 preserve_order=False)}, weight = -0.25), joint_power = None, stand_still_when_zero_command = None), terminations = TerminationsCfg(
#     time_out=TerminationTermCfg(func= < function
# time_out
# at
# 0x7f5e85fe9fc0 >, params = {}, time_out = True), illegal_contact = TerminationTermCfg(func= < function
# illegal_contact
# at
# 0x7f5e85fea4d0 >, params = {
#     'sensor_cfg': SceneEntityCfg(name='contact_forces', joint_names=None, joint_ids=slice(None, None, None),
#                                  fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None),
#                                  body_names=['.*torso_link'], body_ids=slice(None, None, None), preserve_order=False),
#     'threshold': 1.0}, time_out = False)), curriculum = CurriculumCfg(terrain_levels=None), commands = CommandsCfg(
#     base_velocity=UniformVelocityCommandCfg(class_type= <
#
#
# class 'omni.isaac.lab.envs.mdp.commands.velocity_command.UniformVelocityCommand'>, resampling_time_range=(10.0, 10.0), debug_vis=True, asset_name='robot', heading_command=True, heading_control_stiffness=0.5, rel_standing_envs=0.02, rel_heading_envs=1.0, ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(-3.141592653589793, 3.141592653589793)))), _run_disable_zero_weight_rewards=True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
