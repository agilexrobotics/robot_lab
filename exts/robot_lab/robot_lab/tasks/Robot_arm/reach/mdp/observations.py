# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import FrameTransformer

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from omni.isaac.lab.utils.math import transform_points, unproject_depth


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    
    return object_pos_b


def last_action_clamped(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions.clamp(-1,1)


def camera_rgb_data(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return env.scene[object_cfg.name].data.output["rgb"][:, :, :3]

def camera_depth_data(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return env.scene[object_cfg.name].data.output["distance_to_image_plane"]

def camera_pc_data(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    points_3d_cam = unproject_depth(
        env.scene[object_cfg.name].data.output["distance_to_image_plane"], env.scene[object_cfg.name].data.intrinsic_matrices
    )
    # points_3d_world = transform_points(points_3d_cam, camera.data.pos_w, camera.data.quat_w_ros)
    # pc_markers.visualize(translations=points_3d_cam)
    print(points_3d_cam.shape)
    points_3d_cam = points_3d_cam.flatten()
    points_3d_cam = points_3d_cam.unsqueeze(0)
    print(points_3d_cam.shape)
    return points_3d_cam

# def camera_pcrgb(
#         env: ManagerBasedRLEnv,
#         object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """The position of the object in the robot's root frame."""
#     pointcloud = create_pointcloud_from_depth(
#         intrinsic_matrix=env.scene[object_cfg.name].data.intrinsic_matrices[camera_index],
#         depth=env.scene[object_cfg.name].data.output[camera_index]["depth"],
#         # position=env.scene[object_cfg.name].data.pos_w[camera_index],
#         # orientation=env.scene[object_cfg.name].data.quat_w_ros[camera_index],
#         device=sim.device,
#     )
#     return np.concatenate([env.scene[object_cfg.name].data.output["rgb"][:, :, :3], ])