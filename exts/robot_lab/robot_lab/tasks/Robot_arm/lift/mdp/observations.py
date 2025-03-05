# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.utils.math import transform_points, unproject_depth
import open3d as o3d
import cv2
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


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

def camera_rgb_data(
        env: ManagerBasedRLEnv,
        camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    B = env.scene[camera_cfg.name].data.output["rgb"].shape[0]
    # cv2.imwrite("tmp.png", env.scene[camera_cfg.name].data.output["rgb"][0, :, :, :3].clone().detach().cpu().numpy())
    return env.scene[camera_cfg.name].data.output["rgb"][:, :, :, :3].reshape((B, -1))

def camera_depth_data(
        env: ManagerBasedRLEnv,
        camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    B = env.scene[camera_cfg.name].data.output["distance_to_image_plane"].shape[0]
    return env.scene[camera_cfg.name].data.output["distance_to_image_plane"].reshape((B, -1))

def camera_pc_data(
        env: ManagerBasedRLEnv,
        camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    points_3d_cam = unproject_depth(
        env.scene[camera_cfg.name].data.output["distance_to_image_plane"], env.scene[camera_cfg.name].data.intrinsic_matrices
    )
    # points_3d_world = transform_points(points_3d_cam, camera.data.pos_w, camera.data.quat_w_ros)
    # pc_markers.visualize(translations=points_3d_cam)
    B = points_3d_cam.shape[0]
    points_3d_cam = torch.concatenate([points_3d_cam, env.scene[camera_cfg.name].data.output["rgb"][:, :, :, :3].permute(0, 2, 1, 3).reshape((B, -1, 3))], 2)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_3d_cam[0][:, :3].clone().detach().cpu().numpy())  # 点云数据
    # pcd.colors = o3d.utility.Vector3dVector(points_3d_cam[0][:, 3:].clone().detach().cpu().numpy()/255)
    # o3d.io.write_point_cloud("tmp.pcd", pcd)
    # points_3d_cam = points_3d_cam.reshape((B, -1))
    return points_3d_cam.reshape((B, -1))
