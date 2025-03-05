# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import IPython
e = IPython.embed

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from torch import Tensor

# sys.path.append("/home/agilex/aloha_ws/src/aloha-devel/aloha/pointnet2")
# from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


# class Pointnet2Backbone(nn.Module):
#     r"""
#        Backbone network for point cloud feature learning.
#        Based on Pointnet++ single-scale grouping network.
#
#        Parameters
#        ----------
#        input_feature_dim: int
#             Number of input channels in the feature descriptor for each point.
#             e.g. 3 for RGB.
#     """
#
#     def __init__(self, output_dim, input_feature_dim=0,
#                  qpos_vector_dim=0,
#                  instruction_vector_dim=0):
#         super().__init__()
#         self.instruction_vector_dim = instruction_vector_dim
#         self.qpos_vector_dim = qpos_vector_dim
#         self.sa1 = PointnetSAModuleVotes(
#             npoint=2048,
#             radius=0.04,
#             nsample=64,
#             mlp=[input_feature_dim, 64, 64, 128],
#             use_xyz=True,
#             normalize_xyz=True
#         )
#
#         self.sa2 = PointnetSAModuleVotes(
#             npoint=1024,
#             radius=0.1,
#             nsample=32,
#             mlp=[128, 128, 128, 256],
#             use_xyz=True,
#             normalize_xyz=True
#         )
#
#         self.sa3 = PointnetSAModuleVotes(
#             npoint=512,
#             radius=0.2,
#             nsample=16,
#             mlp=[256, 128, 128, 256],
#             use_xyz=True,
#             normalize_xyz=True
#         )
#
#         self.sa4 = PointnetSAModuleVotes(
#             npoint=256,
#             radius=0.3,
#             nsample=16,
#             mlp=[256, 128, 128, 256],
#             use_xyz=True,
#             normalize_xyz=True
#         )
#
#         if instruction_vector_dim:
#             self.instruction_proj1 = nn.Linear(instruction_vector_dim, 128 * 2)
#             self.instruction_proj2 = nn.Linear(instruction_vector_dim, 256 * 2)
#             self.instruction_proj3 = nn.Linear(instruction_vector_dim, 256 * 2)
#             self.instruction_proj4 = nn.Linear(instruction_vector_dim, 256 * 2)
#         if qpos_vector_dim:
#             self.qpos_proj1 = nn.Linear(qpos_vector_dim, 128 * 2)
#             self.qpos_proj2 = nn.Linear(qpos_vector_dim, 256 * 2)
#             self.qpos_proj3 = nn.Linear(qpos_vector_dim, 256 * 2)
#             self.qpos_proj4 = nn.Linear(qpos_vector_dim, 256 * 2)
#
#         self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
#         self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])
#         self.output_proj = nn.Linear(256, output_dim)
#
#     def _break_up_pc(self, pc):
#         xyz = pc[..., 0:3].contiguous()
#         features = (
#             pc[..., 3:].transpose(1, 2).contiguous()
#             if pc.size(-1) > 3 else None
#         )
#
#         return xyz, features
#
#     def forward(self, pointcloud: torch.cuda.FloatTensor, qpos_vector=None, instruction_vector=None, end_points=None):
#         if not end_points: end_points = {}
#         batch_size = pointcloud.shape[0]
#         xyz, features = self._break_up_pc(pointcloud)
#         end_points['input_xyz'] = xyz
#         end_points['input_features'] = features
#
#         # --------- 4 SET ABSTRACTION LAYERS ---------
#         xyz, features, fps_inds = self.sa1(xyz, features)
#         if self.qpos_vector_dim:
#             B, C, N = features.shape
#             beta, gamma = torch.split(
#                 self.qpos_proj1(qpos_vector).reshape(B, C * 2, 1), [C, C], 1
#             )
#             features = (1 + gamma) * features + beta
#         if self.instruction_vector_dim:
#             B, C, N = features.shape
#             beta, gamma = torch.split(
#                 self.instruction_proj1(instruction_vector).reshape(B, C * 2, 1), [C, C], 1
#             )
#             features = (1 + gamma) * features + beta
#         end_points['sa1_inds'] = fps_inds
#         end_points['sa1_xyz'] = xyz
#         end_points['sa1_features'] = features
#
#         xyz, features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
#         if self.qpos_vector_dim:
#             B, C, N = features.shape
#             beta, gamma = torch.split(
#                 self.qpos_proj2(qpos_vector).reshape(B, C * 2, 1), [C, C], 1
#             )
#             features = (1 + gamma) * features + beta
#         if self.instruction_vector_dim:
#             B, C, N = features.shape
#             beta, gamma = torch.split(
#                 self.instruction_proj2(instruction_vector).reshape(B, C * 2, 1), [C, C], 1
#             )
#             features = (1 + gamma) * features + beta
#         end_points['sa2_inds'] = fps_inds
#         end_points['sa2_xyz'] = xyz
#         end_points['sa2_features'] = features
#
#         xyz, features, fps_inds = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,511
#         if self.qpos_vector_dim:
#             B, C, N = features.shape
#             beta, gamma = torch.split(
#                 self.qpos_proj3(qpos_vector).reshape(B, C * 2, 1), [C, C], 1
#             )
#             features = (1 + gamma) * features + beta
#         if self.instruction_vector_dim:
#             B, C, N = features.shape
#             beta, gamma = torch.split(
#                 self.instruction_proj3(instruction_vector).reshape(B, C * 2, 1), [C, C], 1
#             )
#             features = (1 + gamma) * features + beta
#         end_points['sa3_xyz'] = xyz
#         end_points['sa3_features'] = features
#
#         xyz, features, fps_inds = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
#         if self.qpos_vector_dim:
#             B, C, N = features.shape
#             beta, gamma = torch.split(
#                 self.qpos_proj4(qpos_vector).reshape(B, C * 2, 1), [C, C], 1
#             )
#             features = (1 + gamma) * features + beta
#         if self.instruction_vector_dim:
#             B, C, N = features.shape
#             beta, gamma = torch.split(
#                 self.instruction_proj4(instruction_vector).reshape(B, C * 2, 1), [C, C], 1
#             )
#             features = (1 + gamma) * features + beta
#         end_points['sa4_xyz'] = xyz
#         end_points['sa4_features'] = features
#
#         # --------- 2 FEATURE UPSAMPLING LAYERS --------
#         features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'],
#                             end_points['sa4_features'])
#         features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
#         end_points['fp2_features'] = features
#         end_points['fp2_xyz'] = end_points['sa2_xyz']
#         num_seed = end_points['fp2_xyz'].shape[1]
#         end_points['fp2_inds'] = end_points['sa1_inds'][:, 0:num_seed]  # indices among the entire input point clouds
#         features = self.output_proj(torch.max(features, -1)[0])
#         # print("feature:", features.shape)
#         return features


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool,
                 qpos_vector_dim=0, instruction_vector_dim=0):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        self.num_channels = num_channels
        self.qpos_vector_dim = qpos_vector_dim
        self.instruction_vector_dim = instruction_vector_dim
        if instruction_vector_dim:
            self.instruction_proj1 = nn.Linear(instruction_vector_dim, 64 * 2)
            self.instruction_proj2 = nn.Linear(instruction_vector_dim, 64 * 2)
            self.instruction_proj3 = nn.Linear(instruction_vector_dim, 128 * 2)
            self.instruction_proj4 = nn.Linear(instruction_vector_dim, 256 * 2)
        if qpos_vector_dim:
            self.qpos_proj1 = nn.Linear(qpos_vector_dim, 64 * 2)
            self.qpos_proj2 = nn.Linear(qpos_vector_dim, 64 * 2)
            self.qpos_proj3 = nn.Linear(qpos_vector_dim, 128 * 2)
            self.qpos_proj4 = nn.Linear(qpos_vector_dim, 256 * 2)

    def forward(self, tensor, qpos_vecotr=None, instruction_vector=None):
        # xs = self.body(tensor)
        # return xs
        for i in range(len(self.body)):
            tensor = self.body[i](tensor)
            if self.qpos_vector_dim:
                B, C, H, W = tensor.shape
                if i == 2:
                    beta, gamma = torch.split(
                        self.qpos_proj1(qpos_vecotr).reshape(B, C * 2, 1, 1), [C, C], 1
                    )
                    tensor = (1 + gamma) * tensor + beta
                elif i == 4:
                    beta, gamma = torch.split(
                        self.qpos_proj2(qpos_vecotr).reshape(B, C * 2, 1, 1), [C, C], 1
                    )
                    tensor = (1 + gamma) * tensor + beta
                elif i == 5:
                    beta, gamma = torch.split(
                        self.qpos_proj3(qpos_vecotr).reshape(B, C * 2, 1, 1), [C, C], 1
                    )
                    tensor = (1 + gamma) * tensor + beta
                elif i == 6:
                    beta, gamma = torch.split(
                        self.qpos_proj4(qpos_vecotr).reshape(B, C * 2, 1, 1), [C, C], 1
                    )
                    tensor = (1 + gamma) * tensor + beta
            if self.instruction_vector_dim:
                B, C, H, W = tensor.shape
                if i == 2:
                    beta, gamma = torch.split(
                        self.instruction_proj1(instruction_vector).reshape(B, C * 2, 1, 1), [C, C], 1
                    )
                    tensor = (1 + gamma) * tensor + beta
                elif i == 4:
                    beta, gamma = torch.split(
                        self.instruction_proj2(instruction_vector).reshape(B, C * 2, 1, 1), [C, C], 1
                    )
                    tensor = (1 + gamma) * tensor + beta
                elif i == 5:
                    beta, gamma = torch.split(
                        self.instruction_proj3(instruction_vector).reshape(B, C * 2, 1, 1), [C, C], 1
                    )
                    tensor = (1 + gamma) * tensor + beta
                elif i == 6:
                    beta, gamma = torch.split(
                        self.instruction_proj4(instruction_vector).reshape(B, C * 2, 1, 1), [C, C], 1
                    )
                    tensor = (1 + gamma) * tensor + beta
        return tensor
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 qpos_vector_dim=0,
                 instruction_vector_dim=0):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers,
                         qpos_vector_dim, instruction_vector_dim)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, qpos_vector=None, instruction_vector=None):
        # xs = self[0](tensor_list)
        # out: List[NestedTensor] = []
        # pos = []
        # for name, x in xs.items():
        #     out.append(x)
        #     # position encoding
        #     pos.append(self[1](x).to(x.dtype))
        xs = self[0](tensor_list, qpos_vector, instruction_vector)
        out: List[NestedTensor] = [xs]
        pos = [self[1](xs).to(xs.dtype)]
        return out, pos


# def build_backbone(args):
#     backbone = Backbone(args.backbone, args.lr_backbone > 0, args.masks, args.dilation)
#     model = Joiner(backbone, build_position_encoding(args))
#     model.num_channels = backbone.num_channels
#     return model


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class DepthNet(nn.Module):
    def __init__(self, qpos_vector_dim=0, instruction_vector_dim: int = 0):
        super(DepthNet, self).__init__()
        self.instruction_vector_dim = instruction_vector_dim
        self.qpos_vector_dim = qpos_vector_dim
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
        #                             RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [4, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [4, 1]),
                                    RestNetBasicBlock(256, 256, 1))
        self.num_channels = 256
        # self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
        #                             RestNetBasicBlock(512, 512, 1))

        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #
        # self.fc = nn.Linear(512, 10)
        if instruction_vector_dim:
            self.instruction_proj1 = nn.Linear(instruction_vector_dim, 64 * 2)
            self.instruction_proj2 = nn.Linear(instruction_vector_dim, 128 * 2)
            # self.instruction_proj3 = nn.Linear(instruction_vector_dim, 256 * 2)
        if qpos_vector_dim:
            self.qpos_proj1 = nn.Linear(qpos_vector_dim, 64 * 2)
            self.qpos_proj2 = nn.Linear(qpos_vector_dim, 128 * 2)
            # self.instruction_proj3 = nn.Linear(instruction_vector_dim, 256 * 2)

    def forward(self, x, qpos_vector=None, instruction_vector=None):
        out = self.conv1(x)
        if self.qpos_vector_dim:
            B, C, H, W = out.shape
            beta, gamma = torch.split(
                self.qpos_proj1(qpos_vector).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            out = (1 + gamma) * out + beta
        if self.instruction_vector_dim:
            B, C, H, W = out.shape
            beta, gamma = torch.split(
                self.instruction_proj1(instruction_vector).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            out = (1 + gamma) * out + beta

        # out = self.layer1(out)

        out = self.layer2(out)
        if self.qpos_vector_dim:
            B, C, H, W = out.shape
            beta, gamma = torch.split(
                self.qpos_proj2(qpos_vector).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            out = (1 + gamma) * out + beta
        if self.instruction_vector_dim:
            B, C, H, W = out.shape
            beta, gamma = torch.split(
                self.instruction_proj2(instruction_vector).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            out = (1 + gamma) * out + beta

        out = self.layer3(out)
        # if self.qpos_vector_dim:
        #     B, C, H, W = out.shape
        #     beta, gamma = torch.split(
        #         self.qpos_proj3(qpos_vector).reshape(B, C * 2, 1, 1), [C, C], 1
        #     )
        #     out = (1 + gamma) * out + beta
        # if self.instruction_vector_dim:
        #     B, C, H, W = out.shape
        #     beta, gamma = torch.split(
        #         self.instruction_proj3(instruction_vector).reshape(B, C * 2, 1, 1), [C, C], 1
        #     )
        #     out = (1 + gamma) * out + beta
        # out = self.layer4(out)
        # out = self.avgpool(out)
        # out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        return out


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 qpos_vector_dim=0,
                 instruction_vector_dim: int = 0,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        # cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        # cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        self.qpos_vector_dim = qpos_vector_dim
        self.instruction_vector_dim = instruction_vector_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3])
        )
        if instruction_vector_dim:
            self.instruction_proj1 = nn.Linear(instruction_vector_dim, 64 * 2)
            self.instruction_proj2 = nn.Linear(instruction_vector_dim, 128 * 2)
            self.instruction_proj3 = nn.Linear(instruction_vector_dim, 512 * 2)
        if qpos_vector_dim:
            self.qpos_proj1 = nn.Linear(qpos_vector_dim, 64 * 2)
            self.qpos_proj2 = nn.Linear(qpos_vector_dim, 128 * 2)
            self.qpos_proj3 = nn.Linear(qpos_vector_dim, 512 * 2)

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x, qpos_vector=None, instruction_vector=None):
        x = self.mlp1(x)
        if self.qpos_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.qpos_proj1(qpos_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        if self.instruction_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.instruction_proj1(instruction_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        x = self.mlp2(x)
        if self.qpos_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.qpos_proj2(qpos_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        if self.instruction_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.instruction_proj2(instruction_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        x = self.mlp3(x)
        if self.qpos_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.qpos_proj3(qpos_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        if self.instruction_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.instruction_proj3(instruction_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 qpos_vector_dim=0,
                 instruction_vector_dim: int = 0,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        # cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        # cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')

        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
        self.instruction_vector_dim = instruction_vector_dim
        self.qpos_vector_dim = qpos_vector_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU()
        )
        if instruction_vector_dim:
            self.instruction_proj1 = nn.Linear(instruction_vector_dim, 64 * 2)
            self.instruction_proj2 = nn.Linear(instruction_vector_dim, 128 * 2)
            self.instruction_proj3 = nn.Linear(instruction_vector_dim, 256 * 2)
        if qpos_vector_dim:
            self.qpos_proj1 = nn.Linear(qpos_vector_dim, 64 * 2)
            self.qpos_proj2 = nn.Linear(qpos_vector_dim, 128 * 2)
            self.qpos_proj3 = nn.Linear(qpos_vector_dim, 256 * 2)

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)

    def forward(self, x, qpos_vector=None, instruction_vector=None):
        x = self.mlp1(x)
        if self.qpos_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.qpos_proj1(qpos_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        if self.instruction_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.instruction_proj1(instruction_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        x = self.mlp2(x)
        if self.qpos_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.qpos_proj2(qpos_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        if self.instruction_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.instruction_proj2(instruction_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        x = self.mlp3(x)
        if self.qpos_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.qpos_proj3(qpos_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        if self.instruction_vector_dim:
            x = x.permute(0, 2, 1)
            B, C, N = x.shape
            beta, gamma = torch.split(
                self.instruction_proj3(instruction_vector).reshape(B, C * 2, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
            x = x.permute(0, 2, 1)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()

    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()
