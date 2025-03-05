#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from cleandiffuser.nn_condition import EarlyConvViTMultiViewImageCondition
from .backbone import PointNetEncoderXYZRGB
import cv2
class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)
        # num_actor_obs = num_actor_obs - 57600 + 64
        # num_critic_obs = num_critic_obs - 57600 + 64
        # num_actor_obs = num_actor_obs - 115200 + 64
        # num_critic_obs = num_critic_obs - 115200 + 64
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        # self.actor_backbone = EarlyConvViTMultiViewImageCondition(image_sz=((120, 160), ), in_channels=(3, ),
        #                                                           d_model=64, nhead=4,
        #                                                           qpos_vector_dim=None,
        #                                                           instruction_vector_dim=None)
        # self.actor_backbone = PointNetEncoderXYZRGB(in_channels=6, out_channels=64, use_layernorm=True, final_norm="layernorm",
        #                                             qpos_vector_dim=None,
        #                                             instruction_vector_dim=None)
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        # self.critic_backbone = EarlyConvViTMultiViewImageCondition(image_sz=((120, 160), ), in_channels=(3, ),
        #                                                            d_model=64, nhead=4,
        #                                                            qpos_vector_dim=None,
        #                                                            instruction_vector_dim=None)
        # self.actor_backbone = PointNetEncoderXYZRGB(in_channels=6, out_channels=64, use_layernorm=True, final_norm="layernorm",
        #                                             qpos_vector_dim=None,
        #                                             instruction_vector_dim=None)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # color = observations[:, -57600:]
        # observations = observations[:, :-57600]
        # condition = {"image": [], "qpos": None, "instruction": None}
        # condition["image"].append(torch.einsum('k h w c -> k c h w', color.reshape((-1, 120, 160, 3))).unsqueeze(dim=1))
        # # cv2.imwrite("/home/agilex/tmp1.png", color.reshape((-1, 120, 160, 3))[0].clone().detach().cpu().numpy())
        # observations = torch.concatenate([observations, self.actor_backbone(condition)], -1)
        # pc = observations[:, -115200:]
        # observations = observations[:, :-115200]
        # B = observations.shape[0]
        # observations = torch.concatenate([observations, self.actor_backbone(pc.reshape(B, -1, 6))], -1)
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # color = observations[:, -57600:]
        # observations = observations[:, :-57600]
        # condition = {"image": [], "qpos": None, "instruction": None}
        # condition["image"].append(torch.einsum('k h w c -> k c h w', color.reshape((-1, 120, 160, 3))).unsqueeze(dim=1))
        # observations = torch.concatenate([observations, self.actor_backbone(condition)], -1)
        # pc = observations[:, -115200:]
        # observations = observations[:, :-115200]
        # B = observations.shape[0]
        # observations = torch.concatenate([observations, self.actor_backbone(pc.reshape(B, -1, 6))], -1)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        # color = critic_observations[:, -57600:]
        # critic_observations = critic_observations[:, :-57600]
        # condition = {"image": [], "qpos": None, "instruction": None}
        # condition["image"].append(torch.einsum('k h w c -> k c h w', color.reshape((-1, 120, 160, 3))).unsqueeze(dim=1))
        # critic_observations = torch.concatenate([critic_observations, self.actor_backbone(condition)], -1)
        # pc = critic_observations[:, -115200:]
        # critic_observations = critic_observations[:, :-115200]
        # B = critic_observations.shape[0]
        # critic_observations = torch.concatenate([critic_observations, self.actor_backbone(pc.reshape(B, -1, 6))], -1)
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
